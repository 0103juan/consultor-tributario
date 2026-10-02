"""Evaluation against data/golden.jsonl.

    python evaluate.py               retrieval ablation; local models only, no API calls
    python evaluate.py --generation  end-to-end run with Claude (rewrite, answer, hallucination check)

A retrieval hit means more than "the right article": the retrieved chunk must contain the
sentence that answers the question (the `evidence` string), because long articles are split
into many chunks and only one of them holds the answer.
"""

import json
import sys
import unicodedata
from collections import Counter

from pydantic import BaseModel

from pipeline import ABSTAIN, LLM, answer
from retrieval import DATA, MODES, RERANK_MODEL, Chunk, Index

GOLDEN = DATA / "golden.jsonl"
USD_PER_MTOK = {"input_tokens": 2.00, "output_tokens": 10.00}  # claude-sonnet-5-5; thinking bills as output
RERANKERS = ("BAAI/bge-reranker-base", RERANK_MODEL)


def plain(text: str) -> str:
    return unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode()


def is_hit(chunk: Chunk, row: dict) -> bool:
    return chunk.article == row["article"] and plain(row["evidence"]) in plain(chunk.text)


def retrieval_ablation(index: Index, golden: list[dict]) -> dict[str, dict[str, float]]:
    """Rank of the evidence chunk under each retrieval mode, on the raw question (no rewriting)."""
    rows = [row for row in golden if row["article"] and "history" not in row]
    configurations = [(mode, {"mode": mode}) for mode in MODES[:2]] + [
        (f"hybrid+rerank ({model.split('/')[-1]})", {"mode": "hybrid+rerank", "rerank_model": model})
        for model in RERANKERS]
    table = {}
    for name, options in configurations:
        ranks = []
        for row in rows:
            hits = [is_hit(chunk, row) for chunk in index.search([row["question"]], k=5, **options)]
            ranks.append(hits.index(True) + 1 if True in hits else None)
        table[name] = {
            "hit@1": sum(r == 1 for r in ranks) / len(rows),
            "hit@5": sum(r is not None for r in ranks) / len(rows),
            "MRR@5": sum(1 / r for r in ranks if r) / len(rows),
        }
    return table


def metered(client) -> Counter:
    """Total the token usage of every call made through this client."""
    usage = Counter()
    for name in ("create", "parse"):
        call = getattr(client.beta.messages, name)

        def wrapper(*args, _call=call, **kwargs):
            response = _call(*args, **kwargs)
            usage.update(calls=1, input_tokens=response.usage.input_tokens,
                         output_tokens=response.usage.output_tokens)
            return response

        setattr(client.beta.messages, name, wrapper)
    return usage


class Grade(BaseModel):
    correct: bool


def is_correct(client, question: str, reference: str, candidate: str) -> bool:
    response = client.beta.messages.parse(
        **LLM, output_config={"effort": "low"}, output_format=Grade,
        system="Califica una respuesta candidata contra la respuesta de referencia. Es correcta si afirma los "
               "mismos hechos que la referencia; el detalle adicional no importa, una contradicción o la falta "
               "de un hecho clave sí.",
        messages=[{"role": "user", "content":
                   f"Pregunta: {question}\nReferencia: {reference}\nCandidata: {candidate}"}])
    return bool(response.parsed_output and response.parsed_output.correct)


def generation_eval(client, index: Index, golden: list[dict]) -> dict[str, float]:
    totals = {"retrieved": 0, "cited": 0, "correct": 0, "groundedness": 0.0}
    answerable = sum(1 for row in golden if row["article"])
    for row in golden:
        result = answer(client, index, row["question"], row.get("history", ""))
        retrieved = cited = False
        if row["article"] is None:  # out-of-scope question: the only right answer is to abstain
            correct = result.answer == ABSTAIN
        else:
            correct = is_correct(client, row["question"], row["answer"], result.answer)
            retrieved = any(is_hit(chunk, row) for chunk in result.retrieved)
            cited = any(is_hit(chunk, row) for chunk in result.sources)
        totals["retrieved"] += retrieved
        totals["cited"] += cited
        totals["correct"] += correct
        totals["groundedness"] += result.groundedness
        if not correct or result.unsupported:
            print(f"  FAIL {row['question']!r}\n       -> {result.answer}\n       evidence retrieved: {retrieved}, "
                  f"articles: {[c.article for c in result.retrieved]}\n       unsupported: {result.unsupported}")
    return {
        "evidence retrieved": totals["retrieved"] / answerable,
        "evidence cited": totals["cited"] / answerable,
        "answer correct": totals["correct"] / len(golden),
        "groundedness": totals["groundedness"] / len(golden),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # piped output on Windows defaults to cp1252, which has no "≈"
    golden =[json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    index = Index.load()
    if "--generation" in sys.argv:
        import anthropic
        client = anthropic.Anthropic()
        usage = metered(client)
        for metric, value in generation_eval(client, index, golden).items():
            print(f"{metric:<22} {value:.1%}")
        cost = sum(usage[kind] * usd / 1e6 for kind, usd in USD_PER_MTOK.items())
        print(f"{len(golden)} questions, {dict(usage)}, ${cost:.2f} (${cost / len(golden):.4f} per question)")
    else:
        print(f"{len(index.chunks)} chunks from {len(index.by_article)} articles")
        print(f"{'mode':<50}{'hit@1':>8}{'hit@5':>8}{'MRR@5':>8}")
        for mode, metrics in retrieval_ablation(index, golden).items():
            print(f"{mode:<50}" + "".join(f"{value:>8.3f}" for value in metrics.values()))
