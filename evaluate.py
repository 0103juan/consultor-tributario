"""Evaluation against data/golden.jsonl.

    python evaluate.py               retrieval ablation; local models only, no API calls
    python evaluate.py --generation  end-to-end run with Claude (rewrite, answer, hallucination check)
    python evaluate.py --generation --small   the same, with the pipeline on the small model

Model calls go through model-gateway: a repeated run is served from .gateway/cache, every call is written
to .gateway/ledger.jsonl, and a run stops if it spends more than BUDGET_USD.

A retrieval hit means more than "the right article": the retrieved chunk must contain the
sentence that answers the question (the `evidence` string), because long articles are split
into many chunks and only one of them holds the answer.
"""

import json
import sys
import time
import unicodedata
from pathlib import Path

from model_gateway import Gateway
from pydantic import BaseModel

from pipeline import ABSTAIN, answer
from retrieval import DATA, MODES, RERANK_MODEL, Chunk, Index

GOLDEN = DATA / "golden.jsonl"
GATEWAY = Path(__file__).parent / ".gateway"
STAGES = ("rewrite", "generate", "judge")  # the pipeline's own calls; "grade" belongs to the evaluation
BUDGET_USD = 1.00  # per run; the run of 2 October 2026 cost $0.66
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


def spend(calls: list[dict]) -> str:
    """Calls, tokens and USD per task, from the gateway's records. A cached call keeps its tokens and costs nothing."""
    lines = [f"{'task':<10}{'calls':>7}{'cached':>8}{'in tok':>10}{'out tok':>9}{'usd':>9}  models"]
    for task in dict.fromkeys(call["task"] for call in calls):
        rows = [call for call in calls if call["task"] == task]
        lines.append(f"{task:<10}{len(rows):>7}{sum(row['cached'] for row in rows):>8}"
                     f"{sum(row['input_tokens'] for row in rows):>10}{sum(row['output_tokens'] for row in rows):>9}"
                     f"{sum(row['usd'] for row in rows):>9.4f}  {', '.join(sorted({row['model'] for row in rows}))}")
    return "\n".join(lines)


class Grade(BaseModel):
    correct: bool


def is_correct(gateway, question: str, reference: str, candidate: str) -> bool:
    response = gateway.parse(
        task="grade", output_config={"effort": "low"}, output_format=Grade,
        system="Califica una respuesta candidata contra la respuesta de referencia. Es correcta si afirma los "
               "mismos hechos que la referencia; el detalle adicional no importa, una contradicción o la falta "
               "de un hecho clave sí.",
        messages=[{"role": "user", "content":
                   f"Pregunta: {question}\nReferencia: {reference}\nCandidata: {candidate}"}])
    return bool(response.parsed_output and response.parsed_output.correct)


def generation_eval(gateway, index: Index, golden: list[dict]) -> dict[str, float]:
    totals = {"retrieved": 0, "cited": 0, "correct": 0, "groundedness": 0.0}
    answerable = sum(1 for row in golden if row["article"])
    for row in golden:
        result = answer(gateway, index, row["question"], row.get("history", ""))
        retrieved = cited = False
        if row["article"] is None:  # out-of-scope question: the only right answer is to abstain
            correct = result.answer == ABSTAIN
        else:
            correct = is_correct(gateway, row["question"], row["answer"], result.answer)
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
        tier = "small" if "--small" in sys.argv else "large"  # the grader stays on the large model either way
        gateway = Gateway(anthropic.Anthropic(), routes=dict.fromkeys(STAGES, tier),
                          key=f"eval-{tier}-{time.strftime('%Y%m%d-%H%M%S')}", budget_usd=BUDGET_USD,
                          ledger=GATEWAY / "ledger.jsonl", cache=GATEWAY / "cache")
        for metric, value in generation_eval(gateway, index, golden).items():
            print(f"{metric:<22} {value:.1%}")
        print(spend(gateway.calls))
        print(f"{len(golden)} questions on the {tier} tier, {len(gateway.calls)} model calls, "
              f"${gateway.spent:.2f} (${gateway.spent / len(golden):.4f} per question)")
    else:
        print(f"{len(index.chunks)} chunks from {len(index.by_article)} articles")
        print(f"{'mode':<50}{'hit@1':>8}{'hit@5':>8}{'MRR@5':>8}")
        for mode, metrics in retrieval_ablation(index, golden).items():
            print(f"{mode:<50}" + "".join(f"{value:>8.3f}" for value in metrics.values()))
