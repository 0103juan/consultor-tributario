"""Writes data/golden-generated.jsonl: questions a model writes from the articles, kept only when they can be checked.

    PROVIDER=vertex GOOGLE_CLOUD_PROJECT=... python golden.py 270

For each article it samples, the model writes the question a taxpayer would ask, the shortest passage that
answers it, copied word for word, and a reference answer. A row is kept only if that passage really is in one
chunk of the article (so retrieval can be scored as for the hand-written set), the question does not name the
article, and no other row comes from the same article. Nobody has read these rows: the evaluation reports them
apart from the 33 written by hand, and a model wrote both them and the answers they grade.
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from evaluate import GATEWAY, GENERATED, GOLDEN, plain
from pipeline import make_gateway
from retrieval import DATA, Index

SYSTEM = """Escribes preguntas de prueba para un asistente que responde sobre el Estatuto Tributario de Colombia.

Del artículo que te dan, escribe:
- question: la pregunta que haría un contribuyente con sus propias palabras, sin mencionar el número del \
artículo ni copiar su redacción. Debe tener una respuesta concreta en el artículo (un plazo, una tarifa, un \
requisito, quién está obligado, qué pasa si...).
- evidence: el fragmento más corto del artículo que responde la pregunta, copiado exactamente, letra por letra, \
de 3 a 25 palabras.
- answer: la respuesta correcta en una o dos frases.
Si el artículo está derogado o no permite una pregunta así, devuelve skip = true y deja los demás campos vacíos."""


class Row(BaseModel):
    skip: bool = Field(description="true si el artículo no permite una pregunta con respuesta concreta")
    question: str
    evidence: str
    answer: str


def main(wanted: int, seed: int = 7) -> None:
    index = Index.load()
    articles = [json.loads(line) for line in (DATA / "articles.jsonl").read_text(encoding="utf-8").splitlines()]
    taken = {json.loads(line)["article"] for line in GOLDEN.read_text(encoding="utf-8").splitlines()}
    # Articles in force, long enough to say something and short enough to read, not already in the hand-written set.
    pool = [a for a in articles if a["in_force"] and 300 <= len(a["text"]) <= 6000 and a["id"] not in taken]
    random.Random(seed).shuffle(pool)
    gateway = make_gateway(routes={"write": "large"}, key="golden", budget_usd=3.00,
                           ledger=GATEWAY / "ledger.jsonl", cache=GATEWAY / "cache")

    def write(article: dict) -> Row | None:
        return gateway.parse(task="write", system=SYSTEM, output_format=Row, output_config={"effort": "low"},
                             messages=[{"role": "user", "content": f"Artículo {article['id']}. {article['title']}\n\n{article['text']}"}]).parsed_output

    kept, dropped = [], {"skip": 0, "not verbatim": 0, "names the article": 0}
    # Eight calls at a time, in batches of 16 read back in the pool's order: the same seed gives the same rows.
    with GENERATED.open("w", encoding="utf-8") as out, ThreadPoolExecutor(8) as workers:
        batches = (pool[start:start + 16] for start in range(0, len(pool), 16))
        candidates = ((article, row) for batch in batches if len(kept) < wanted
                      for article, row in zip(batch, workers.map(write, batch)))
        for article, row in candidates:
            if len(kept) == wanted:
                break
            if row is None or row.skip:
                dropped["skip"] += 1
                continue
            chunks = [index.chunks[i] for i in index.by_article.get(article["id"], [])]
            if len(plain(row.evidence)) < 8 or not any(plain(row.evidence) in plain(chunk.text) for chunk in chunks):
                dropped["not verbatim"] += 1
                continue
            if article["id"] in row.question or "artículo" in row.question.lower():
                dropped["names the article"] += 1
                continue
            kept.append(article["id"])
            out.write(json.dumps({"question": row.question, "article": article["id"], "evidence": row.evidence,
                                  "answer": row.answer}, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} of {len(kept) + sum(dropped.values())} articles; dropped {dropped}; "
          f"{len(gateway.calls)} calls, ${gateway.spent:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 270)
