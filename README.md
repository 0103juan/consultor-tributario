# consultor-tributario

[![CI](https://github.com/0103juan/consultor-tributario/actions/workflows/ci.yml/badge.svg)](https://github.com/0103juan/consultor-tributario/actions/workflows/ci.yml)

Question answering over Colombia's **Estatuto Tributario** (the national tax code) that cites the article behind every statement, refuses when the text does not hold the answer, and is measured against a set of questions whose answers are quoted from the law.

It answers in Spanish; this README is in English.

```
"Presenté la declaración tarde, ¿de cuánto es la sanción?"
      │
 1. Rewrite ────▶ standalone question + 2 variants in statutory vocabulary        (Claude)
      │            "multa" → "sanción", "4x1000" → "gravamen a los movimientos financieros"
      ▼
 2. Retrieve ───▶ dense + BM25 for every variant, plus exact lookup of any         (local)
      │            "artículo 641" named in the question, fused with RRF
      ▼
 3. Rerank ─────▶ multilingual cross-encoder over the 50 fused candidates, keep 5   (local)
      ▼
 4. Answer ─────▶ two or three sentences, each citing its passage and article       (Claude)
      ▼
 5. Verify ─────▶ every claim checked against the passages; an unsupported one      (Claude)
                   triggers one corrective rewrite
```

## The corpus is the project

Tax law is public, but it is not published as data. `ingest.py` downloads the Senate's compilation (37 HTML pages, one request per second) and turns it into 1,301 articles and 238,516 words. What that took:

- **Three anchor layouts for the same thing.** Most articles are marked `<a class="bookmarkaj" name="26">ARTICULO 26. TITLE.</a>`. About 200 use a bare `<A name="714">` wrapped in bold, and a few wrap only the first letter (`<a name="868">A</a>RTICULO 868.`). The first parser found 1,102 articles and looked finished. Counting anchors against parsed articles showed it was missing 199.
- **Editorial apparatus is not law.** Vigencia notes, case-law boxes and "previous wording" headers are stripped, and so is each page's footer, which otherwise ends up inside the last article of the page.
- **Repealed is not the same as "mentions repeal".** 197 articles are no longer in force. Matching the word "derogado" in the leading marker also flagged articles that only lost a paragraph, so the rule is: a repeal marker *and* no text left. Repealed articles stay in the index, labelled, so the system can say "that article was repealed by Ley 1819 de 2016" instead of going silent.
- **19 tables are images.** The income tax brackets (art. 241) and the payroll withholding table (art. 383) are published as GIFs. They are replaced with an explicit marker, and the system is instructed to say the table is not available as text. The alternative, a model reciting tax brackets from memory, is the failure this project exists to prevent.
- **The statute has duplicate numbers.** Two different articles are numbered 623-2, so articles are keyed by anchor, not by number.

The corpus is downloaded to your machine and never committed: the law is public, the compilation belongs to its publisher. `data/meta.json` records the compilation date (15 September 2026 for the numbers below), and the CLI prints it under every answer.

## Results

`data/golden.jsonl` holds 33 questions: 30 single-turn, 2 follow-ups that need their history, and 1 out of scope. Each answerable question names its article and an **evidence quote**. A test asserts that every quote exists verbatim in the article it names, so a reference answer cannot drift from the law.

A retrieval hit is strict: the retrieved chunk must contain the evidence quote. Returning the right article is not enough, because article 240 alone is 3,318 words in 18 chunks.

### Retrieval ablation

Raw questions, no rewriting, local models only (`uv run python evaluate.py`), 1,989 chunks:

| Mode | hit@1 | hit@5 | MRR@5 |
|---|---|---|---|
| dense (multilingual MiniLM) | 0.167 | 0.400 | 0.250 |
| hybrid (dense + BM25 + article lookup, RRF) | 0.267 | 0.533 | 0.355 |
| hybrid + rerank, `bge-reranker-base` | 0.367 | 0.533 | 0.431 |
| hybrid + rerank, `jina-reranker-v2-base-multilingual` | **0.467** | **0.767** | **0.579** |

Where the candidates come from, before any reranking:

| First stage | recall@5 | recall@20 | recall@50 | recall@100 |
|---|---|---|---|---|
| dense | 0.40 | 0.53 | 0.67 | 0.77 |
| BM25 | 0.40 | 0.63 | 0.83 | 0.83 |
| hybrid | 0.53 | 0.83 | 0.90 | 0.97 |

And what a bigger candidate pool buys, with the chosen reranker on a laptop CPU:

| Pool | hit@1 | hit@5 | Seconds per query |
|---|---|---|---|
| 20 | 0.467 | 0.700 | 3.8 |
| 50 (default) | 0.467 | 0.767 | 10.7 |
| 100 | 0.467 | 0.800 | 21.4 |

What I take from this:

- **Vector search alone is not usable here.** It puts the answer first one time in six. People ask "¿cuánto es el 4x1000?" and the law says "gravamen a los movimientos financieros ... cuatro por mil".
- **Hybrid search matters on a real corpus.** In my earlier project on a small synthetic corpus ([enterprise-rag](../enterprise-rag)), adding BM25 made the ranking slightly worse. Here neither retriever gets past 0.63 recall@20 alone and together they reach 0.83.
- **The choice of reranker is worth 23 points of hit@5.** The Chinese-English `bge-reranker-base` adds nothing over hybrid at hit@5; the multilingual one takes it from 0.533 to 0.767.
- **The pool size is a latency decision.** Going from 50 to 100 candidates doubles the wait for one more question out of thirty.
- **A quarter of the questions still miss at 5.** That is the gap the rewriting step is meant to close, and this table does not include it.

### End to end

One real question so far, on 1 October 2026 with `claude-sonnet-5-5`:

```
$ uv run python pipeline.py "¿Cuál es la tarifa general de renta para sociedades?"
Según el artículo 240 del Estatuto Tributario, modificado por el artículo 10 de la Ley 2277 de 2022, la tarifa
general del impuesto sobre la renta para las sociedades nacionales y sus asimiladas, los establecimientos
permanentes de entidades del exterior y las personas jurídicas extranjeras con o sin residencia en el país es
del 35% [1]. El mismo artículo prevé tarifas distintas en casos particulares; por ejemplo, las instituciones
financieras y otras entidades del sector liquidan cinco puntos adicionales durante 2023 a 2027, para un total
del 40% (parágrafo 2) [1].

  fuente: Artículo 240. TARIFA GENERAL PARA PERSONAS JURÍDICAS
          http://www.secretariasenado.gov.co/senado/basedoc/estatuto_tributario_pr010.html#240
  respaldo de las afirmaciones: 100%

Texto según la compilación del Senado actualizada al 15 de septiembre de 2026. Esto no es asesoría tributaria.
```

That is one answer, cited and with every claim supported according to the judge. It is not a measurement. The evaluation over the 30 questions has not been run yet: `uv run python evaluate.py --generation` (needs `ANTHROPIC_API_KEY`) reports whether the evidence was retrieved after rewriting, whether the answer cites it, LLM-graded correctness, groundedness, token usage and cost. Until then, the only measured evidence here is about retrieval.

## Run it

```bash
uv sync
uv run python ingest.py        # about a minute; writes data/articles.jsonl
uv run pytest                  # 11 tests, no model downloads, no API key
uv run python evaluate.py      # retrieval ablation; downloads about 2.4 GB of ONNX models once
uv run python pipeline.py "¿Qué porcentaje del salario es renta exenta?"   # needs ANTHROPIC_API_KEY
```

## Design decisions

- **Chunks follow articles.** Each article is a chunk; long ones are split on paragraph boundaries and every part repeats the article's number and title, so a fragment is never anonymous.
- **Exact lookup next to search.** A question that names "artículo 730" gets that article's chunks injected directly. Without it, dense retrieval never finds an article by number.
- **Accent-insensitive BM25 with Spanish stopwords.** The source writes both ARTICULO and ARTÍCULO.
- **Amounts stay in UVT.** The peso value of the UVT changes every year and is not in the statute, so the generator is told not to convert.
- **One stated inference step is allowed.** In the English project the generator refused to connect "Germany" to a rule about the European Union. Here it may make one evident step if it says so ("una SAS es una sociedad nacional, así que...").
- **The reranker was chosen by measurement, and its licence is a constraint.** `jina-reranker-v2-base-multilingual` is CC BY-NC 4.0: acceptable for a portfolio, not for a commercial product, which would need a licensed or different model.

## Limits, stated plainly

- **This is not tax advice.** It retrieves and quotes one statute. Colombian tax practice also depends on regulatory decrees, DIAN rulings and case law, none of which are in the corpus.
- The text is the compilation as of its last update. A reform published after that date is missing until `ingest.py` is run again, and the system has no notion of "the rule that applied in 2021".
- 19 tables and formulas are images and cannot be answered from.
- The reference answers were written by a software engineer reading the statute, not by a tax lawyer. The evidence quotes guarantee that each one is anchored in the text, not that the interpretation is complete.
- Thirty questions is a development set. One question moves hit@5 by 3.3 points, and I chose the pool size and the reranker while looking at it.
- Book, title and chapter headings are discarded, so a chunk does not know whether it sits in the income tax book or the VAT book.
- Reranking 50 candidates takes about ten seconds on a CPU.

## Layout

```
ingest.py          download and parse the statute into articles
retrieval.py       chunking, BM25, RRF, dense search, article lookup, reranking, cached vectors
pipeline.py        rewrite, generate, judge, and answer() which ties them together
evaluate.py        retrieval ablation and end-to-end evaluation with token and cost accounting
data/golden.jsonl  questions, the article and evidence quote for each, and reference answers
test_consultor.py  parser, chunking and pipeline tests
```
