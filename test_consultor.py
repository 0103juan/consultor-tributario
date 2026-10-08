"""Fast checks with no model downloads and no API calls. `python evaluate.py` is the retrieval-quality check."""

import json

import anthropic
import httpx2
import pytest
from model_gateway import Gateway

import ingest
import pipeline
from evaluate import plain, spend
from pipeline import ABSTAIN, JUDGE_SYSTEM, REWRITE_SYSTEM, Claim
from retrieval import ARTICLE_REF, BM25, DATA, Chunk, chunk_article, rrf, tokenize

PAGE = """<html><body><!--Inicio documento-->
<p style="text-align:center;"><a class=antsig href="x.html">Anterior</a> | <a class=antsig href="y.html">Siguiente</a></p>
<p class="centrado"><a class="bookmarkaj" name="TITULO I">TITULO I.</a></p>
<p><a class="bookmarkaj" name="26">ARTICULO 26. LOS INGRESOS SON BASE DE LA RENTA LIQUIDA.</A> &lt;Fuente original&gt; La renta l&iacute;quida se determina as&iacute;.</p>
<div><a class="caja_vja_encabezado" href="javascript:insRow1()">Notas de Vigencia</a></div>
<table id="Table1" class="caja_vja_v" width="100%"></table>
<p><span class="b_aj">PAR&Aacute;GRAFO.</span> Segundo p&aacute;rrafo.</p>
<p><B><A name="27">ART&Iacute;CULO 27. <span class="i_aj">TARIFAS</span>.</A></B> Seg&uacute;n la tabla:</p>
<p class="centrado"><IMG SRC="../graficas/estatuto_tributario_obj_20.gif" WIDTH="371"/></p>
<table><tr><td>Rango</td><td>Tarifa</td></tr><tr><td>0 a 1.090 UVT</td><td>0%</td></tr></table>
<p><a class="bookmarkaj" name="28">A</A>RTICULO 28. <span class="i_aj"><B>EFECTOS DEL REAJUSTE</B></span><B>. </B>El reajuste produce efecto.</p>
<p><a class="bookmarkaj" name="29">ART&Iacute;CULO 29. VALOR DE LOS INGRESOS.</A> &lt;Art&iacute;culo derogado por el art&iacute;culo 376 de la Ley 1819 de 2016&gt;</p>
<p><a class="bookmarkaj" name="30">ART&Iacute;CULO 30. DIVIDENDOS.</A> &lt;Aparte tachado INEXEQUIBLE&gt; Se entiende por dividendo toda distribuci&oacute;n de utilidades.</p>
<p style="text-align:center;"><a class=antsig href="x.html">Anterior</a> | <a class=antsig href="y.html">Siguiente</a></p>
<p>Pie de p&aacute;gina del compilador</p></body></html>"""


@pytest.fixture
def articles(tmp_path):
    page = tmp_path / "estatuto_tributario_pr001.html"
    page.write_bytes(PAGE.encode("cp1252"))
    return {a["id"]: a for a in ingest.parse(page)}


def test_parser_finds_every_anchor_layout_and_skips_headings(articles):
    assert list(articles) == ["26", "27", "28", "29", "30"]
    assert articles["26"]["title"] == "LOS INGRESOS SON BASE DE LA RENTA LIQUIDA"
    assert articles["27"]["title"] == "TARIFAS" and articles["28"]["title"] == "EFECTOS DEL REAJUSTE"
    assert articles["28"]["text"] == "El reajuste produce efecto."


def test_editorial_notes_and_footer_never_reach_the_text(articles):
    assert articles["26"]["text"] == "<Fuente original> La renta líquida se determina así.\nPARÁGRAFO. Segundo párrafo."
    assert "Pie de página" not in articles["30"]["text"] and "Siguiente" not in articles["30"]["text"]


def test_tables_become_rows_and_images_are_flagged_not_dropped(articles):
    assert "PUBLICADA COMO IMAGEN" in articles["27"]["text"]
    assert "Rango | Tarifa\n0 a 1.090 UVT | 0%" in articles["27"]["text"]


def test_repealed_means_no_text_left_not_just_a_struck_down_phrase(articles):
    assert not articles["29"]["in_force"]
    assert articles["30"]["in_force"]  # one phrase was struck down; the article still has a body


def test_long_articles_split_and_every_part_keeps_the_title():
    article = {"id": "240", "title": "TARIFA GENERAL", "in_force": True, "url": "u",
               "text": "\n".join(" ".join(["palabra"] * 100) for _ in range(3))}
    chunks = chunk_article(article, max_words=220)
    assert [c.id for c in chunks] == ["240", "240#2"] and {c.article for c in chunks} == {"240"}
    assert all(c.text.startswith("Artículo 240. TARIFA GENERAL\n\n") for c in chunks)
    repealed = chunk_article({**article, "in_force": False, "text": "<Artículo derogado>"})[0]
    assert "DEROGADO" in repealed.title


def test_tokenizer_ignores_accents_and_question_words():
    assert tokenize("¿Cuál es la SANCIÓN del artículo 641?") == ["sancion", "articulo", "641"]
    scores = BM25([tokenize(d) for d in ["ARTICULO 641. Extemporaneidad", "Tarifa general", "Sanción mínima"]]
                  ).scores(tokenize("artículo 641"))
    assert scores.index(max(scores)) == 0


def test_article_numbers_in_a_question_are_looked_up_exactly():
    assert ARTICLE_REF.findall("¿Qué dicen el artículo 771-5 y el art. 240 del estatuto?") == ["771-5", "240"]
    assert ARTICLE_REF.findall("¿Cuánto es el IVA en 2024?") == []


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([[1, 2, 3], [2, 1, 4], [2, 5, 1]])[:2] == [2, 1]


class FakeIndex:
    chunks = [Chunk("639", "639", "Artículo 639. SANCIÓN MÍNIMA", "Artículo 639. SANCIÓN MÍNIMA\n\n10 uvt", "u"),
              Chunk("641", "641", "Artículo 641. EXTEMPORANEIDAD", "Artículo 641. EXTEMPORANEIDAD\n\n5%", "u")]

    def search(self, queries):
        return self.chunks


def test_unsupported_claim_triggers_one_corrective_retry(monkeypatch):
    drafts = iter(["La sanción mínima es de 10 UVT [1], unos 500.000 pesos [1].", "La sanción mínima es de 10 UVT [1]."])
    verdicts = iter([[Claim(text="10 UVT", supported=True), Claim(text="unos 500.000 pesos", supported=False)],
                     [Claim(text="10 UVT", supported=True)]])
    corrections = []
    monkeypatch.setattr(pipeline, "rewrite", lambda client, q, history: [q])
    monkeypatch.setattr(pipeline, "generate", lambda c, q, chunks, corr="": corrections.append(corr) or next(drafts))
    monkeypatch.setattr(pipeline, "judge", lambda c, text, chunks: next(verdicts))

    result = pipeline.answer(None, FakeIndex(), "¿sanción mínima?")

    assert result.answer == "La sanción mínima es de 10 UVT [1]."
    assert "unos 500.000 pesos" in corrections[1] and corrections[0] == ""
    assert result.groundedness == 1.0 and [s.article for s in result.sources] == ["639"]


def test_abstention_is_not_judged_and_cites_nothing(monkeypatch):
    monkeypatch.setattr(pipeline, "rewrite", lambda client, q, history: [q])
    monkeypatch.setattr(pipeline, "generate", lambda *a: ABSTAIN)
    monkeypatch.setattr(pipeline, "judge", lambda *a: 1 / 0)

    result = pipeline.answer(None, FakeIndex(), "¿salario mínimo?")

    assert result.answer == ABSTAIN and result.sources == [] and result.groundedness == 1.0


def test_each_stage_names_its_task_and_the_evaluation_reports_spend_per_task(tmp_path):
    def api(request):  # the real SDK over a fake HTTP transport: the answer depends on the stage that asks
        body = json.loads(request.content)
        text = {REWRITE_SYSTEM: '{"standalone": "sanción mínima", "variants": []}',
                JUDGE_SYSTEM: '{"claims": [{"text": "son 10 UVT", "supported": true}]}'}.get(
                    body["system"], "La sanción mínima es de 10 UVT [1].")
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
            "stop_sequence": None, "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 1000, "output_tokens": 100}})

    client = anthropic.Anthropic(api_key="test", http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(api)))
    gateway = Gateway(client, routes={"rewrite": "small"}, cache=tmp_path)

    result = pipeline.answer(gateway, FakeIndex(), "¿sanción mínima?")
    pipeline.answer(gateway, FakeIndex(), "¿sanción mínima?")  # the same question again comes from the cache

    assert result.answer == "La sanción mínima es de 10 UVT [1]." and result.groundedness == 1.0
    assert [call["task"] for call in gateway.calls[:3]] == ["rewrite", "generate", "judge"]
    assert [call["cached"] for call in gateway.calls] == [False] * 3 + [True] * 3
    assert gateway.spent == pytest.approx((1000 * 1 + 100 * 5 + 2 * (1000 * 2 + 100 * 10)) / 1e6)
    report = spend(gateway.calls).splitlines()
    assert report[1].split() == ["rewrite", "2", "1", "2000", "200", "0.0015", "claude-haiku-4-5"]


@pytest.mark.skipif(not (DATA / "articles.jsonl").exists(), reason="run `python ingest.py` first")
def test_every_golden_answer_is_backed_by_the_statute_text():
    """The reference answers are only as good as their evidence: each quote must exist in the article it names."""
    articles = {a["id"]: a for a in map(json.loads, (DATA / "articles.jsonl").read_text(encoding="utf-8").splitlines())}
    golden = [json.loads(line) for line in (DATA / "golden.jsonl").read_text(encoding="utf-8").splitlines()]
    missing = [row["question"] for row in golden
               if row["article"] and plain(row["evidence"]) not in plain(articles[row["article"]]["text"])]
    assert not missing
