"""Download Colombia's Estatuto Tributario from the Senate's public compilation and parse it into articles.

    python ingest.py     ->  data/raw/*.html (cache)  ->  data/articles.jsonl

The legal text is public; the compilation and its editorial notes belong to their publisher,
so the corpus is downloaded to your machine and never committed to this repository.
"""

import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://www.secretariasenado.gov.co/senado/basedoc/"
DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
ARTICLES = DATA / "articles.jsonl"

# Editorial apparatus, not law: the (empty, script-filled) note boxes and their "Notas de Vigencia" headers.
NOTES = re.compile(r'<table[^>]*class="caja_vja[^"]*".*?</table>|<a[^>]*class="caja_vja_encabezado".*?</a>',
                   re.S | re.I)
ANCHOR = re.compile(r'<a\s+(?:class="bookmarkaj"\s+)?name="([^"]+)"\s*>(.*?)</a>', re.S | re.I)
HEADING = re.compile(r"ART[IÍ]CULO[^.]*\.\s*(.*)", re.S | re.I)
FOOTER = re.compile(r"<p[^>]*>\s*<a class=antsig", re.I)
# 19 rate tables and formulas are published as GIFs. They are flagged rather than silently dropped,
# so the system can say "this table is not available as text" instead of answering without it.
IMAGE = " [TABLA O FÓRMULA PUBLICADA COMO IMAGEN: no disponible como texto] "
UPDATED = re.compile(r"ltima actualizaci&oacute;n:\s*([^<(]+?)\s*-?\s*\(")


def download() -> list[Path]:
    """Fetch estatuto_tributario.html, _pr001.html, _pr002.html... until the server says there are no more."""
    RAW.mkdir(parents=True, exist_ok=True)
    pages = []
    for n in range(500):
        name = "estatuto_tributario.html" if n == 0 else f"estatuto_tributario_pr{n:03d}.html"
        path = RAW / name
        if not path.exists():
            request = urllib.request.Request(BASE + name, headers={"User-Agent": "consultor-tributario (research)"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                raise
            time.sleep(1)  # one request per second: this is a public server, not an API
        pages.append(path)
    return pages


def to_text(fragment: str) -> str:
    """HTML fragment -> plain text, one paragraph per line, table rows as 'cell | cell'."""
    fragment = re.sub(r"<img[^>]*graficas/[^>]*>", IMAGE, fragment, flags=re.I)
    fragment = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", fragment, flags=re.I)
    fragment = re.sub(r"</(p|tr|div|li|h\d)>|<br\s*/?>", "\n", fragment, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment)).replace("\xa0", " ")
    lines = (re.sub(r"\s+", " ", line).strip(" |") for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def parse(page: Path) -> list[dict]:
    """Split one page into articles. Everything from one named anchor to the next belongs to the first."""
    source = NOTES.sub("", page.read_bytes().decode("cp1252", errors="replace"))
    footers = [m.start() for m in FOOTER.finditer(source)]
    anchors = list(ANCHOR.finditer(source))
    articles = []
    for anchor, following in zip(anchors, [*anchors[1:], None]):
        if not anchor.group(1)[0].isdigit():  # anchors for Libro/Título/Capítulo headings are named with letters
            continue
        end = following.start() if following else next((f for f in footers if f > anchor.end()), len(source))
        # The anchor usually wraps "ARTICULO 26. TITLE." but sometimes only its first letter, so the
        # title is read from the anchor when it is there and from the running text when it is not.
        inner = HEADING.match(to_text(anchor.group(2)))
        if inner and inner.group(1).strip(" ."):
            title, body = inner.group(1), to_text(source[anchor.end():end])
        else:
            rest = HEADING.match(to_text(source[anchor.start():end]))
            title, _, body = (rest.group(1) if rest else "").partition(". ")
        if title.lstrip().startswith("<"):  # no title at all: what follows the number is the compiler's note
            title, body = "", f"{title}. {body}"
        body = re.split(r"\n-{5,}", body)[0]  # the last article is followed by signatures and footnotes
        articles.append({
            "id": anchor.group(1),  # the anchor name: unique even where the statute repeats a number "(sic)"
            "title": title.strip(" .") or "(sin título)",
            "text": body.strip(),
            # A repealed or struck-down article keeps its number and a compiler's marker, and has no text
            # left: "<Artículo derogado por el artículo 376 de la Ley 1819 de 2016>". An article that only
            # lost a paragraph carries the same words but still has a body, so it stays in force.
            "in_force": not (re.search(r"derogad|INEXEQUIBLE", body, flags=re.I)
                             and len(re.sub(r"<[^>]*>", "", body).split()) < 5),
            "url": f"{BASE}{page.name}#{anchor.group(1)}",
        })
    return articles


if __name__ == "__main__":
    pages = download()
    articles = [article for page in pages for article in parse(page)]
    ARTICLES.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in articles), encoding="utf-8")
    updated = UPDATED.search(pages[0].read_bytes().decode("cp1252", errors="replace"))
    (DATA / "meta.json").write_text(json.dumps({
        "source": BASE + "estatuto_tributario.html",
        "compilation_updated": html.unescape(updated.group(1)) if updated else None,
        "downloaded": time.strftime("%Y-%m-%d"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    words = sum(len(a["text"].split()) for a in articles)
    print(f"{len(pages)} pages -> {len(articles)} articles ({sum(a['in_force'] for a in articles)} in force), {words:,} words")
