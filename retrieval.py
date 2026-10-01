"""Retrieval over the Estatuto Tributario: article-aware chunks, BM25 + dense, RRF, cross-encoder rerank."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Chosen by measurement (see README). Its licence is CC BY-NC 4.0: fine for a portfolio, not for a product.
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
MODES = ("dense", "hybrid", "hybrid+rerank")
ARTICLE_REF = re.compile(r"\bart(?:[ií]culos?|s?\.)\s*(\d+(?:-\d+)?)", re.I)

STOPWORDS = frozenset(
    "a al algo ante como con cual cuales cuando cuanto cuanta cuantos cuantas de del desde donde el ella en entre "
    "era es esa ese eso esta este esto estan hay la las le les lo los me mi mas no o para pero por que quien quienes "
    "se segun ser si sin sobre son su sus tiene tienen tengo un una uno unos unas y ya yo debo debe puedo puede".split())


@dataclass(frozen=True)
class Chunk:
    id: str  # "240" or "240#3" for the third part of a long article
    article: str
    title: str  # "Artículo 240. TARIFA GENERAL PARA PERSONAS JURÍDICAS"
    text: str  # title + body: the title travels with the text into the embedding, BM25 and the prompt
    url: str


def chunk_article(article: dict, max_words: int = 220) -> list[Chunk]:
    """One chunk per article; long articles are split on paragraph boundaries and every part keeps the title."""
    status = "" if article["in_force"] else " (DEROGADO: ya no rige)"
    title = f"Artículo {article['id']}. {article['title']}{status}"
    parts, words = [[]], 0
    for paragraph in article["text"].split("\n"):
        if parts[-1] and words + len(paragraph.split()) > max_words:
            parts.append([])
            words = 0
        parts[-1].append(paragraph)
        words += len(paragraph.split())
    return [Chunk(article["id"] + (f"#{n + 1}" if n else ""), article["id"], title,
                  f"{title}\n\n" + "\n".join(part), article["url"])
            for n, part in enumerate(parts)]


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (the source writes both ARTICULO and ARTÍCULO), drop stopwords."""
    plain = unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode()
    return [t for t in re.findall(r"[a-z0-9]+", plain) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in docs]
        self.lengths = [len(doc) for doc in docs]
        self.avg_len = sum(self.lengths) / len(docs)
        df = Counter(term for tf in self.tf for term in tf)
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        return [
            sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                for t in query if t in tf)
            for tf, length in zip(self.tf, self.lengths)
        ]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: merges rankings using ranks only, so BM25 and cosine scores never need calibrating."""
    scores = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1 / (k + rank)
    return [item for item, _ in scores.most_common()]


@lru_cache
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


@lru_cache
def reranker(model: str = RERANK_MODEL):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model)


class Index:
    # ponytail: vectors live in one .npy file and search is brute force; fine for ~2,000 chunks.
    # Move to pgvector/Qdrant when the corpus grows to more statutes than one.
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_article = {}
        for i, chunk in enumerate(chunks):
            self.by_article.setdefault(chunk.article, []).append(i)
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        # Embedding the statute takes minutes on a CPU, so the vectors are cached and keyed by content.
        digest = hashlib.sha256((EMBED_MODEL + "".join(c.text for c in chunks)).encode()).hexdigest()[:16]
        cache = DATA / f"vectors-{digest}.npy"
        if cache.exists():
            self.vectors = np.load(cache)
        else:
            self.vectors = np.array(list(embedder().passage_embed([c.text for c in chunks])), dtype=np.float32)
            self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
            np.save(cache, self.vectors)

    @classmethod
    def load(cls, path: Path = DATA / "articles.jsonl") -> "Index":
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls([chunk for line in lines for chunk in chunk_article(json.loads(line))])

    def _dense(self, query: str, n: int) -> list[int]:
        vector = next(iter(embedder().query_embed(query)))
        return np.argsort(-(self.vectors @ (vector / np.linalg.norm(vector))))[:n].tolist()

    def _sparse(self, query: str, n: int) -> list[int]:
        scores = self.bm25.scores(tokenize(query))
        return [i for i in np.argsort(scores)[::-1][:n].tolist() if scores[i] > 0]

    def _cited(self, query: str) -> list[int]:
        """Chunks of the articles a query names outright ("artículo 240"): an exact lookup, not a search."""
        return [i for number in ARTICLE_REF.findall(query) for i in self.by_article.get(number, [])]

    def search(self, queries: list[str], k: int = 5, mode: str = "hybrid+rerank", pool: int = 50,
               rerank_model: str = RERANK_MODEL) -> list[Chunk]:
        """Retrieve for every query variant, fuse, then rerank the pool against queries[0] (the canonical question)."""
        rankings = []
        for query in queries:
            rankings.append(self._dense(query, pool))
            if mode != "dense":
                rankings.append(self._sparse(query, pool))
                rankings.append(self._cited(query))
        order = rrf([r for r in rankings if r])[:pool]
        if mode == "hybrid+rerank":
            scores = list(reranker(rerank_model).rerank(queries[0], [self.chunks[i].text for i in order]))
            order = [i for _, i in sorted(zip(scores, order), reverse=True)]
        return [self.chunks[i] for i in order[:k]]
