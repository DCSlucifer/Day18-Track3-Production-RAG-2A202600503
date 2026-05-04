"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words. underthesea nếu sẵn có; fallback lower."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    seg = segment_vietnamese(text)
    return [t for t in re.split(r"\s+", seg.lower()) if t and re.search(r"\w", t)]


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        self.corpus_tokens = [_tokens(c["text"]) or [c["text"][:20]] for c in self.documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if not self.bm25 or not self.documents:
            return []
        tokens = _tokens(query) or [query.lower()]
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        results = []
        for i in ranked:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    """Dense retrieval qua Qdrant; fallback in-memory cosine khi Qdrant offline."""

    def __init__(self):
        self._encoder = None
        self._client = None
        self._memory: dict[str, list[dict]] = {}  # collection -> list of {vector, text, metadata}
        self._using_qdrant = False
        self._init_client()

    def _init_client(self):
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
            client.get_collections()  # ping
            self._client = client
            self._using_qdrant = True
        except Exception:
            self._client = None
            self._using_qdrant = False

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = _HashEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        if not chunks:
            return
        encoder = self._get_encoder()
        texts = [c["text"] for c in chunks]
        vectors = encoder.encode(texts, show_progress_bar=False) if hasattr(encoder, "encode") else [encoder(t) for t in texts]
        vectors = [list(map(float, v)) for v in vectors]

        self._memory[collection] = [
            {"vector": v, "text": chunks[i]["text"], "metadata": chunks[i].get("metadata", {})}
            for i, v in enumerate(vectors)
        ]

        if self._using_qdrant:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            try:
                self._client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
                )
                points = [
                    PointStruct(id=i, vector=v, payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]})
                    for i, v in enumerate(vectors)
                ]
                self._client.upsert(collection_name=collection, points=points)
                return
            except Exception:
                self._using_qdrant = False  # fall back

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        encoder = self._get_encoder()
        qv = encoder.encode([query], show_progress_bar=False)[0] if hasattr(encoder, "encode") else encoder(query)
        qv = list(map(float, qv))

        if self._using_qdrant:
            try:
                hits = self._client.search(
                    collection_name=collection, query_vector=qv, limit=top_k,
                )
                if hits:
                    return [
                        SearchResult(
                            text=h.payload.get("text", ""),
                            score=float(h.score),
                            metadata={k: v for k, v in h.payload.items() if k != "text"},
                            method="dense",
                        )
                        for h in hits
                    ]
            except Exception:
                pass

        items = self._memory.get(collection, [])
        scored = [
            (_cosine(qv, item["vector"]), item)
            for item in items
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(text=it["text"], score=float(s), metadata=it["metadata"], method="dense")
            for s, it in scored[:top_k]
        ]


def _cosine(a, b) -> float:
    import numpy as np
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) or 1e-9
    return float(np.dot(av, bv) / denom)


class _HashEncoder:
    """Deterministic hash-based fallback embedding (offline)."""
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False):  # noqa: ARG002
        return [self._embed(t) for t in texts]

    def __call__(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        import numpy as np
        v = np.zeros(self.dim, dtype=float)
        for tok in _tokens(text):
            h = hash(tok) % self.dim
            v[h] += 1.0
        n = np.linalg.norm(v) or 1.0
        return (v / n).tolist()


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, r in enumerate(results):
            entry = rrf_scores.setdefault(r.text, {"score": 0.0, "result": r})
            entry["score"] += 1.0 / (k + rank + 1)
    merged = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(text=e["result"].text, score=e["score"],
                     metadata=e["result"].metadata, method="hybrid")
        for e in merged
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
