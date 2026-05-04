"""Tests for Module 2: Hybrid Search."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m2_search import (
    segment_vietnamese, BM25Search, DenseSearch,
    reciprocal_rank_fusion, SearchResult,
)

CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
    {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
]

def test_segment_returns_string():
    assert isinstance(segment_vietnamese("nghỉ phép năm"), str)

def test_bm25_search():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép", top_k=2)
    assert len(results) > 0 and results[0].method == "bm25"

def test_bm25_relevant_first():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép năm", top_k=2)
    if results:
        assert "nghỉ" in results[0].text.lower() or "12" in results[0].text

def test_rrf_merges():
    a = [SearchResult("doc1", 0.9, {}, "bm25"), SearchResult("doc2", 0.8, {}, "bm25")]
    b = [SearchResult("doc2", 0.95, {}, "dense"), SearchResult("doc3", 0.85, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(merged) > 0 and "doc2" in [r.text for r in merged]

def test_rrf_method():
    a = [SearchResult("d1", 0.9, {}, "bm25")]
    b = [SearchResult("d1", 0.8, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=1)
    if merged:
        assert merged[0].method == "hybrid"


def test_dense_search_falls_back_when_qdrant_returns_empty(monkeypatch):
    class FakeClient:
        def recreate_collection(self, **kwargs):  # noqa: ARG002
            return None

        def upsert(self, **kwargs):  # noqa: ARG002
            return None

        def search(self, **kwargs):  # noqa: ARG002
            return []

    class FakeEncoder:
        def encode(self, texts, show_progress_bar=False):  # noqa: ARG002
            vectors = []
            for text in texts:
                vectors.append([
                    1.0 if "nghỉ" in text.lower() else 0.0,
                    1.0 if "mật" in text.lower() else 0.0,
                ])
            return vectors

    def fake_init_client(self):
        self._client = FakeClient()
        self._using_qdrant = True

    monkeypatch.setattr(DenseSearch, "_init_client", fake_init_client)
    monkeypatch.setattr(DenseSearch, "_get_encoder", lambda self: FakeEncoder())

    dense = DenseSearch()
    dense.index(CHUNKS, collection="test_dense_fallback")
    results = dense.search("nghỉ phép", top_k=1, collection="test_dense_fallback")

    assert len(results) == 1
    assert "nghỉ" in results[0].text.lower()
