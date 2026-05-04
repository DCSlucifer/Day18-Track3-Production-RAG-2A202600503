"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if t}


def _is_model_cached(model_name: str) -> bool:
    """True nếu cả weights lẫn tokenizer của model đều đã có trong HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    weight_candidates = ("model.safetensors", "pytorch_model.bin", "model.onnx")
    has_weights = any(
        try_to_load_from_cache(repo_id=model_name, filename=f) not in (None, False)
        for f in weight_candidates
    )
    has_tokenizer = any(
        try_to_load_from_cache(repo_id=model_name, filename=f) not in (None, False)
        for f in ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model")
    )
    return has_weights and has_tokenizer


class CrossEncoderReranker:
    """BGE reranker khi tải được; fallback lexical scorer cho offline/tests."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fallback: bool | None = None):
        self.model_name = model_name
        self._model = None
        self._kind = None  # "flag", "ce", "fallback"
        self._use_fallback = use_fallback  # None → auto, True → force fallback

    def _load_model(self):
        if self._model is not None or self._kind == "fallback":
            return self._model
        if self._use_fallback is True:
            self._kind = "fallback"
            return None

        # Tránh download model trong tests/CI: chỉ tải nếu cache đã có hoặc bật flag
        force_download = os.environ.get("LAB18_DOWNLOAD_RERANKER") == "1"
        if not force_download and not _is_model_cached(self.model_name):
            self._kind = "fallback"
            return None

        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_name, use_fp16=True)
            self._kind = "flag"
            return self._model
        except Exception:
            pass
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._kind = "ce"
            return self._model
        except Exception:
            self._kind = "fallback"
            self._model = None
            return None

    def _score_pair(self, query: str, doc_text: str) -> float:
        """Lexical fallback: kết hợp jaccard + overlap với độ dài."""
        q = _tokens(query)
        d = _tokens(doc_text)
        if not q or not d:
            return 0.0
        inter = len(q & d)
        union = len(q | d) or 1
        jaccard = inter / union
        overlap = inter / len(q)
        return 0.5 * jaccard + 0.5 * overlap

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]

        if self._kind == "flag":
            try:
                scores = model.compute_score(pairs)
                if not isinstance(scores, list):
                    scores = [scores]
            except Exception:
                scores = [self._score_pair(q, d) for q, d in pairs]
        elif self._kind == "ce":
            try:
                scores = list(model.predict(pairs))
            except Exception:
                scores = [self._score_pair(q, d) for q, d in pairs]
        else:
            scores = [self._score_pair(q, d) for q, d in pairs]

        ranked = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)[:top_k]
        results: list[RerankResult] = []
        for i, (score, doc) in enumerate(ranked):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional — fallback to lexical if not installed."""
    def __init__(self):
        self._model = None
        try:
            from flashrank import Ranker
            self._model = Ranker()
        except Exception:
            self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        if self._model is None:
            return CrossEncoderReranker(use_fallback=True).rerank(query, documents, top_k=top_k)
        from flashrank import RerankRequest
        passages = [{"id": i, "text": d.get("text", "")} for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        out: list[RerankResult] = []
        for i, r in enumerate(results[:top_k]):
            idx = r.get("id", i)
            doc = documents[idx]
            out.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(r.get("score", 0.0)),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return out


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs (ms)."""
    times: list[float] = []
    for _ in range(max(n_runs, 1)):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_runs": len(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
