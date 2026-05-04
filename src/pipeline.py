"""Production RAG Pipeline — M1+M2+M3+M4+M5 với LLM generation và latency tracking."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch, SearchResult
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


_LATENCY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "reports", "latency_breakdown.json")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _llm_answer(query: str, contexts: list[str]) -> str:
    """Sinh câu trả lời với OpenAI; fallback: ghép câu đầu mỗi context."""
    if not contexts:
        return "Không tìm thấy thông tin trong tài liệu."

    context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    if OPENAI_API_KEY and _env_flag("LAB18_USE_OPENAI_GENERATION"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": (
                        "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu nội bộ. "
                        "Chỉ sử dụng thông tin trong CONTEXT để trả lời. "
                        "Trả lời ngắn gọn, đủ ý, bằng tiếng Việt. "
                        "Nếu CONTEXT không chứa câu trả lời, nói: 'Không tìm thấy thông tin trong tài liệu.'")},
                    {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nCÂU HỎI: {query}\n\nTRẢ LỜI:"},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[LLM] OpenAI error: {e} — fallback to extractive answer")

    # Fallback: lấy câu đầu của context đầu tiên
    return contexts[0]


def _attach_parent(results: list[SearchResult], parent_index: dict) -> list[str]:
    """Sau khi rerank child → trả parent để tăng faithfulness/recall."""
    seen = set()
    parents: list[str] = []
    for r in results:
        pid = r.metadata.get("parent_id")
        text = parent_index.get(pid, r.text) if pid else r.text
        if text in seen:
            continue
        seen.add(text)
        parents.append(text)
    return parents or [r.text for r in results]


def build_pipeline(verbose: bool = True):
    """Build production RAG pipeline. Trả search, reranker, parent_index, latency dict."""
    latency = {}
    if verbose:
        print("=" * 60)
        print("PRODUCTION RAG PIPELINE")
        print("=" * 60)

    # Step 1: Load & Chunk (M1)
    if verbose:
        print("\n[1/4] Chunking documents (M1: hierarchical)...")
    t0 = _now_ms()
    docs = load_documents()
    parents_all, children_all = [], []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parents_all.extend(parents)
        children_all.extend(children)
    parent_index = {p.metadata["parent_id"]: p.text for p in parents_all}
    raw_chunks = [
        {"text": c.text, "metadata": {**c.metadata, "parent_id": c.parent_id}}
        for c in children_all
    ]
    latency["chunking_ms"] = round(_now_ms() - t0, 2)
    if verbose:
        print(f"  {len(parents_all)} parents · {len(raw_chunks)} children "
              f"từ {len(docs)} documents · {latency['chunking_ms']:.1f} ms")

    # Step 2: Enrichment (M5)
    if verbose:
        print("\n[2/4] Enriching chunks (M5: contextual + HyQA + metadata)...")
    t0 = _now_ms()
    enriched = enrich_chunks(raw_chunks, methods=["contextual", "hyqa", "metadata"])
    latency["enrichment_ms"] = round(_now_ms() - t0, 2)

    if enriched:
        index_chunks = []
        for orig, e in zip(raw_chunks, enriched):
            md = {**orig["metadata"], **(e.auto_metadata or {})}
            md["parent_id"] = orig["metadata"].get("parent_id")
            index_chunks.append({"text": e.enriched_text, "metadata": md})
        if verbose:
            print(f"  Enriched {len(enriched)} chunks · {latency['enrichment_ms']:.1f} ms")
    else:
        index_chunks = raw_chunks
        if verbose:
            print("  ⚠️  Skipped enrichment — using raw chunks")

    # Step 3: Index (M2)
    if verbose:
        print("\n[3/4] Indexing (BM25 + Dense)...")
    t0 = _now_ms()
    search = HybridSearch()
    search.index(index_chunks)
    latency["indexing_ms"] = round(_now_ms() - t0, 2)
    if verbose:
        print(f"  Indexed {len(index_chunks)} chunks · {latency['indexing_ms']:.1f} ms")

    # Step 4: Reranker (M3)
    if verbose:
        print("\n[4/4] Loading reranker (M3)...")
    t0 = _now_ms()
    reranker = CrossEncoderReranker()
    reranker._load_model()  # warm load
    latency["reranker_load_ms"] = round(_now_ms() - t0, 2)

    return search, reranker, parent_index, latency


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              parent_index: dict | None = None, latency: dict | None = None) -> tuple[str, list[str]]:
    """Run single query: hybrid search → rerank → parent attach → LLM."""
    parent_index = parent_index or {}
    latency = latency if latency is not None else {}

    t0 = _now_ms()
    results = search.search(query)
    latency.setdefault("search_ms", []).append(round(_now_ms() - t0, 2))

    docs_for_rerank = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = _now_ms()
    reranked = reranker.rerank(query, docs_for_rerank, top_k=RERANK_TOP_K)
    latency.setdefault("rerank_ms", []).append(round(_now_ms() - t0, 2))

    # Lấy contexts: ưu tiên parent của child sau rerank
    if reranked:
        rerank_results_as_search = [
            SearchResult(text=r.text, score=r.rerank_score, metadata=r.metadata, method="rerank")
            for r in reranked
        ]
        contexts = _attach_parent(rerank_results_as_search, parent_index)
    else:
        contexts = [r.text for r in results[:RERANK_TOP_K]]

    t0 = _now_ms()
    answer = _llm_answer(query, contexts)
    latency.setdefault("generation_ms", []).append(round(_now_ms() - t0, 2))

    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker,
                      parent_index: dict, latency: dict):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, parent_index, latency)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:60]}")

    print("\n[Eval] Running RAGAS...")
    t0 = _now_ms()
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    latency["evaluation_ms"] = round(_now_ms() - t0, 2)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        marker = "✓" if s >= 0.75 else "✗"
        print(f"  {marker} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures, path="reports/ragas_report.json")

    _save_latency(latency, len(test_set))
    return results


def _save_latency(latency: dict, n_questions: int) -> None:
    """Tổng hợp latency từng bước (ms) và ghi reports/latency_breakdown.json."""
    os.makedirs(os.path.dirname(_LATENCY_PATH), exist_ok=True)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0, "n": 0}
        s = sorted(values)
        return {
            "avg_ms": round(sum(values) / len(values), 2),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
            "p50_ms": round(s[len(s) // 2], 2),
            "n": len(values),
        }

    breakdown = {
        "num_questions": n_questions,
        "build_phase_ms": {
            "chunking": latency.get("chunking_ms", 0),
            "enrichment": latency.get("enrichment_ms", 0),
            "indexing": latency.get("indexing_ms", 0),
            "reranker_load": latency.get("reranker_load_ms", 0),
        },
        "per_query_ms": {
            "search": _stats(latency.get("search_ms", [])),
            "rerank": _stats(latency.get("rerank_ms", [])),
            "generation": _stats(latency.get("generation_ms", [])),
        },
        "evaluation_ms": latency.get("evaluation_ms", 0),
    }
    with open(_LATENCY_PATH, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, ensure_ascii=False, indent=2)
    print(f"\nLatency breakdown saved to {_LATENCY_PATH}")
    print("  build:", breakdown["build_phase_ms"])
    print("  per-query avg (ms):", {k: v["avg_ms"] for k, v in breakdown["per_query_ms"].items()})


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_index, latency = build_pipeline()
    evaluate_pipeline(search, reranker, parent_index, latency)
    print(f"\nTotal: {time.time() - start:.1f}s")
