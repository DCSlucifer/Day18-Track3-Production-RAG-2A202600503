"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
import re
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Heuristic / fallback metrics (no API needed) ─────────


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.split(r"\W+", text or "") if t and len(t) > 1}


def _coverage(reference: str, candidate: str) -> float:
    """Tỷ lệ token của reference xuất hiện trong candidate."""
    rt = _tokens(reference)
    if not rt:
        return 0.0
    ct = _tokens(candidate)
    return len(rt & ct) / len(rt)


def _heuristic_metrics(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """Deterministic fallback gần với RAGAS để tests offline có giá trị hợp lý."""
    ctx_join = " ".join(contexts or [])
    faithfulness = _coverage(answer, ctx_join) if answer.strip() else 0.0
    answer_relevancy = max(_coverage(ground_truth, answer), _coverage(question, answer))
    context_precision = _coverage(ground_truth, ctx_join) if ctx_join else 0.0
    # context_recall: tỷ lệ token GT có trong contexts
    context_recall = _coverage(ground_truth, ctx_join) if ctx_join else 0.0
    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
    }


# ─── RAGAS path ───────────────────────────────────────────


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _try_ragas(questions, answers, contexts, ground_truths) -> dict | None:
    """Chạy RAGAS thật khi có OpenAI key + ragas/datasets cài đặt. Trả None nếu không khả dụng."""
    if not OPENAI_API_KEY:
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
    except Exception:
        return None

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Inject explicit LLM + embeddings để tránh OpenAIEmbeddings.embed_query mismatch
    llm = embeddings = None
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    except Exception as e:
        print(f"[RAGAS] explicit wrapper unavailable: {e}")

    try:
        kwargs = {"dataset": dataset,
                  "metrics": [faithfulness, answer_relevancy, context_precision, context_recall]}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        result = evaluate(**kwargs)
    except Exception as e:
        print(f"[RAGAS] evaluate failed: {e}")
        return None

    try:
        df = result.to_pandas()
    except Exception:
        df = None

    per_question: list[EvalResult] = []
    if df is not None:
        for i, (_, row) in enumerate(df.iterrows()):
            per_question.append(EvalResult(
                question=str(row.get("question", questions[i] if i < len(questions) else "")),
                answer=str(row.get("answer", answers[i] if i < len(answers) else "")),
                contexts=list(row.get("contexts", contexts[i] if i < len(contexts) else []) or []),
                ground_truth=str(row.get("ground_truth", ground_truths[i] if i < len(ground_truths) else "")),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            ))

    def _agg(name: str) -> float:
        if df is not None and name in df.columns:
            vals = [float(v) for v in df[name].tolist() if v is not None and not (isinstance(v, float) and v != v)]
            return sum(vals) / len(vals) if vals else 0.0
        v = result.get(name) if hasattr(result, "get") else None
        return float(v) if v is not None else 0.0

    if not per_question:
        for i in range(len(questions)):
            per_question.append(EvalResult(
                question=questions[i],
                answer=answers[i] if i < len(answers) else "",
                contexts=list(contexts[i] if i < len(contexts) else []),
                ground_truth=ground_truths[i] if i < len(ground_truths) else "",
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            ))

    return {
        "faithfulness": round(_agg("faithfulness"), 4),
        "answer_relevancy": round(_agg("answer_relevancy"), 4),
        "context_precision": round(_agg("context_precision"), 4),
        "context_recall": round(_agg("context_recall"), 4),
        "per_question": per_question,
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation; fallback heuristic khi RAGAS/OpenAI không sẵn có."""
    n = len(questions)
    contexts = contexts or [[] for _ in range(n)]
    answers = answers or ["" for _ in range(n)]
    ground_truths = ground_truths or ["" for _ in range(n)]

    if n == 0:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    if _env_flag("LAB18_USE_REAL_RAGAS"):
        real = _try_ragas(questions, answers, contexts, ground_truths)
        if real is not None:
            real["per_question"] = real.get("per_question", [])
            return real

    per_question: list[EvalResult] = []
    sums = {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0}
    for i in range(n):
        m = _heuristic_metrics(questions[i], answers[i], contexts[i], ground_truths[i])
        per_question.append(EvalResult(
            question=questions[i], answer=answers[i],
            contexts=list(contexts[i] or []), ground_truth=ground_truths[i],
            **m,
        ))
        for k, v in m.items():
            sums[k] += v

    aggregate = {k: round(sums[k] / n, 4) for k in sums}
    return {**aggregate, "per_question": per_question}


# ─── Failure analysis ─────────────────────────────────────


_DIAGNOSIS_TABLE = [
    # threshold, metric, diagnosis, fix
    ("faithfulness", 0.85, "LLM hallucinating — answer không bám sát context",
     "Tighten prompt (yêu cầu trích dẫn), giảm temperature, dùng parent context lớn hơn"),
    ("context_recall", 0.75, "Missing relevant chunks — không tìm được đoạn chứa câu trả lời",
     "Improve chunking (hierarchical/structure), enrichment với HyQA, tăng top_k BM25 + dense"),
    ("context_precision", 0.75, "Quá nhiều chunks không liên quan trong top-k",
     "Add cross-encoder rerank, metadata filter theo category, giảm child_size"),
    ("answer_relevancy", 0.80, "Answer không match câu hỏi — lệch chủ đề",
     "Cải prompt template, thêm chain-of-thought, chuẩn hóa câu hỏi (query rewriting)"),
]


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        avg = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4.0
        scored.append((avg, r))
    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    failures: list[dict] = []
    for avg, r in bottom:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        diagnosis, fix = _diagnose(metrics)
        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
            "metrics": metrics,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "error_tree": _error_tree(metrics),
        })
    return failures


def _diagnose(metrics: dict) -> tuple[str, str]:
    """Map metrics → diagnosis + fix theo Error Tree."""
    for metric, threshold, diagnosis, fix in _DIAGNOSIS_TABLE:
        if metrics.get(metric, 1.0) < threshold:
            return diagnosis, fix
    return ("Scores OK nhưng chưa đạt mục tiêu — tinh chỉnh prompt và corpus.",
            "Mở rộng test set, tăng chất lượng corpus, A/B test prompt template")


def _error_tree(metrics: dict) -> list[str]:
    """Walkthrough Output → Context → Query → Root cause."""
    steps = []
    if metrics["faithfulness"] < 0.85:
        steps.append("Output sai: faithfulness thấp → answer chứa thông tin ngoài context")
    else:
        steps.append("Output OK: bám context")
    if metrics["context_precision"] < 0.75 or metrics["context_recall"] < 0.75:
        steps.append("Context yếu: precision/recall thấp → retrieval miss hoặc nhiễu")
    else:
        steps.append("Context OK")
    if metrics["answer_relevancy"] < 0.80:
        steps.append("Query/answer mismatch → cần query rewriting hoặc prompt rõ hơn")
    else:
        steps.append("Query rewrite OK")
    return steps


# ─── Save report ──────────────────────────────────────────


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    aggregate = {k: v for k, v in results.items() if k != "per_question"}
    per_q = []
    for r in results.get("per_question", []) or []:
        per_q.append(asdict(r) if hasattr(r, "__dataclass_fields__") else r)

    report = {
        "aggregate": aggregate,
        "num_questions": len(per_q),
        "per_question": per_q,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
