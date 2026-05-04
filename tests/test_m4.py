"""Tests for Module 4: Evaluation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))


def test_evaluate_default_does_not_call_real_ragas(monkeypatch):
    """Default evaluation must stay deterministic/offline unless explicitly enabled."""

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("real RAGAS should be opt-in")

    monkeypatch.delenv("LAB18_USE_REAL_RAGAS", raising=False)
    monkeypatch.setattr("src.m4_eval._try_ragas", fail_if_called)

    r = evaluate_ragas(
        ["AI là gì?"],
        ["AI là trí tuệ nhân tạo."],
        [["AI là trí tuệ nhân tạo, giúp máy tính thực hiện nhiệm vụ thông minh."]],
        ["AI là trí tuệ nhân tạo."],
    )

    assert r["faithfulness"] == 1.0
    assert r["per_question"][0].question == "AI là gì?"
    assert r["per_question"][0].contexts[0].startswith("AI là trí tuệ")

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]
