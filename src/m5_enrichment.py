"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import re
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full", "+".join(...)


# ─── OpenAI helper ────────────────────────────────────────


_OPENAI_CLIENT = None


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _openai_client():
    """Trả OpenAI client nếu có key, None otherwise."""
    global _OPENAI_CLIENT
    if not _env_flag("LAB18_USE_OPENAI_ENRICHMENT"):
        return None
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
        return _OPENAI_CLIENT
    except Exception:
        return None


def _chat(messages: list[dict], max_tokens: int = 200, temperature: float = 0.0) -> str | None:
    client = _openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """Tóm tắt 2-3 câu; fallback extractive 2 câu đầu."""
    out = _chat(
        messages=[
            {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
    )
    if out:
        return out
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return ""
    summary = " ".join(sentences[:2]).strip()
    if not summary.endswith(("?", "!", ".")):
        summary += "."
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate câu hỏi mà chunk có thể trả lời. Fallback heuristic."""
    out = _chat(
        messages=[
            {"role": "system", "content": (
                f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
                f"Trả về mỗi câu hỏi trên 1 dòng, không đánh số.")},
            {"role": "user", "content": text},
        ],
        max_tokens=250,
    )
    if out:
        questions = [q.strip().lstrip("0123456789.-)• ") for q in out.split("\n") if q.strip()]
        questions = [q for q in questions if q]
        if questions:
            return questions[:n_questions]

    return _heuristic_questions(text, n_questions)


def _heuristic_questions(text: str, n_questions: int) -> list[str]:
    """Fallback: tạo câu hỏi từ câu khẳng định."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    qs: list[str] = []
    for s in sentences[:n_questions * 2]:
        # Tách phần đầu câu làm chủ đề
        head = s.split(",")[0].split(":")[0]
        head = head.strip().rstrip(".")
        if not head:
            continue
        if any(num in s for num in ("%", " ngày", " giờ", " tháng", " năm", " đồng", " USD")):
            qs.append(f"{head} là bao nhiêu?")
        elif " là " in head:
            subj = head.split(" là ")[0].strip()
            qs.append(f"{subj} là gì?")
        else:
            qs.append(f"{head} như thế nào?")
        if len(qs) >= n_questions:
            break
    return qs[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend 1 câu mô tả vị trí + chủ đề; preserve original text."""
    out = _chat(
        messages=[
            {"role": "system", "content": (
                "Viết 1 câu ngắn (tối đa 25 từ) mô tả đoạn văn này thuộc tài liệu nào và nói về chủ đề gì. "
                "Chỉ trả về 1 câu duy nhất, không xuống dòng.")},
            {"role": "user", "content": f"Tài liệu: {document_title or 'không rõ'}\n\nĐoạn văn:\n{text}"},
        ],
        max_tokens=80,
    )
    if out:
        context_line = out.replace("\n", " ").strip()
        return f"{context_line}\n\n{text}"

    title = document_title.replace(".md", "").replace("_", " ").strip() or "tài liệu nội bộ"
    head = re.split(r"(?<=[.!?])\s+", text.strip())[0][:80]
    return f"Trích từ {title}, đoạn nói về: {head}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


_CATEGORY_HINTS = {
    "policy": ["chính sách", "quy định", "nghị định", "luật", "phép"],
    "hr": ["nhân viên", "lương", "thưởng", "tuyển dụng", "hr"],
    "it": ["mật khẩu", "vpn", "bảo mật", "backup", "server"],
    "finance": ["doanh thu", "lợi nhuận", "chi phí", "tài chính", "thuế"],
    "ai": ["ai", "machine learning", "deep learning", "rag", "embedding", "llm", "model", "mô hình"],
}


def extract_metadata(text: str) -> dict:
    """Trích metadata: topic, entities, category, language. Fallback heuristic."""
    out = _chat(
        messages=[
            {"role": "system", "content": (
                "Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ với các trường sau và KHÔNG thêm text khác: "
                '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|ai|other", '
                '"language": "vi|en"}.')},
            {"role": "user", "content": text},
        ],
        max_tokens=200,
    )
    if out:
        try:
            cleaned = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        except Exception:
            pass

    return _heuristic_metadata(text)


def _heuristic_metadata(text: str) -> dict:
    lower = text.lower()
    category = "other"
    for cat, keywords in _CATEGORY_HINTS.items():
        if any(k in lower for k in keywords):
            category = cat
            break

    # entities: tìm các từ viết hoa (đơn giản)
    entities = re.findall(r"\b[A-ZĐÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][\wÀ-ỹ]{2,}\b",
                          text)
    entities = list(dict.fromkeys(entities))[:5]

    # topic: câu đầu hoặc phần đầu trước dấu chấm
    first_sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    topic = first_sentence[:80].rstrip(",.") or "general"

    is_vietnamese = bool(re.search(r"[ăâđêôơưƯÂÂĐÊÔƠƯ]|à|á|ả|ã|ạ", text.lower()))
    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": "vi" if is_vietnamese else "en",
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Chạy enrichment pipeline trên danh sách chunks."""
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]
    use_full = "full" in methods

    enriched: list[EnrichedChunk] = []
    for chunk in chunks or []:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "")

        summary = summarize_chunk(text) if (use_full or "summary" in methods) else ""
        questions = generate_hypothesis_questions(text) if (use_full or "hyqa" in methods) else []
        if use_full or "contextual" in methods:
            enriched_text = contextual_prepend(text, source)
        else:
            enriched_text = text
        auto_meta = extract_metadata(text) if (use_full or "metadata" in methods) else {}

        # Index hypothesis questions cùng chunk → bridge vocabulary gap
        if questions:
            qa_block = "Câu hỏi liên quan: " + " | ".join(questions)
            enriched_text = f"{enriched_text}\n\n{qa_block}"

        merged_meta = {**meta, **auto_meta}
        method_label = "+".join([m for m in methods if m in {"contextual", "hyqa", "summary", "metadata", "full"}]) or "raw"

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata=merged_meta,
            method=method_label,
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên VinUni 2024')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")
