"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Helpers ──────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Tách câu cho cả tiếng Việt lẫn tiếng Anh."""
    parts = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _embed_sentences(sentences: list[str]):
    """Dùng SentenceTransformer nếu sẵn có; nếu lỗi tải model trả None."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(sentences, show_progress_bar=False)
    except Exception:
        return None


def _lexical_similarity(a: str, b: str) -> float:
    """Jaccard token similarity — fallback offline cho semantic chunking."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = _embed_sentences(sentences)
    chunks: list[Chunk] = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        if embeddings is not None:
            sim = _cosine(embeddings[i - 1], embeddings[i])
        else:
            sim = _lexical_similarity(sentences[i - 1], sentences[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group).strip(),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group).strip(),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    current = ""
    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            parents.append(_make_parent(current.strip(), len(parents), metadata))
            current = ""
        current += para + "\n\n"
    if current.strip():
        parents.append(_make_parent(current.strip(), len(parents), metadata))

    if not parents and text.strip():
        parents.append(_make_parent(text.strip(), 0, metadata))

    for parent in parents:
        pid = parent.metadata["parent_id"]
        ptext = parent.text
        step = max(child_size, 1)
        for start in range(0, len(ptext), step):
            piece = ptext[start:start + step].strip()
            if not piece:
                continue
            children.append(Chunk(
                text=piece,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "child_index": len(children), "strategy": "hierarchical"},
                parent_id=pid,
            ))

    return parents, children


def _make_parent(text: str, index: int, metadata: dict) -> Chunk:
    source = str(metadata.get("source", "")).strip()
    source_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_")
    pid = f"{source_prefix}_parent_{index}" if source_prefix else f"parent_{index}"
    return Chunk(
        text=text,
        metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                  "parent_index": index, "strategy": "hierarchical"},
    )


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r"(?m)(^#{1,6}\s+.+$)", text)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def flush():
        nonlocal current_content
        body = current_content.strip()
        header = current_header.strip()
        if not body and not header:
            return
        section_text = (f"{header}\n{body}".strip() if header else body)
        chunks.append(Chunk(
            text=section_text,
            metadata={**metadata, "section": header, "chunk_index": len(chunks),
                      "strategy": "structure"},
        ))
        current_content = ""

    for part in parts:
        if not part:
            continue
        if re.match(r"^#{1,6}\s+", part):
            flush()
            current_header = part.strip()
        else:
            current_content += part
    flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    """
    if not documents:
        return {k: {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
                for k in ("basic", "semantic", "hierarchical", "structure")}

    def stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic, semantic, hier_parents, hier_children, structure = [], [], [], [], []
    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})
        basic.extend(chunk_basic(text, metadata=meta))
        semantic.extend(chunk_semantic(text, metadata=meta))
        p, c = chunk_hierarchical(text, metadata=meta)
        hier_parents.extend(p)
        hier_children.extend(c)
        structure.extend(chunk_structure_aware(text, metadata=meta))

    hier_stats = stats(hier_children)
    hier_stats["num_parents"] = len(hier_parents)

    return {
        "basic": stats(basic),
        "semantic": stats(semantic),
        "hierarchical": hier_stats,
        "structure": stats(structure),
    }


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, st in results.items():
        print(f"  {name}: {st}")
