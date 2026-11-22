"""Retriever (report §5.6).

Connects the query to the vector database and returns the top-k most similar
chunks. Also formats retrieved chunks into a numbered, citable context block.
"""

from __future__ import annotations

from .vectorstore import Retrieved, VectorStore


class Retriever:
    def __init__(self, vectorstore: VectorStore, k: int = 3, search_type: str = "similarity"):
        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type

    def retrieve(self, question: str, k: int | None = None) -> list[Retrieved]:
        return self.vectorstore.query(question, k=k or self.k)


def source_label(meta: dict) -> str:
    src = meta.get("source", "unknown")
    page = meta.get("page")
    return f"{src}, p.{int(page) + 1}" if isinstance(page, int) else str(src)


def format_context(retrieved: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    if not retrieved:
        return "(no relevant context retrieved)"
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        blocks.append(f"[S{i}] ({source_label(r.metadata)})\n{r.text.strip()}")
    return "\n\n".join(blocks)


def unique_sources(retrieved: list[Retrieved]) -> list[str]:
    seen: list[str] = []
    for r in retrieved:
        label = source_label(r.metadata)
        if label not in seen:
            seen.append(label)
    return seen
