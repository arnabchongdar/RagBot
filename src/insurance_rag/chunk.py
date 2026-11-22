"""Text chunking (report §5.3).

Splits loaded pages into overlapping chunks with LangChain's
``RecursiveCharacterTextSplitter`` (chunk size 512, overlap 20 by default),
preserving source/page metadata and assigning each chunk a stable id so
re-ingestion is idempotent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ingest import LoadedPage


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "doc"


def _build_splitter(chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Prefer splitting on paragraph / clause boundaries so related policy
        # conditions and exclusions stay together (report §5.3).
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )


def chunk_pages(
    pages: list[LoadedPage],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> list[Chunk]:
    """Split pages into chunks, carrying forward source and page metadata."""
    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    per_source_counter: dict[str, int] = {}

    for page in pages:
        text = (page.text or "").strip()
        if not text:
            continue
        source = str(page.metadata.get("source", "unknown"))
        page_no = page.metadata.get("page", 0)
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            idx = per_source_counter.get(source, 0)
            per_source_counter[source] = idx + 1
            chunk_id = f"{_slug(source)}::p{page_no}::c{idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={
                        "source": source,
                        "page": int(page_no) if isinstance(page_no, int) else 0,
                        "chunk": idx,
                    },
                )
            )
    return chunks
