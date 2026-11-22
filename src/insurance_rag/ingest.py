"""Document ingestion (report §5.1, §5.3).

Loads insurance PDFs page-by-page using PyMuPDF. Prefers LangChain's
``PyMuPDFLoader`` (as named in the report) and falls back to raw PyMuPDF so
ingestion never hard-fails on a LangChain API change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadedPage:
    """A single page of a source document."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_with_langchain(pdf_path: Path) -> list[LoadedPage] | None:
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except Exception:
        return None
    try:
        docs = PyMuPDFLoader(str(pdf_path)).load()
    except Exception:
        return None
    pages: list[LoadedPage] = []
    for d in docs:
        meta = dict(d.metadata or {})
        pages.append(
            LoadedPage(
                text=d.page_content or "",
                metadata={
                    "source": Path(meta.get("source", pdf_path)).name,
                    "page": int(meta.get("page", 0)),
                },
            )
        )
    return pages


def _load_with_fitz(pdf_path: Path) -> list[LoadedPage]:
    try:
        import pymupdf  # PyMuPDF >= 1.24 preferred module name
    except ImportError:  # pragma: no cover - fallback for older installs
        import fitz as pymupdf

    pages: list[LoadedPage] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            pages.append(
                LoadedPage(
                    text=page.get_text("text"),
                    metadata={"source": pdf_path.name, "page": i},
                )
            )
    return pages


def load_pdf(pdf_path: Path) -> list[LoadedPage]:
    """Load a single PDF into per-page ``LoadedPage`` records."""
    pages = _load_with_langchain(pdf_path)
    if pages is None:
        pages = _load_with_fitz(pdf_path)
    return pages


def load_documents(docs_dir: Path, *, recursive: bool = True) -> list[LoadedPage]:
    """Load every PDF under ``docs_dir`` (recursively by default).

    Returns a flat list of pages across all documents. Non-PDF files are ignored.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(docs_dir.glob(pattern))
    pages: list[LoadedPage] = []
    for pdf in pdfs:
        pages.extend(load_pdf(pdf))
    return pages


def list_pdfs(docs_dir: Path, *, recursive: bool = True) -> list[Path]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(docs_dir.glob(pattern))
