"""End-to-end RAG pipeline (report §5.8, §6, §7).

Assembles the components (embedder, vector store, retriever, LLM) from config
and provides:
  * :func:`ingest`     - load -> chunk -> embed -> store
  * :class:`RAGPipeline` - retrieve -> build prompt -> generate grounded answer
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .chunk import chunk_pages
from .config import Config
from .embed import Embedder
from .ingest import load_documents
from .llm import GenParams, OllamaLLM
from .retriever import Retriever, format_context, unique_sources
from .vectorstore import Retrieved, VectorStore


# --------------------------------------------------------------------------
# Component factories
# --------------------------------------------------------------------------
def build_embedder(cfg: Config) -> Embedder:
    return Embedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        normalize=cfg.embedding.normalize,
    )


def build_vectorstore(cfg: Config, embedder: Embedder | None = None) -> VectorStore:
    embedder = embedder or build_embedder(cfg)
    return VectorStore(
        persist_dir=cfg.chroma_path,
        collection_name=cfg.vectorstore.collection_name,
        embedder=embedder,
    )


def build_llm(cfg: Config) -> OllamaLLM:
    return OllamaLLM(
        model=cfg.llm.model,
        host=cfg.llm.host,
        defaults=GenParams(
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            top_k=cfg.llm.top_k,
            max_tokens=cfg.llm.max_tokens,
            num_ctx=cfg.llm.num_ctx,
        ),
    )


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
@dataclass
class IngestReport:
    pdf_dirs: list[str]
    pages: int
    chunks: int
    stored: int
    collection_size: int


def ingest(cfg: Config, *, reset: bool = False, embedder: Embedder | None = None) -> IngestReport:
    """Load all PDFs under the docs dir, chunk, embed, and store in Chroma."""
    store = build_vectorstore(cfg, embedder)
    if reset:
        store.reset()

    pages = load_documents(cfg.docs_path)
    chunks = chunk_pages(
        pages,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    stored = store.add_chunks(chunks) if chunks else 0
    return IngestReport(
        pdf_dirs=[str(cfg.docs_path)],
        pages=len(pages),
        chunks=len(chunks),
        stored=stored,
        collection_size=store.count(),
    )


# --------------------------------------------------------------------------
# RAG query pipeline
# --------------------------------------------------------------------------
@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[Retrieved] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return format_context(self.contexts)


class RAGPipeline:
    def __init__(self, cfg: Config, embedder: Embedder | None = None):
        self.cfg = cfg
        self.embedder = embedder or build_embedder(cfg)
        self.store = build_vectorstore(cfg, self.embedder)
        self.retriever = Retriever(
            self.store, k=cfg.retriever.k, search_type=cfg.retriever.search_type
        )
        self.llm = build_llm(cfg)

    # -- report §3: baseline LLM, no retrieval --
    def baseline(self, question: str, **gen) -> str:
        return self.llm.generate(question, system=prompts.SYSTEM_BASELINE, **gen)

    # -- report §4: prompt engineering, no retrieval --
    def prompted(self, question: str, **gen) -> str:
        return self.llm.generate(question, system=prompts.SYSTEM_PROMPTED, **gen)

    # -- report §5.8/§8: full RAG --
    def answer(self, question: str, *, k: int | None = None, **gen) -> RAGResult:
        contexts = self.retriever.retrieve(question, k=k)
        context_block = format_context(contexts)
        user_msg = prompts.build_grounded_user_message(context_block, question)
        text = self.llm.generate(user_msg, system=prompts.SYSTEM_GROUNDED, **gen)
        return RAGResult(
            question=question,
            answer=text,
            contexts=contexts,
            sources=unique_sources(contexts),
        )
