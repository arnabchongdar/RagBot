"""Configuration loading.

Reads ``config.yaml`` from the project root and exposes it as typed dataclasses.
All tunables (chunking, embedding model, retriever k, LLM params, sweep configs,
evaluation queries) live in the YAML so the code stays declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/insurance_rag/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Paths:
    docs_dir: Path
    synthetic_dir: Path
    chroma_dir: Path


@dataclass
class Chunking:
    chunk_size: int = 512
    chunk_overlap: int = 20


@dataclass
class Embedding:
    model_name: str = "thenlper/gte-large"
    device: str = "cpu"
    normalize: bool = True


@dataclass
class VectorStore:
    collection_name: str = "insurance_docs"


@dataclass
class Retriever:
    search_type: str = "similarity"
    k: int = 3


@dataclass
class LLM:
    provider: str = "ollama"
    model: str = "mistral"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 256
    num_ctx: int = 4096


@dataclass
class Evaluation:
    judge_model: str = "mistral"
    judge_temperature: float = 0.0


@dataclass
class SweepConfig:
    name: str
    k: int
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int


@dataclass
class Query:
    id: str
    question: str
    gold: str | None = None


@dataclass
class Config:
    paths: Paths
    chunking: Chunking
    embedding: Embedding
    vectorstore: VectorStore
    retriever: Retriever
    llm: LLM
    evaluation: Evaluation
    sweep_configs: list[SweepConfig] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    root: Path = PROJECT_ROOT

    # -- convenience path resolvers (absolute) --
    @property
    def docs_path(self) -> Path:
        return self._resolve(self.paths.docs_dir)

    @property
    def synthetic_path(self) -> Path:
        return self._resolve(self.paths.synthetic_dir)

    @property
    def chroma_path(self) -> Path:
        return self._resolve(self.paths.chroma_dir)

    def _resolve(self, p: Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)


def _clean(gold: str | None) -> str | None:
    if gold is None:
        return None
    return " ".join(gold.split()).strip() or None


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate configuration from ``config.yaml``."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    paths = Paths(
        docs_dir=Path(raw.get("paths", {}).get("docs_dir", "docs")),
        synthetic_dir=Path(raw.get("paths", {}).get("synthetic_dir", "docs/synthetic")),
        chroma_dir=Path(raw.get("paths", {}).get("chroma_dir", "data/chroma")),
    )
    chunking = Chunking(**raw.get("chunking", {}))
    embedding = Embedding(**raw.get("embedding", {}))
    vectorstore = VectorStore(**raw.get("vectorstore", {}))
    retriever = Retriever(**raw.get("retriever", {}))
    llm = LLM(**raw.get("llm", {}))
    evaluation = Evaluation(**raw.get("evaluation", {}))
    sweep = [SweepConfig(**c) for c in raw.get("sweep_configs", [])]
    queries = [
        Query(id=q["id"], question=q["question"], gold=_clean(q.get("gold")))
        for q in raw.get("queries", [])
    ]

    return Config(
        paths=paths,
        chunking=chunking,
        embedding=embedding,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        evaluation=evaluation,
        sweep_configs=sweep,
        queries=queries,
        root=cfg_path.resolve().parent,
    )
