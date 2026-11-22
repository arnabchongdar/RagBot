"""Vector database (report §5.5).

Thin wrapper over a persistent ChromaDB collection. Embeddings are supplied
explicitly by :class:`~insurance_rag.embed.Embedder`; Chroma stores vectors,
documents, and metadata and performs similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunk import Chunk
from .embed import Embedder


@dataclass
class Retrieved:
    text: str
    metadata: dict[str, Any]
    score: float  # cosine distance (lower = more similar)


class VectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedder: Embedder,
    ):
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # Cosine space matches normalized embeddings from gte-large.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        """Embed and upsert chunks. Idempotent on chunk id."""
        added = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.embed_documents([c.text for c in batch])
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            added += len(batch)
        return added

    def query(self, text: str, k: int = 3) -> list[Retrieved]:
        query_emb = self.embedder.embed_query(text)
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[Retrieved] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(Retrieved(text=doc, metadata=meta or {}, score=float(dist)))
        return out
