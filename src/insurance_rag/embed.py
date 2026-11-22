"""Embedding generation (report §5.4).

Wraps a Sentence-Transformers model (``thenlper/gte-large`` by default) and
produces normalized vector embeddings. Embeddings are computed explicitly and
handed to ChromaDB, so we don't depend on Chroma's embedding-function API.
"""

from __future__ import annotations

from functools import lru_cache

# gte models do not require query/document instruction prefixes.


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class Embedder:
    """Encodes text into embedding vectors using Sentence-Transformers."""

    def __init__(
        self,
        model_name: str = "thenlper/gte-large",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.encode([text])[0]
