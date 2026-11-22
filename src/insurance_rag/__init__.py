"""Insurance RAG Q&A Bot.

A Retrieval-Augmented Generation pipeline that answers insurance policy and
claims questions grounded in approved documents, with source citations.

Pipeline (see the project report, sections 5-8):
    ingest -> chunk -> embed (gte-large) -> store (ChromaDB)
           -> retrieve (top-k) -> generate (Ollama/Mistral) -> grounded answer
"""

__version__ = "0.1.0"
