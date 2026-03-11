"""Embedding helpers for Sukhibava."""

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the sentence-transformer embedder used by the FAISS index."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
