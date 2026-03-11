"""FAISS vector store creation and loading utilities."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag.embeddings import get_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "medical_data.txt"
INDEX_DIR = BASE_DIR / "faiss_index"


def build_vectorstore() -> FAISS:
    """Build and persist the FAISS vector store from the medical knowledge base."""
    medical_text = DATA_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(medical_text)
    vectorstore = FAISS.from_texts(chunks, embedding=get_embeddings())
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously saved FAISS index from disk."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python ingest.py` before starting the chatbot."
        )

    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
