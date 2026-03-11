"""Build the Sukhibava FAISS index from the medical knowledge base."""

from dotenv import load_dotenv

from rag.vectorstore import build_vectorstore


def main() -> None:
    """Load environment variables and build the local vector store."""
    load_dotenv()
    build_vectorstore()
    print("Vector store built successfully in faiss_index/")


if __name__ == "__main__":
    main()
