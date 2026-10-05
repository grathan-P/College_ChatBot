from pathlib import Path

from src.rag.embeddings import EmbeddingModel
from src.rag.loader import load_directory
from src.rag.vector_store import FAISSVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"
FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"


def main() -> None:
    print("Loading PDF documents...")

    documents = load_directory(DOCUMENTS_DIR)

    if not documents:
        raise RuntimeError(
            f"No PDF documents found in {DOCUMENTS_DIR}"
        )

    print(f"Loaded {len(documents)} PDF pages.")

    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    print("Creating embeddings...")

    texts = [document.page_content for document in documents]
    embeddings = embedding_model.embed_documents(texts)

    print(
        f"Created {len(embeddings)} embeddings "
        f"with dimension {embedding_model.dimension}."
    )

    print("Creating FAISS vector store...")

    vector_store = FAISSVectorStore(
        dimension=embedding_model.dimension
    )

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
    )

    print("Saving FAISS index...")

    vector_store.save(FAISS_INDEX_DIR)

    print(f"FAISS index saved to: {FAISS_INDEX_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()