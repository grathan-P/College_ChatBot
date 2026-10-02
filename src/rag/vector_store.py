
import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document


class FAISSVectorStore:
    """
    FAISS-backed vector store for LangChain Documents.

    The FAISS index stores vectors, while the JSON metadata file
    stores the corresponding document content and metadata.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than 0.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[Document] = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add documents and their corresponding embeddings to the index.

        Args:
            documents: Documents/chunks to store.
            embeddings: Embedding vectors corresponding to each document.

        Raises:
            ValueError: If documents and embeddings don't match.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        if not documents:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"but received {vectors.shape[1]}."
            )

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Search the vector store for the most similar documents.

        Args:
            query_embedding: Embedding vector for the query.
            top_k: Maximum number of results to return.

        Returns:
            List of (Document, similarity_score) tuples.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"but received {query_vector.shape[1]}."
            )

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_vector,
            k,
        )

        results: list[tuple[Document, float]] = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (
                    self.documents[index],
                    float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Save the FAISS index and document metadata to disk.
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory_path / "index.faiss"),
        )

        metadata = [
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            for document in self.documents
        ]

        with open(
            directory_path / "documents.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """
        Load a previously saved FAISS vector store.
        """
        directory_path = Path(directory)

        index_path = directory_path / "index.faiss"
        documents_path = directory_path / "documents.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not documents_path.exists():
            raise FileNotFoundError(
                f"Document metadata not found: {documents_path}"
            )

        index = faiss.read_index(str(index_path))

        with open(
            documents_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        store = cls(index.d)

        store.index = index

        store.documents = [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in metadata
        ]

        if store.index.ntotal != len(store.documents):
            raise ValueError(
                "FAISS index and document metadata are inconsistent."
            )

        return store
