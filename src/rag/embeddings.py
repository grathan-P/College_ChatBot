
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around a Sentence Transformers embedding model.

    This keeps the rest of the RAG pipeline independent
    from the specific embedding provider/model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert documents into embedding vectors.

        Args:
            texts: Texts to embed.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a query into an embedding vector.

        Args:
            query: User's search query.

        Returns:
            Query embedding vector.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """
        return self.model.get_embedding_dimension()
