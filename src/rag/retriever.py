
from dataclasses import dataclass, field
from langchain_core.documents import Document
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import FAISSVectorStore

@dataclass
class RetrievalResult:
    """
    Represents one retrieved document chunk.
    """
    content: str
    source: str
    page: int | None
    score: float
    metadata: dict = field(default_factory=dict)  # Expose the raw metadata dict

    @property
    def academic_year(self) -> str:
        return self.metadata.get("academic_year", "N/A")

    @property
    def document_type(self) -> str:
        return self.metadata.get("document_type", "N/A")

    @property
    def document_status(self) -> str:
        return self.metadata.get("document_status", "N/A")


class Retriever:
    """
    High-level retrieval interface for the RAG pipeline.
    """
    def __init__(
        self, 
        embedding_model: EmbeddingModel, 
        vector_store: FAISSVectorStore
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = self.embedding_model.embed_query(query)
        search_results = self.vector_store.search(
            query_embedding=query_embedding, 
            top_k=top_k
        )

        results: list[RetrievalResult] = []
        for document, score in search_results:
            results.append(
                self._create_result(document=document, score=score)
            )
        return results

    @staticmethod
    def _create_result(
        document: Document, 
        score: float
    ) -> RetrievalResult:
        source = str(document.metadata.get("source", "unknown"))
        page = document.metadata.get("page")
        if page is not None:
            page = int(page)
            
        return RetrievalResult(
            content=document.page_content,
            source=source,
            page=page,
            score=score,
            metadata=document.metadata # Pass the entire dict
        )
