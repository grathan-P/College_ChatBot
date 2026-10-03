
"""Utilities for formatting retrieved college documents for the LLM."""

from src.rag.retriever import RetrievalResult


def format_retrieved_context(
    results: list[RetrievalResult],
) -> str:
    """Convert retrieved RAG results into structured LLM context.

    Args:
        results: Retrieved chunks returned by the RAG Retriever.

    Returns:
        A formatted string containing source metadata and chunk content.
        Returns an empty string when no results are supplied.
    """
    if not results:
        return ""

    context_parts: list[str] = []

    for index, result in enumerate(results, start=1):
        page = str(result.page) if result.page is not None else "Unknown"
        academic_year = result.academic_year or "Unknown"
        document_type = result.document_type or "Unknown"
        document_status = result.document_status or "Unknown"

        context_part = (
            f"[Source {index}]\n"
            f"Document: {result.source}\n"
            f"Page: {page}\n"
            f"Academic Year: {academic_year}\n"
            f"Document Type: {document_type}\n"
            f"Document Status: {document_status}\n\n"
            f"Content:\n{result.content}"
        )

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)
