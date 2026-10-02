
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create a recursive text splitter for academic documents.

    Args:
        chunk_size: Maximum target size of each chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        Configured RecursiveCharacterTextSplitter.

    Raises:
        ValueError: If the chunking configuration is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks while preserving metadata.

    Args:
        documents: Documents produced by the document loader.
        chunk_size: Maximum target size of each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        A list of chunked LangChain Documents.
    """
    if not documents:
        return []

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    return chunks
