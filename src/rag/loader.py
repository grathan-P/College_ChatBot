
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF file and return its pages as LangChain Documents.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the supplied path is not a PDF file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, but received: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, but received: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    return documents


def load_directory(directory_path: str | Path) -> list[Document]:
    """
    Recursively load all PDF files from a directory.

    Args:
        directory_path: Directory containing PDF documents.

    Returns:
        A combined list of LangChain Documents from all PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, but received: {directory}"
        )

    pdf_files = sorted(directory.rglob("*.pdf"))

    documents: list[Document] = []

    for pdf_file in pdf_files:
        documents.extend(load_pdf(pdf_file))

    return documents
