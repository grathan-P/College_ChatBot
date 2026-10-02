
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
        ValueError: If the supplied file is not a PDF.
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
