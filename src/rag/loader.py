from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def _get_document_metadata(path: Path) -> dict[str, str]:
    """Return stable metadata for a known project document."""

    filename = path.name

    if "B.Tech." in filename:
        return {
            "source": filename,
            "academic_year": "2026-27",
            "document_type": "regulations_and_syllabus",
            "document_status": "current",
        }

    if "NMAMIT_Student_Academic_Internship_Guide" in filename:
        return {
            "source": filename,
            "academic_year": "2026-27",
            "document_type": "student_reference_guide",
            "document_status": "reference",
        }

    if "Rules & Regulations" in filename:
        return {
            "source": filename,
            "academic_year": "2024-25",
            "document_type": "regulations",
            "document_status": "historical",
        }

    return {
        "source": filename,
        "academic_year": "unknown",
        "document_type": "unknown",
        "document_status": "unknown",
    }


def load_pdf(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, but received: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, but received: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    metadata = _get_document_metadata(path)

    for document in documents:
        document.metadata.update(metadata)

    return documents


def load_directory(directory_path: str | Path) -> list[Document]:
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