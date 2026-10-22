"""Factory to detect file types and dispatch appropriate document loaders."""

from pathlib import Path
from typing import Dict, Type, Union
from src.domain.documents import Document, DocumentType
from src.ingestion.docx_loader import DocxLoader
from src.ingestion.pdf_loader import PDFLoader


class BaseDocumentLoader:
    """Abstract interface for all document loaders."""
    def load(self, file_path: Union[str, Path]) -> Document:
        raise NotImplementedError("Subclasses must implement load()")


class LoaderFactory:
    """Registry and factory for document loaders based on file extensions."""
    
    _loaders: Dict[DocumentType, Type[BaseDocumentLoader]] = {
        DocumentType.PDF: PDFLoader,
        DocumentType.DOCX: DocxLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Union[str, Path]) -> BaseDocumentLoader:
        """Instantiate appropriate loader based on file suffix."""
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")

        match extension:
            case "pdf":
                file_type = DocumentType.PDF
            case "docx" | "doc":
                file_type = DocumentType.DOCX
            case _:
                file_type = DocumentType.UNKNOWN

        loader_cls = cls._loaders.get(file_type)
        if not loader_cls:
            raise ValueError(f"Unsupported file format: '.{extension}' for file: {path.name}")

        return loader_cls()

    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> Document:
        """Convenience method to resolve loader and parse file in one step."""
        loader = cls.get_loader(file_path)
        return loader.load(file_path)