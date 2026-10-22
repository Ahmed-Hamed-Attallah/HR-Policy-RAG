"""Application service orchestrating document upload, indexing, and registry management."""

import logging
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile

from app.backend.exceptions import DocumentNotFoundError, InvalidFileTypeError
from app.backend.schemas.documents import (
    DocumentDetailResponse,
    DocumentSummaryResponse,
    DocumentUploadResponse,
)
from src.config.settings import settings
from src.domain.documents import Document
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.loader_factory import LoaderFactory
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.metadata_extractor import MetadataExtractor
from src.utils.file_utils import ensure_directory, read_json
from src.vector_store.index_manager import IndexManager

logger = logging.getLogger(__name__)


class DocumentService:
    """Business logic for policy document ingestion and lifecycle management."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(
        self,
        registry: Optional[DocumentRegistry] = None,
        index_manager: Optional[IndexManager] = None,
    ):
        self.registry = registry or DocumentRegistry()
        self.index_manager = index_manager or IndexManager()
        self.raw_dir = settings.DATA_RAW_DIR
        self.processed_dir = settings.DATA_PROCESSED_DIR
        ensure_directory(self.raw_dir)
        ensure_directory(self.processed_dir)

    def list_documents(self) -> List[DocumentSummaryResponse]:
        """Fetch all indexed documents recorded in the registry."""
        registry_docs = self.registry.list_documents()
        summaries: List[DocumentSummaryResponse] = []

        for record in registry_docs:
            summaries.append(
                DocumentSummaryResponse(
                    doc_id=record["doc_id"],
                    file_name=record["file_name"],
                    file_type=record["file_type"],
                    file_size_bytes=record["file_size_bytes"],
                    total_pages=record.get("total_pages"),
                    created_at=record["created_at"],
                    extra_attributes=record.get("extra_attributes", {}),
                )
            )
        return summaries

    def get_document(self, doc_id: str) -> DocumentDetailResponse:
        """Retrieve full metadata and preview for a specific document."""
        registry_doc = self.registry.get_document(doc_id)
        if not registry_doc:
            raise DocumentNotFoundError(doc_id)

        # Attempt to read processed JSON for text preview
        json_path = self.processed_dir / f"{doc_id}.json"
        content_preview = ""
        total_chars = 0
        if json_path.exists():
            data = read_json(json_path)
            full_content = data.get("content", "")
            total_chars = len(full_content)
            content_preview = full_content[:500] + ("..." if total_chars > 500 else "")

        return DocumentDetailResponse(
            doc_id=registry_doc["doc_id"],
            file_name=registry_doc["file_name"],
            file_type=registry_doc["file_type"],
            file_size_bytes=registry_doc["file_size_bytes"],
            total_pages=registry_doc.get("total_pages"),
            created_at=registry_doc["created_at"],
            extra_attributes=registry_doc.get("extra_attributes", {}),
            content_preview=content_preview,
            total_characters=total_chars,
        )

    async def upload_document(self, upload_file: UploadFile) -> DocumentUploadResponse:
        """Save raw document to disk, parse, enrich metadata, index, and register."""
        filename = upload_file.filename or "uploaded_policy"
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise InvalidFileTypeError(ext)

        destination_path = self.raw_dir / filename
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        file_size = destination_path.stat().st_size

        # Ingest and parse
        loader = LoaderFactory.get_loader(destination_path)
        doc: Document = loader.load(destination_path)

        # Preprocess
        doc = DocumentCleaner.clean_document(doc)
        doc = MetadataExtractor.enrich_document_metadata(doc)

        # Save processed JSON
        doc_json_path = self.processed_dir / f"{doc.doc_id}.json"
        import json
        with open(doc_json_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

        # Add to FAISS index incrementally
        chunks_indexed = self.index_manager.index_documents([doc])

        # Register in registry
        self.registry.register(doc)

        logger.info(f"Uploaded and indexed policy document: '{filename}' ({chunks_indexed} chunks)")
        return DocumentUploadResponse(
            doc_id=doc.doc_id,
            file_name=filename,
            file_size_bytes=file_size,
            status="indexed",
            chunks_created=chunks_indexed,
            message="Document successfully processed and indexed into FAISS.",
        )