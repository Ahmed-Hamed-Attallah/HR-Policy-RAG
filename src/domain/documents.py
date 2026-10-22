"""Domain data models for raw and parsed documents."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata attributes associated with an ingested document."""
    source_path: str = Field(description="Absolute or relative file path to the original document")
    file_name: str = Field(description="Document base file name (e.g. Leave_Policy.pdf)")
    file_type: DocumentType = Field(default=DocumentType.UNKNOWN, description="File type enum")
    file_size_bytes: int = Field(default=0, ge=0, description="Size of file in bytes")
    file_hash: str = Field(description="SHA-256 checksum hash for change detection & deduplication")
    total_pages: Optional[int] = Field(default=None, description="Total number of pages if applicable")
    title: Optional[str] = Field(default=None, description="Extracted or declared document title")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")
    extra_attributes: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom metadata")


class RawDocument(BaseModel):
    """Container for unparsed, raw document binary and location."""
    file_path: Path
    file_hash: str
    file_type: DocumentType


class ParsedPage(BaseModel):
    """Represents a single extracted page or section before chunking."""
    page_number: int = Field(ge=1, description="Page index (1-based)")
    text: str = Field(description="Raw or lightly normalized page text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Page-level metadata")


class Document(BaseModel):
    """Core domain representation of a fully parsed document."""
    doc_id: str = Field(description="Unique deterministic ID (typically based on file hash)")
    content: str = Field(description="Full concatenated text content of the document")
    pages: list[ParsedPage] = Field(default_factory=list, description="List of individual parsed pages")
    metadata: DocumentMetadata = Field(description="Comprehensive metadata envelope")