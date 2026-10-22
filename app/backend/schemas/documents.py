"""Pydantic schemas for document ingestion and management endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentSummaryResponse(BaseModel):
    """Overview payload for an ingested policy document."""
    doc_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    total_pages: Optional[int] = None
    created_at: datetime
    extra_attributes: Dict[str, Any] = Field(default_factory=dict)


class DocumentDetailResponse(DocumentSummaryResponse):
    """Detailed document payload including extracted text preview."""
    content_preview: str = Field(description="First 500 characters of extracted text")
    total_characters: int = Field(ge=0)


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading a new policy document."""
    doc_id: str
    file_name: str
    file_size_bytes: int
    status: str = Field(description="'indexed', 'skipped', or 'failed'")
    chunks_created: int = Field(default=0, ge=0)
    message: str


class DocumentDeleteResponse(BaseModel):
    """Status returned after unregistering and purging a document."""
    doc_id: str
    success: bool
    message: str