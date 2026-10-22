"""Domain models for textual chunks and their vector-ready representations."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata tied to a specific chunk for attribution and filtering."""
    doc_id: str = Field(description="Parent document unique identifier")
    document_name: str = Field(description="Original document file name")
    page_number: Optional[int] = Field(default=None, description="Origin page number if applicable")
    chunk_index: int = Field(ge=0, description="Sequential index of the chunk within the document")
    char_length: int = Field(ge=0, description="Character count of chunk text")
    token_count: Optional[int] = Field(default=None, description="Estimated or exact token count")
    section_title: Optional[str] = Field(default=None, description="Detected header or section name")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class DocumentChunk(BaseModel):
    """A semantic text unit produced by text splitters, ready for indexing/retrieval."""
    chunk_id: str = Field(description="Deterministic ID (e.g. {doc_id}_chunk_{chunk_index})")
    content: str = Field(description="Sanitized text content of the chunk")
    metadata: ChunkMetadata = Field(description="Associated chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Dense vector embedding")

    def to_vector_payload(self) -> Dict[str, Any]:
        """Convert chunk metadata and text to dictionary payload for storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.metadata.doc_id,
            "document_name": self.metadata.document_name,
            "page_number": self.metadata.page_number,
            "chunk_index": self.metadata.chunk_index,
            "section_title": self.metadata.section_title,
            "content": self.content,
        }