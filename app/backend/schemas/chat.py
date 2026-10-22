"""Pydantic schemas for RAG query and chat interaction endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatFilterRequest(BaseModel):
    """Optional metadata criteria to narrow document retrieval."""
    document_names: Optional[List[str]] = Field(default=None, description="List of source file names")
    department: Optional[str] = Field(default=None, description="Department filter")
    section_title: Optional[str] = Field(default=None, description="Policy section title")


class ChatQueryRequest(BaseModel):
    """Request payload for submitting a query to the HR policy RAG engine."""
    query: str = Field(..., min_length=2, max_length=1000, description="Employee question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of passages to retrieve")
    rerank_k: Optional[int] = Field(default=3, ge=1, le=10, description="Top passages kept after reranking")
    filters: Optional[ChatFilterRequest] = Field(default=None, description="Metadata filter constraints")


class CitationResponse(BaseModel):
    """Source evidence card attached to an answer."""
    source_name: str
    page_number: Optional[int] = None
    snippet: str
    score: Optional[float] = None


class RetrievedPassageResponse(BaseModel):
    """Diagnostic detail of passages considered during generation."""
    chunk_id: str
    document_name: str
    page_number: Optional[int] = None
    content: str
    similarity_score: float
    rerank_score: Optional[float] = None


class ChatQueryResponse(BaseModel):
    """Complete response returned by the RAG orchestrator."""
    query: str
    answer: str
    citations: List[CitationResponse] = Field(default_factory=list)
    retrieved_passages: List[RetrievedPassageResponse] = Field(default_factory=list)
    model_name: Optional[str] = None
    is_fallback: bool = False
    latency_ms: Optional[float] = None