"""Data contracts for retrieval results, generated answers, and citations."""

from typing import List, Optional
from pydantic import BaseModel, Field
from src.domain.chunks import DocumentChunk


class Citation(BaseModel):
    """Citation reference pointing to the source document and page."""
    source_name: str = Field(description="File name of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number where evidence is found")
    snippet: str = Field(description="Exact or representative text snippet supporting the answer")
    score: Optional[float] = Field(default=None, description="Relevance or retrieval score")


class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector search or reranker with relevance score."""
    chunk: DocumentChunk
    similarity_score: float = Field(description="Raw vector similarity score")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder reranker score")


class RAGResponse(BaseModel):
    """Unified system response returned by the RAG orchestrator."""
    query: str = Field(description="Original user query")
    answer: str = Field(description="Synthesized LLM answer")
    citations: List[Citation] = Field(default_factory=list, description="Extracted citations")
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list, description="Candidate chunks used for generation"
    )
    model_name: Optional[str] = Field(default=None, description="Model used for generation")
    is_fallback: bool = Field(
        default=False, description="True if safe fallback occurred (e.g. not enough context or guardrails)"
    )