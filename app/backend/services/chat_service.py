"""Application service coordinating the RAG query pipeline and response translation."""

import logging
import time
from typing import Optional

from app.backend.exceptions import GuardrailViolationError
from app.backend.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    CitationResponse,
    RetrievedPassageResponse,
)
from src.rag.pipeline import RAGPipeline
from src.retrieval.filters import MetadataFilter

logger = logging.getLogger(__name__)


class ChatService:
    """Service wrapping the RAG pipeline execution for HTTP clients."""

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.pipeline = rag_pipeline or RAGPipeline()

    def process_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """Translate API query parameters, execute pipeline, and package response."""
        start_time = time.time()

        # Build metadata filter if provided
        filter_criteria = None
        if request.filters:
            filter_criteria = MetadataFilter(
                document_names=request.filters.document_names,
                department=request.filters.department,
                section_title=request.filters.section_title,
            )

        rag_result = self.pipeline.run(
            query=request.query,
            retrieval_k=request.top_k,
            rerank_k=request.rerank_k,
            filter_criteria=filter_criteria,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Map domain Citations to API schema
        citations_payload = [
            CitationResponse(
                source_name=c.source_name,
                page_number=c.page_number,
                snippet=c.snippet,
                score=c.score,
            )
            for c in rag_result.citations
        ]

        # Map candidate chunks to diagnostic payload
        passages_payload = [
            RetrievedPassageResponse(
                chunk_id=rc.chunk.chunk_id,
                document_name=rc.chunk.metadata.document_name,
                page_number=rc.chunk.metadata.page_number,
                content=rc.chunk.content,
                similarity_score=rc.similarity_score,
                rerank_score=rc.rerank_score,
            )
            for rc in rag_result.retrieved_chunks
        ]

        return ChatQueryResponse(
            query=rag_result.query,
            answer=rag_result.answer,
            citations=citations_payload,
            retrieved_passages=passages_payload,
            model_name=rag_result.model_name,
            is_fallback=rag_result.is_fallback,
            latency_ms=round(latency_ms, 2),
        )