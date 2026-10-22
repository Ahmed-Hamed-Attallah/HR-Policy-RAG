"""Unified RAG orchestrator integrating retrieval, reranking, context, LLM, and guardrails."""

import logging
from typing import Optional

from src.domain.responses import RAGResponse
from src.generation.answer_generator import AnswerGenerator
from src.generation.prompt_templates import FALLBACK_EMPTY_CONTEXT_MESSAGE
from src.rag.citations import CitationManager
from src.rag.context_builder import ContextBuilder
from src.rag.guardrails import PolicyGuardrails
from src.reranking.reranker import PolicyReranker
from src.retrieval.filters import MetadataFilter
from src.retrieval.retriever import PolicyRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG orchestrator for enterprise HR policy question-answering."""

    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        reranker: Optional[PolicyReranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[AnswerGenerator] = None,
        guardrails: Optional[PolicyGuardrails] = None,
    ):
        self.retriever = retriever or PolicyRetriever()
        self.reranker = reranker or PolicyReranker.get_instance()
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator or AnswerGenerator()
        self.guardrails = guardrails or PolicyGuardrails()

    def run(
        self,
        query: str,
        retrieval_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        filter_criteria: Optional[MetadataFilter] = None,
        score_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """Execute the end-to-end RAG workflow with guardrails and structured responses."""
        logger.info(f"Initiating RAG pipeline for query: '{query[:60]}...'")

        # 1. Input Guardrail Inspection
        input_verdict = self.guardrails.validate_input(query)
        if not input_verdict.is_safe:
            logger.warning(f"Guardrail input rejection: {input_verdict.violation_reason}")
            return RAGResponse(
                query=query,
                answer=f"Unable to process query: {input_verdict.violation_reason}",
                citations=[],
                retrieved_chunks=[],
                model_name=None,
                is_fallback=True,
            )

        sanitized_query = input_verdict.sanitized_text

        # 2. Vector Retrieval
        retrieved_items = self.retriever.retrieve(
            query=sanitized_query,
            top_k=retrieval_k,
            filter_criteria=filter_criteria,
            score_threshold=score_threshold,
        )

        if not retrieved_items:
            logger.info("Retrieval returned zero candidate passages.")
            return RAGResponse(
                query=sanitized_query,
                answer=FALLBACK_EMPTY_CONTEXT_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                model_name=None,
                is_fallback=True,
            )

        # 3. Neural Re-ranking
        ranked_items = self.reranker.rerank(
            query=sanitized_query,
            candidates=retrieved_items,
            top_k=rerank_k,
        )

        # 4. Context Assembly
        assembly = self.context_builder.build_context(ranked_items)

        # 5. Citation Generation & Preparation
        initial_citations = CitationManager.build_citations(assembly.included_chunks)

        # 6. LLM Answer Synthesis
        raw_answer = self.generator.generate_answer(
            query=sanitized_query,
            context=assembly.assembled_text,
        )

        # 7. Output Guardrail Inspection
        output_verdict = self.guardrails.validate_output(raw_answer)
        final_answer = output_verdict.sanitized_text if output_verdict.is_safe else raw_answer

        # 8. Align Citations to Explicit In-Text References
        active_citations = CitationManager.filter_citations_referenced_in_text(
            answer_text=final_answer,
            all_citations=initial_citations,
        )

        return RAGResponse(
            query=sanitized_query,
            answer=final_answer,
            citations=active_citations,
            retrieved_chunks=assembly.included_chunks,
            model_name=getattr(self.generator.client.config, "model_name", "unknown"),
            is_fallback=not output_verdict.is_safe or final_answer == FALLBACK_EMPTY_CONTEXT_MESSAGE,
        )