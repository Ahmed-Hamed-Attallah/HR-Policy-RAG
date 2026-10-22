"""Integration test verifying end-to-end RAG orchestrator execution."""

from unittest.mock import MagicMock
from src.domain.chunks import ChunkMetadata, DocumentChunk
from src.domain.responses import RetrievedChunk
from src.rag.pipeline import RAGPipeline


def test_rag_pipeline_end_to_end_flow():
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_generator = MagicMock()

    sample_chunk = DocumentChunk(
        chunk_id="chk_prob_1",
        content="The standard probation period is three months from employment date.",
        metadata=ChunkMetadata(
            doc_id="doc_hr",
            document_name="Policy.pdf",
            page_number=2,
            chunk_index=1,
            char_length=70,
        ),
    )
    retrieved = [RetrievedChunk(chunk=sample_chunk, similarity_score=0.91)]

    mock_retriever.retrieve.return_value = retrieved
    mock_reranker.rerank.return_value = retrieved
    mock_generator.generate_answer.return_value = (
        "The standard probation period is three months [Policy.pdf, Page 2]."
    )
    mock_generator.client = MagicMock()
    mock_generator.client.config.model_name = "mock-llm"

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator=mock_generator,
    )

    result = pipeline.run(query="What is the probation period?", retrieval_k=2)

    assert result.is_fallback is False
    assert "three months" in result.answer
    assert len(result.retrieved_chunks) == 1
    assert len(result.citations) == 1
    assert result.citations[0].source_name == "Policy.pdf"