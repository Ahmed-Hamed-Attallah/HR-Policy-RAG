"""Unit tests for metadata filtering and similarity retrieval logic."""

from unittest.mock import MagicMock
import pytest
from src.domain.chunks import ChunkMetadata, DocumentChunk
from src.retrieval.filters import MetadataFilter, apply_filters, matches_filter
from src.retrieval.similarity_search import SimilaritySearchService


@pytest.fixture
def mock_chunks():
    c1 = DocumentChunk(
        chunk_id="chunk_1",
        content="Annual leave is 21 days for full-time employees.",
        metadata=ChunkMetadata(
            doc_id="doc_leave",
            document_name="Leave_Policy.pdf",
            page_number=3,
            chunk_index=0,
            char_length=47,
            section_title="Annual Leave",
            extra_metadata={"department": "HR"},
        ),
    )
    c2 = DocumentChunk(
        chunk_id="chunk_2",
        content="IT remote work equipment security requirements.",
        metadata=ChunkMetadata(
            doc_id="doc_it",
            document_name="IT_Security.pdf",
            page_number=1,
            chunk_index=0,
            char_length=48,
            section_title="Equipment",
            extra_metadata={"department": "IT"},
        ),
    )
    return [c1, c2]


def test_matches_filter(mock_chunks):
    c1, c2 = mock_chunks
    flt_hr = MetadataFilter(document_names=["Leave_Policy.pdf"])
    assert matches_filter(c1, flt_hr) is True
    assert matches_filter(c2, flt_hr) is False


def test_apply_filters(mock_chunks):
    c1, c2 = mock_chunks
    scored_candidates = [(c1, 0.92), (c2, 0.45)]
    flt = MetadataFilter(department="IT")
    filtered = apply_filters(scored_candidates, flt)

    assert len(filtered) == 1
    assert filtered[0][0].chunk_id == "chunk_2"
    assert filtered[0][1] == 0.45


def test_similarity_search_delegation(mock_chunks):
    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_store.search.return_value = [(mock_chunks[0], 0.88)]

    searcher = SimilaritySearchService(vector_store=mock_store, embedding_model=mock_embedder)
    results = searcher.search(query="leave days", top_k=1)

    assert len(results) == 1
    assert results[0][0].chunk_id == "chunk_1"
    mock_embedder.embed_query.assert_called_once_with("leave days")
    mock_store.search.assert_called_once()