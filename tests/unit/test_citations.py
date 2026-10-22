"""Unit tests for citation construction, deduplication, and citation attribution."""

from src.domain.chunks import ChunkMetadata, DocumentChunk
from src.domain.responses import Citation, RetrievedChunk
from src.rag.citations import CitationManager


def test_build_citations():
    chunk1 = DocumentChunk(
        chunk_id="c1",
        content="Working hours are from 9:00 AM to 5:00 PM Sunday through Thursday.",
        metadata=ChunkMetadata(
            doc_id="doc_1",
            document_name="Handbook.pdf",
            page_number=4,
            chunk_index=0,
            char_length=65,
        ),
    )
    rc1 = RetrievedChunk(chunk=chunk1, similarity_score=0.89)

    citations = CitationManager.build_citations([rc1])
    assert len(citations) == 1
    assert citations[0].source_name == "Handbook.pdf"
    assert citations[0].page_number == 4
    assert len(citations[0].snippet) > 0


def test_filter_citations_referenced_in_text():
    all_citations = [
        Citation(source_name="Leave_Policy.pdf", page_number=2, snippet="21 days leave"),
        Citation(source_name="IT_Policy.pdf", page_number=5, snippet="VPN security rules"),
    ]

    answer_with_ref = (
        "Employees get 21 days leave per year as stated in [Leave_Policy.pdf, Page 2]."
    )
    filtered = CitationManager.filter_citations_referenced_in_text(answer_with_ref, all_citations)

    # Should retain the Leave_Policy citation and drop the unreferenced IT citation
    assert len(filtered) == 1
    assert filtered[0].source_name == "Leave_Policy.pdf"