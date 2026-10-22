"""Unit tests for document chunking and text splitting strategies."""

import pytest
from src.chunking.text_splitter import ContextualTextSplitter
from src.config.model_config import ChunkingConfig
from src.domain.documents import Document, DocumentMetadata, DocumentType, ParsedPage


@pytest.fixture
def sample_document() -> Document:
    text_p1 = (
        "SECTION 1: WORKING HOURS AND ATTENDANCE\n"
        "Standard official working hours are from 9:00 AM to 5:00 PM Sunday through Thursday.\n"
        "Employees must record attendance using the biometric access terminal."
    )
    text_p2 = (
        "SECTION 2: ANNUAL LEAVE AND VACATION\n"
        "Employees are entitled to 21 working days of paid annual leave per calendar year.\n"
        "Leave requests must be submitted at least two weeks in advance. "
        "Except when an emergency occurs, verbal notification to the direct manager is required."
    )
    return Document(
        doc_id="test_doc_001",
        content=f"{text_p1}\n\n{text_p2}",
        pages=[
            ParsedPage(page_number=1, text=text_p1),
            ParsedPage(page_number=2, text=text_p2),
        ],
        metadata=DocumentMetadata(
            source_path="/fake/HR_Policy.pdf",
            file_name="HR_Policy.pdf",
            file_type=DocumentType.PDF,
            file_size_bytes=1024,
            file_hash="fakehash123",
            total_pages=2,
            title="HR General Policy",
        ),
    )


def test_chunking_generates_valid_chunks(sample_document: Document):
    config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
    splitter = ContextualTextSplitter(config=config)
    chunks = splitter.split_document(sample_document)

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"test_doc_001_chunk_{idx}"
        assert chunk.metadata.doc_id == "test_doc_001"
        assert chunk.metadata.document_name == "HR_Policy.pdf"
        assert chunk.metadata.page_number in [1, 2]
        assert len(chunk.content.strip()) > 0


def test_chunking_empty_document():
    empty_doc = Document(
        doc_id="empty_001",
        content="",
        pages=[],
        metadata=DocumentMetadata(
            source_path="/fake/empty.pdf",
            file_name="empty.pdf",
            file_type=DocumentType.PDF,
            file_size_bytes=0,
            file_hash="none",
            total_pages=0,
        ),
    )
    splitter = ContextualTextSplitter()
    chunks = splitter.split_document(empty_doc)
    assert chunks == []