"""Chunk metadata extraction and enrichment utilities."""

import re
from typing import Any, Dict, Optional
from src.domain.chunks import ChunkMetadata
from src.utils.text_utils import estimate_token_count


HEADER_PATTERNS = [
    re.compile(r"^(?:#{1,4}\s+|(?:\d+\.)+\s+)([^\n]+)", re.MULTILINE),
    re.compile(r"^([A-Z][A-Za-z0-9\s]{3,40}):\s*$", re.MULTILINE),
]


def build_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a deterministic unique chunk ID from document ID and sequential index."""
    return f"{doc_id}_c{chunk_index:04d}"


def extract_section_header(text: str) -> Optional[str]:
    """Inspect text block to find an embedded Markdown or numbered section header."""
    for pattern in HEADER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def create_chunk_metadata(
    doc_id: str,
    document_name: str,
    chunk_index: int,
    text: str,
    page_number: Optional[int] = None,
    inherited_metadata: Optional[Dict[str, Any]] = None,
) -> ChunkMetadata:
    """Build a complete, validated ChunkMetadata instance."""
    section_title = extract_section_header(text)
    token_count = estimate_token_count(text)
    char_length = len(text)
    extra_metadata = inherited_metadata.copy() if inherited_metadata else {}

    return ChunkMetadata(
        doc_id=doc_id,
        document_name=document_name,
        page_number=page_number,
        chunk_index=chunk_index,
        char_length=char_length,
        token_count=token_count,
        section_title=section_title,
        extra_metadata=extra_metadata,
    )