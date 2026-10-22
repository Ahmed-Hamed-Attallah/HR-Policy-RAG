"""Metadata filtering logic for narrowing vector retrieval results."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.domain.chunks import DocumentChunk


class MetadataFilter(BaseModel):
    """Declarative criteria for filtering chunks based on metadata attributes."""

    doc_ids: Optional[List[str]] = Field(
        default=None, description="Filter chunks matching specific parent document IDs"
    )
    document_names: Optional[List[str]] = Field(
        default=None, description="Filter chunks matching document file names"
    )
    pages: Optional[List[int]] = Field(
        default=None, description="Filter chunks from specific page numbers"
    )
    section_titles: Optional[List[str]] = Field(
        default=None, description="Filter chunks from specific section/policy headers"
    )
    custom_filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Key-value filters for chunk extra_metadata attributes"
    )

    def is_empty(self) -> bool:
        """Check if any filtering condition has been defined."""
        return not any(
            [
                self.doc_ids,
                self.document_names,
                self.pages,
                self.section_titles,
                self.custom_filters,
            ]
        )


def matches_filter(chunk: DocumentChunk, filter_criteria: MetadataFilter) -> bool:
    """Evaluate whether a single DocumentChunk satisfies the metadata filter criteria."""
    if filter_criteria.is_empty():
        return True

    meta = chunk.metadata

    # 1. Document ID Filter
    if filter_criteria.doc_ids and meta.doc_id not in filter_criteria.doc_ids:
        return False

    # 2. Document Name Filter (case-insensitive substring or exact match)
    if filter_criteria.document_names:
        matched_name = any(
            target.lower() in meta.document_name.lower()
            for target in filter_criteria.document_names
        )
        if not matched_name:
            return False

    # 3. Page Number Filter
    if filter_criteria.pages:
        if meta.page_number is None or meta.page_number not in filter_criteria.pages:
            return False

    # 4. Section Title Filter
    if filter_criteria.section_titles:
        if not meta.section_title:
            return False
        matched_section = any(
            target.lower() in meta.section_title.lower()
            for target in filter_criteria.section_titles
        )
        if not matched_section:
            return False

    # 5. Dynamic Key-Value Filters on inherited or extra metadata
    if filter_criteria.custom_filters:
        extra = meta.extra_metadata or {}
        for key, expected_value in filter_criteria.custom_filters.items():
            actual_val = extra.get(key)
            if actual_val is None:
                return False
            if isinstance(expected_value, list):
                if actual_val not in expected_value:
                    return False
            elif str(actual_val).lower() != str(expected_value).lower():
                return False

    return True


def apply_filters(
    candidates: List[Tuple[DocumentChunk, float]],
    filter_criteria: Optional[MetadataFilter] = None,
) -> List[Tuple[DocumentChunk, float]]:
    """Filter retrieved (chunk, score) pairs according to specified metadata constraints."""
    if not filter_criteria or filter_criteria.is_empty():
        return candidates

    return [
        (chunk, score)
        for chunk, score in candidates
        if matches_filter(chunk, filter_criteria)
    ]