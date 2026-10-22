"""Citation formatting, extraction, and deduplication utilities."""

import re
from typing import List, Set, Tuple
from src.domain.responses import Citation, RetrievedChunk


class CitationManager:
    """Manages creation, deduplication, and verification of citations."""

    SOURCE_TAG_PATTERN = re.compile(r"\[(?:Source\s+)?(\d+)\]", re.IGNORECASE)

    @classmethod
    def build_citations(cls, retrieved_chunks: List[RetrievedChunk]) -> List[Citation]:
        """Construct a list of Citation contracts from retrieved and reranked chunks."""
        citations: List[Citation] = []

        for item in retrieved_chunks:
            meta = item.chunk.metadata
            # Create a concise snippet preview (first 180 chars)
            preview_snippet = (
                item.chunk.content[:180].replace("\n", " ").strip() + "..."
                if len(item.chunk.content) > 180
                else item.chunk.content.replace("\n", " ").strip()
            )

            # Use rerank score if available, otherwise vector similarity score
            effective_score = (
                item.rerank_score
                if item.rerank_score is not None
                else item.similarity_score
            )

            citations.append(
                Citation(
                    source_name=meta.document_name,
                    page_number=meta.page_number,
                    snippet=preview_snippet,
                    score=round(effective_score, 4) if effective_score is not None else None,
                )
            )

        return cls.deduplicate_citations(citations)

    @staticmethod
    def deduplicate_citations(citations: List[Citation]) -> List[Citation]:
        """Remove redundant citations referencing identical document and page combinations."""
        seen: Set[Tuple[str, int | None]] = set()
        unique_citations: List[Citation] = []

        for c in citations:
            key = (c.source_name, c.page_number)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        return unique_citations

    @classmethod
    def filter_citations_referenced_in_text(
        cls, answer_text: str, all_citations: List[Citation]
    ) -> List[Citation]:
        """Extract only those citations explicitly referenced as [Source N] or [N] by LLM."""
        matches = cls.SOURCE_TAG_PATTERN.findall(answer_text)
        if not matches:
            # If LLM did not adopt strict bracket tags, return all candidate context citations
            return all_citations

        referenced_indices = {int(m) for m in matches if m.isdigit()}
        active_citations: List[Citation] = []

        for idx, citation in enumerate(all_citations, start=1):
            if idx in referenced_indices:
                active_citations.append(citation)

        # Fallback: if tags matched invalid numbers, return all deduplicated citations
        return active_citations if active_citations else all_citations