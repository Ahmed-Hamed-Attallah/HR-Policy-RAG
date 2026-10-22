"""Assembles retrieved policy passages into structured context blocks for LLM prompts."""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from src.domain.responses import RetrievedChunk
from src.utils.text_utils import estimate_token_count

logger = logging.getLogger(__name__)


class ContextAssemblyResult(BaseModel):
    """Holds formatted context string and assembly diagnostics."""

    assembled_text: str = Field(description="Formatted context ready for prompt injection")
    included_chunks: List[RetrievedChunk] = Field(
        default_factory=list, description="Chunks that fit within token constraints"
    )
    total_tokens_estimated: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)


class ContextBuilder:
    """Formats and budget-caps retrieved chunks into an LLM-ready context block."""

    def __init__(self, default_max_context_tokens: int = 2500):
        self.default_max_context_tokens = default_max_context_tokens

    def format_single_chunk(self, index: int, item: RetrievedChunk) -> str:
        """Format an individual chunk with clear attribution headers."""
        meta = item.chunk.metadata
        page_str = f", Page {meta.page_number}" if meta.page_number else ""
        section_str = f" | Section: {meta.section_title}" if meta.section_title else ""

        header = f"[Source {index}: {meta.document_name}{page_str}{section_str}]"
        return f"{header}\n{item.chunk.content.strip()}"

    def build_context(
        self,
        retrieved_chunks: List[RetrievedChunk],
        max_context_tokens: Optional[int] = None,
    ) -> ContextAssemblyResult:
        """Concatenate chunks respecting maximum token allowances."""
        if not retrieved_chunks:
            return ContextAssemblyResult(
                assembled_text="No relevant HR policy passages found.",
                included_chunks=[],
                total_tokens_estimated=0,
                chunk_count=0,
            )

        token_budget = max_context_tokens or self.default_max_context_tokens
        formatted_blocks: List[str] = []
        included_chunks: List[RetrievedChunk] = []
        accumulated_tokens = 0

        for idx, item in enumerate(retrieved_chunks, start=1):
            block_text = self.format_single_chunk(idx, item)
            block_tokens = estimate_token_count(block_text)

            # Check if adding this block violates the context budget
            if accumulated_tokens + block_tokens > token_budget and len(formatted_blocks) > 0:
                logger.info(
                    f"Context budget reached ({accumulated_tokens}/{token_budget} tokens). "
                    f"Skipping remaining {len(retrieved_chunks) - len(included_chunks)} chunks."
                )
                break

            formatted_blocks.append(block_text)
            included_chunks.append(item)
            accumulated_tokens += block_tokens

        separator = "\n\n" + ("=" * 40) + "\n\n"
        assembled_context = separator.join(formatted_blocks)

        return ContextAssemblyResult(
            assembled_text=assembled_context,
            included_chunks=included_chunks,
            total_tokens_estimated=accumulated_tokens,
            chunk_count=len(included_chunks),
        )