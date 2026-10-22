"""Context-aware text chunking for documents and parsed pages."""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.chunk_metadata import build_chunk_id, create_chunk_metadata
from src.config.model_config import ChunkingConfig, chunking_config
from src.domain.chunks import DocumentChunk
from src.domain.documents import Document, ParsedPage


class ContextualTextSplitter:
    """Splits full documents or page streams into structured DocumentChunks."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or chunking_config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_page(
        self,
        page: ParsedPage,
        doc_id: str,
        document_name: str,
        start_chunk_index: int = 0,
    ) -> List[DocumentChunk]:
        """Split a single page while preserving page attribution."""
        raw_chunks = self.splitter.split_text(page.text)
        chunks: List[DocumentChunk] = []

        for offset, text_content in enumerate(raw_chunks):
            current_index = start_chunk_index + offset
            chunk_id = build_chunk_id(doc_id, current_index)
            metadata = create_chunk_metadata(
                doc_id=doc_id,
                document_name=document_name,
                chunk_index=current_index,
                text=text_content,
                page_number=page.page_number,
                inherited_metadata=page.metadata,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=text_content,
                    metadata=metadata,
                )
            )
        return chunks

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Split an entire Document page-by-page to guarantee exact page references."""
        all_chunks: List[DocumentChunk] = []
        chunk_counter = 0

        # If individual parsed pages exist, split by page to preserve exact citation locations
        if document.pages:
            for page in document.pages:
                page_chunks = self.split_page(
                    page=page,
                    doc_id=document.doc_id,
                    document_name=document.metadata.file_name,
                    start_chunk_index=chunk_counter,
                )
                all_chunks.extend(page_chunks)
                chunk_counter += len(page_chunks)
        else:
            # Fallback to document full text if no page breakdown is available
            raw_splits = self.splitter.split_text(document.content)
            for idx, text_block in enumerate(raw_splits):
                chunk_id = build_chunk_id(document.doc_id, idx)
                metadata = create_chunk_metadata(
                    doc_id=document.doc_id,
                    document_name=document.metadata.file_name,
                    chunk_index=idx,
                    text=text_block,
                    page_number=None,
                    inherited_metadata=document.metadata.extra_attributes,
                )
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=text_block,
                        metadata=metadata,
                    )
                )

        return all_chunks