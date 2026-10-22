"""Orchestrator for managing index lifecycles, embeddings generation, and persistence."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.chunking.text_splitter import ContextualTextSplitter
from src.config.model_config import EmbeddingConfig
from src.config.settings import settings
from src.domain.chunks import DocumentChunk
from src.domain.documents import Document
from src.embeddings.embedding_model import EmbeddingModel
from src.vector_store.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


class IndexManager:
    """Coordinates document splitting, embedding generation, and vector index operations."""

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.index_path = Path(index_path or settings.VECTOR_INDEX_PATH)
        self.metadata_path = Path(metadata_path or settings.METADATA_STORE_PATH)
        self.embedding_model = embedding_model or EmbeddingModel.get_instance()
        self.vector_store = FaissVectorStore(dimension=self.embedding_model.dimension)
        self.text_splitter = ContextualTextSplitter()

    def load_or_create(self) -> None:
        """Load index from disk if artifacts exist, otherwise initialize an empty store."""
        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("Found existing vector index. Loading from storage...")
            self.vector_store.load(self.index_path, self.metadata_path)
        else:
            logger.info("No existing vector store found. Initialized new empty index.")

    def index_documents(self, documents: List[Document]) -> int:
        """Process documents into chunks, generate embeddings, and persist to FAISS."""
        if not documents:
            logger.warning("No documents supplied for indexing.")
            return 0

        # 1. Chunk all documents
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            chunks = self.text_splitter.split_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("Documents yielded zero chunks after splitting.")
            return 0

        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} documents.")

        # 2. Extract texts and generate dense embeddings in batches
        texts = [c.content for c in all_chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        # 3. Add to vector store and persist
        self.vector_store.add_chunks(all_chunks, embeddings)
        self.save()

        return len(all_chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Vectorize user query and retrieve top_k matching chunks with similarity scores."""
        query_vector = self.embedding_model.embed_query(query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k)

    def save(self) -> None:
        """Persist vector index and metadata mapping to designated storage paths."""
        self.vector_store.save(self.index_path, self.metadata_path)

    def clear_index(self) -> None:
        """Purge in-memory and disk-persisted vector artifacts."""
        self.vector_store.clear()
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        logger.info("Cleared vector index and removed stored artifact files.")

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics about current vector storage state."""
        return {
            "total_vectors": self.vector_store.total_vectors(),
            "vector_dimension": self.vector_store.dimension,
            "embedding_model": self.embedding_model.config.model_name,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "is_persisted": self.index_path.exists() and self.metadata_path.exists(),
        }