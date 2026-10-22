"""FAISS vector store implementation with persistent metadata mapping."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import faiss
import numpy as np

from src.domain.chunks import ChunkMetadata, DocumentChunk
from src.utils.file_utils import ensure_directory, load_json, save_json
from src.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FaissVectorStore(BaseVectorStore):
    """FAISS-backed vector store using IndexFlatIP for cosine similarity."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        # Internal sequential ID -> serialized chunk dict
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add dense vectors to FAISS and map their sequential integer IDs to metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        if len(chunks) == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)

        # Store metadata under current sequential indices
        for chunk in chunks:
            self.metadata_store[self._next_id] = chunk.to_vector_payload()
            self._next_id += 1

        logger.info(f"Added {len(chunks)} chunks. Total vectors in index: {self.total_vectors()}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search top-K nearest neighbors using cosine similarity (inner product)."""
        if self.total_vectors() == 0:
            return []

        # Ensure 2D float32 array
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        k = min(top_k, self.total_vectors())
        scores, indices = self.index.search(query_vector, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue  # FAISS padding when fewer vectors exist than k

            payload = self.metadata_store.get(int(idx))
            if not payload:
                continue

            metadata = ChunkMetadata(
                doc_id=payload["doc_id"],
                document_name=payload["document_name"],
                page_number=payload.get("page_number"),
                chunk_index=payload["chunk_index"],
                char_length=len(payload["content"]),
                section_title=payload.get("section_title"),
            )

            chunk = DocumentChunk(
                chunk_id=payload["chunk_id"],
                content=payload["content"],
                metadata=metadata,
            )
            results.append((chunk, float(score)))

        return results

    def save(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """Persist FAISS index binary and json metadata store."""
        idx_p = Path(index_path)
        meta_p = Path(metadata_path)

        ensure_directory(idx_p.parent)
        ensure_directory(meta_p.parent)

        faiss.write_index(self.index, str(idx_p))
        save_json(self.metadata_store, meta_p)
        logger.info(f"Saved FAISS index to {idx_p} and metadata to {meta_p}")

    def load(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """Load index binary and JSON metadata."""
        idx_p = Path(index_path)
        meta_p = Path(metadata_path)

        if not idx_p.exists() or not meta_p.exists():
            raise FileNotFoundError(
                f"Missing index files. Ensure both exist: {idx_p} and {meta_p}"
            )

        self.index = faiss.read_index(str(idx_p))
        raw_meta = load_json(meta_p)
        # Convert JSON string keys back to integers
        self.metadata_store = {int(k): v for k, v in raw_meta.items()}
        self._next_id = len(self.metadata_store)
        self.dimension = self.index.d
        logger.info(f"Loaded {self.total_vectors()} vectors from {idx_p}")

    def total_vectors(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()
        self._next_id = 0