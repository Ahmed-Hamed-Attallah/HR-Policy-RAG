"""Abstract vector store contract for indexing and similarity search."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np

from src.domain.chunks import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract interface defining required vector store operations."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Insert textual chunks and corresponding dense vectors into store."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Find top_k nearest neighbor chunks given a query vector, returning chunk and score."""
        pass

    @abstractmethod
    def save(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """Persist index binaries and serialized metadata payloads to disk."""
        pass

    @abstractmethod
    def load(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """Load index binaries and serialized metadata payloads from disk."""
        pass

    @abstractmethod
    def total_vectors(self) -> int:
        """Return total number of vectors indexed in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Reset and purge all indexed vectors and cached metadata."""
        pass