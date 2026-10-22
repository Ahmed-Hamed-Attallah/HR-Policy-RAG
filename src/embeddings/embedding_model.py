"""Dense vector embedding model wrapper using Sentence-Transformers."""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.model_config import EmbeddingConfig, model_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thread-safe embedding generator with batch processing and normalization."""

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or model_settings.get_embedding_config()
        logger.info(
            f"Loading embedding model: {self.config.model_name} on device: {self.config.device}"
        )
        self.model = SentenceTransformer(
            model_name_or_path=self.config.model_name,
            device=self.config.device,
        )

    @classmethod
    def get_instance(cls, config: Optional[EmbeddingConfig] = None) -> "EmbeddingModel":
        """Singleton accessor to prevent reloading the neural model across modules."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text snippets into a normalized 2D numpy array (N, Dimension)."""
        if not texts:
            return np.empty((0, self.config.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            sentences=texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1D vector."""
        if not query.strip():
            raise ValueError("Query string cannot be empty for embedding generation.")

        # BGE models benefit from instruction prefixing on queries
        bge_prefix = "Represent this sentence for searching relevant passages: "
        formatted_query = (
            f"{bge_prefix}{query}" if "bge" in self.config.model_name.lower() else query
        )

        embedding = self.model.encode(
            sentences=formatted_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Return the actual dimensionality of the embedding vector."""
        return self.config.dimension