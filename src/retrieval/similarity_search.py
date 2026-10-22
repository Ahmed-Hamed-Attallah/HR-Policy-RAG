"""Dense similarity search engine combining vector indexing with metadata filtering."""

import logging
from typing import List, Optional, Tuple

from src.config.model_config import RetrievalConfig, retrieval_config
from src.domain.chunks import DocumentChunk
from src.retrieval.filters import MetadataFilter, apply_filters
from src.vector_store.index_manager import IndexManager

logger = logging.getLogger(__name__)


class SimilaritySearchEngine:
    """Executes dense vector nearest-neighbor search with score thresholding and filtering."""

    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[RetrievalConfig] = None,
    ):
        self.index_manager = index_manager
        self.config = config or retrieval_config

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_criteria: Optional[MetadataFilter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top matching chunks for a text query, applying filters and score pruning."""
        cleaned_query = query.strip()
        if not cleaned_query:
            logger.warning("Empty query passed to similarity search.")
            return []

        target_k = top_k or self.config.top_k
        min_threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.score_threshold
        )

        # Over-fetch candidate pool if filtering is requested to prevent under-population
        fetch_limit = target_k * 3 if (filter_criteria and not filter_criteria.is_empty()) else target_k

        # 1. Execute FAISS similarity retrieval
        raw_candidates = self.index_manager.search(query=cleaned_query, top_k=fetch_limit)

        if not raw_candidates:
            logger.info("Vector search returned zero candidates.")
            return []

        # 2. Apply metadata filters if provided
        filtered_candidates = apply_filters(raw_candidates, filter_criteria)

        # 3. Filter by similarity score threshold
        passed_threshold: List[Tuple[DocumentChunk, float]] = [
            (chunk, score)
            for chunk, score in filtered_candidates
            if score >= min_threshold
        ]

        # 4. Limit to requested top_k
        final_results = passed_threshold[:target_k]
        logger.debug(
            f"Query '{cleaned_query[:30]}...' retrieved {len(final_results)} chunks "
            f"after filtering (from {len(raw_candidates)} initial hits)."
        )
        return final_results