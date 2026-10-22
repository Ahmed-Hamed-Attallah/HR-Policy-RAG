"""Cross-encoder re-ranking for refining candidate passage relevance."""

import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder

from src.config.model_config import RerankerConfig, model_settings
from src.domain.responses import RetrievedChunk

logger = logging.getLogger(__name__)


class PolicyReranker:
    """Neural re-ranker based on cross-encoder scoring for high precision passage selection."""

    _instance: Optional["PolicyReranker"] = None

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or model_settings.get_reranker_config()
        self.model: Optional[CrossEncoder] = None

        if self.config.enabled:
            logger.info(f"Loading Cross-Encoder re-ranker: {self.config.model_name}")
            try:
                self.model = CrossEncoder(self.config.model_name, device="cpu")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder ({self.config.model_name}): {e}")
                self.model = None

    @classmethod
    def get_instance(cls, config: Optional[RerankerConfig] = None) -> "PolicyReranker":
        """Singleton accessor to preserve memory."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Score (query, chunk) pairs and return top_k candidates sorted by cross-encoder score."""
        if not candidates:
            return []

        limit = top_k or self.config.top_k

        # If reranker is disabled or model failed to load, fallback to similarity order
        if not self.config.enabled or self.model is None:
            logger.debug("Reranker bypassed; returning top similarity candidates.")
            return candidates[:limit]

        # Formulate query-passage pairs for cross-encoding
        pairs = [(query, c.chunk.content) for c in candidates]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error executing cross-encoder prediction: {e}")
            return candidates[:limit]

        # Attach rerank score to each retrieved chunk
        scored_candidates: List[RetrievedChunk] = []
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
            scored_candidates.append(candidate)

        # Sort descending by rerank score
        scored_candidates.sort(
            key=lambda x: x.rerank_score if x.rerank_score is not None else -999.0,
            reverse=True,
        )

        final_ranked = scored_candidates[:limit]
        logger.debug(
            f"Reranked {len(candidates)} candidates down to {len(final_ranked)} chunks."
        )
        return final_ranked