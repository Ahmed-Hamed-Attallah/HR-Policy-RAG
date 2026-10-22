"""Information retrieval evaluation metrics: HitRate, Recall, Precision, MRR, nDCG."""

from typing import Any, Dict, List, Set, Union
import numpy as np
from pydantic import BaseModel, Field


class RetrievalEvaluationResult(BaseModel):
    """Container for multi-cutoff retrieval evaluation metrics."""
    k: int
    hit_rate: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)


class RetrievalMetrics:
    """Calculates classical Information Retrieval (IR) metrics against ground truth IDs."""

    @staticmethod
    def calculate_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Return 1.0 if any ground truth item exists within top-k, else 0.0."""
        top_k = retrieved_ids[:k]
        return 1.0 if any(item_id in ground_truth_ids for item_id in top_k) else 0.0

    @staticmethod
    def calculate_recall(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Ratio of relevant items retrieved within top-k over total ground truth items."""
        if not ground_truth_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant_retrieved = top_k.intersection(ground_truth_ids)
        return float(len(relevant_retrieved) / len(ground_truth_ids))

    @staticmethod
    def calculate_precision(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Ratio of relevant items within top-k over k."""
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for item in top_k if item in ground_truth_ids)
        return float(relevant_count / k)

    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Reciprocal rank of the first relevant item retrieved within top-k."""
        for rank_idx, item_id in enumerate(retrieved_ids[:k], start=1):
            if item_id in ground_truth_ids:
                return float(1.0 / rank_idx)
        return 0.0

    @staticmethod
    def calculate_ndcg(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain with binary relevance at cutoff k."""
        top_k = retrieved_ids[:k]
        if not top_k or not ground_truth_ids:
            return 0.0

        dcg = 0.0
        for rank_idx, item_id in enumerate(top_k, start=1):
            rel = 1.0 if item_id in ground_truth_ids else 0.0
            dcg += rel / np.log2(rank_idx + 1)

        # Ideal DCG: all ground truth items packed at the highest ranks up to k
        num_relevant = min(len(ground_truth_ids), k)
        idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, num_relevant + 1))

        return float(dcg / idcg) if idcg > 0.0 else 0.0

    @classmethod
    def evaluate_retrieval(
        cls,
        retrieved_ids: List[str],
        ground_truth_ids: Union[List[str], Set[str]],
        k: int = 5,
    ) -> RetrievalEvaluationResult:
        """Compute complete suite of IR metrics at cutoff k for a single query."""
        gt_set = set(ground_truth_ids)
        return RetrievalEvaluationResult(
            k=k,
            hit_rate=round(cls.calculate_hit_rate(retrieved_ids, gt_set, k), 4),
            recall=round(cls.calculate_recall(retrieved_ids, gt_set, k), 4),
            precision=round(cls.calculate_precision(retrieved_ids, gt_set, k), 4),
            mrr=round(cls.calculate_mrr(retrieved_ids, gt_set, k), 4),
            ndcg=round(cls.calculate_ndcg(retrieved_ids, gt_set, k), 4),
        )