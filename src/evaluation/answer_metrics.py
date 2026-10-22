"""Answer quality evaluation: Token F1, Exact Match, ROUGE-L, and Semantic Similarity."""

import re
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from src.embeddings.embedding_model import EmbeddingModel
from src.utils.text_utils import normalize_whitespace


class AnswerEvaluationResult(BaseModel):
    """Container for answer quality comparison against ground truth."""
    exact_match: float = Field(ge=0.0, le=1.0)
    token_f1: float = Field(ge=0.0, le=1.0)
    rouge_l: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)


class AnswerMetrics:
    """Evaluates generated responses against reference ground truth answers."""

    _embedder: Optional[EmbeddingModel] = None

    @classmethod
    def get_embedder(cls) -> EmbeddingModel:
        """Lazy load shared embedding instance for semantic comparison."""
        if cls._embedder is None:
            cls._embedder = EmbeddingModel.get_instance()
        return cls._embedder

    @staticmethod
    def _normalize_tokens(text: str) -> List[str]:
        """Sanitize text and extract normalized alphanumeric token list."""
        cleaned = normalize_whitespace(text).lower()
        tokens = re.findall(r"\w+", cleaned)
        return tokens

    @classmethod
    def exact_match_score(cls, prediction: str, reference: str) -> float:
        """Compute binary exact match after whitespace and case normalization."""
        pred_norm = " ".join(cls._normalize_tokens(prediction))
        ref_norm = " ".join(cls._normalize_tokens(reference))
        return 1.0 if pred_norm == ref_norm else 0.0

    @classmethod
    def token_f1_score(cls, prediction: str, reference: str) -> float:
        """Compute token-level F1 score between prediction and reference."""
        pred_tokens = cls._normalize_tokens(prediction)
        ref_tokens = cls._normalize_tokens(reference)

        if not pred_tokens or not ref_tokens:
            return 1.0 if pred_tokens == ref_tokens else 0.0

        common: Dict[str, int] = {}
        ref_counts: Dict[str, int] = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1

        for t in pred_tokens:
            if ref_counts.get(t, 0) > 0:
                common[t] = common.get(t, 0) + 1
                ref_counts[t] -= 1

        num_common = sum(common.values())
        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return float((2 * precision * recall) / (precision + recall))

    @classmethod
    def rouge_l_score(cls, prediction: str, reference: str) -> float:
        """Calculate sentence-level Longest Common Subsequence (LCS) F1 score."""
        p_tokens = cls._normalize_tokens(prediction)
        r_tokens = cls._normalize_tokens(reference)
        m, n = len(p_tokens), len(r_tokens)

        if m == 0 or n == 0:
            return 0.0

        # Dynamic programming table for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if p_tokens[i] == r_tokens[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

        lcs_len = dp[m][n]
        if lcs_len == 0:
            return 0.0

        precision = lcs_len / m
        recall = lcs_len / n
        return float((2 * precision * recall) / (precision + recall))

    @classmethod
    def semantic_similarity(cls, prediction: str, reference: str) -> float:
        """Compute cosine similarity between dense embeddings of prediction and reference."""
        if not prediction.strip() or not reference.strip():
            return 0.0

        embedder = cls.get_embedder()
        pred_emb = embedder.embed_query(prediction)
        ref_emb = embedder.embed_query(reference)

        dot = np.dot(pred_emb, ref_emb)
        norm_pred = np.linalg.norm(pred_emb)
        norm_ref = np.linalg.norm(ref_emb)

        if norm_pred == 0.0 or norm_ref == 0.0:
            return 0.0

        similarity = float(dot / (norm_pred * norm_ref))
        # Bound score between 0.0 and 1.0
        return float(np.clip(similarity, 0.0, 1.0))

    @classmethod
    def evaluate_answer(cls, prediction: str, reference: str) -> AnswerEvaluationResult:
        """Execute full answer quality benchmark against reference text."""
        return AnswerEvaluationResult(
            exact_match=round(cls.exact_match_score(prediction, reference), 4),
            token_f1=round(cls.token_f1_score(prediction, reference), 4),
            rouge_l=round(cls.rouge_l_score(prediction, reference), 4),
            semantic_similarity=round(cls.semantic_similarity(prediction, reference), 4),
        )