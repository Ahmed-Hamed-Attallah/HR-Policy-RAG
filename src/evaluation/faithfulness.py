"""Faithfulness and groundedness verification engine to detect hallucinations."""

import re
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field

from src.embeddings.embedding_model import EmbeddingModel
from src.utils.text_utils import normalize_whitespace


class FaithfulnessResult(BaseModel):
    """Detailed diagnostic results of answer grounding against context passages."""
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="Ratio of verified sentences")
    total_statements: int = Field(ge=0)
    supported_statements: int = Field(ge=0)
    unsupported_statements: List[str] = Field(default_factory=list)
    is_faithful: bool = Field(description="True if faithfulness score reaches target threshold")


class FaithfulnessEvaluator:
    """Evaluates whether generated answers are strictly faithful to retrieved context passages."""

    # Common boilerplate phrases to disregard during sentence claim verification
    BOILERPLATE_PATTERNS = [
        re.compile(r"^based on (?:the )?(?:available )?(?:hr )?policy", re.IGNORECASE),
        re.compile(r"^according to (?:the )?documents?", re.IGNORECASE),
        re.compile(r"^please refer to", re.IGNORECASE),
        re.compile(r"^for more information", re.IGNORECASE),
    ]

    def __init__(self, threshold: float = 0.70, similarity_cutoff: float = 0.65):
        self.threshold = threshold
        self.similarity_cutoff = similarity_cutoff
        self.embedder = EmbeddingModel.get_instance()

    def split_into_statements(self, text: str) -> List[str]:
        """Decompose text into non-trivial proposition sentences."""
        cleaned = normalize_whitespace(text)
        # Split on sentence terminals while respecting abbreviations
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        valid_statements: List[str] = []

        for sent in raw_sentences:
            s = sent.strip()
            # Ignore tiny fragments or citation tags
            if len(s) < 15 or re.match(r"^\[Source\s+\d+\]$", s):
                continue
            # Strip preamble boilerplate
            is_bp = any(bp.match(s) for bp in self.BOILERPLATE_PATTERNS)
            if not is_bp:
                valid_statements.append(s)

        return valid_statements

    def is_statement_grounded(self, statement: str, context_chunks: List[str]) -> bool:
        """Check whether an individual statement is semantically and lexically supported."""
        if not context_chunks:
            return False

        statement_lower = statement.lower()
        stmt_words = set(re.findall(r"\w{4,}", statement_lower))

        # 1. Lexical exact or high keyword overlap
        for chunk in context_chunks:
            chunk_lower = chunk.lower()
            if statement_lower in chunk_lower:
                return True
            if stmt_words:
                overlap = sum(1 for w in stmt_words if w in chunk_lower)
                if overlap / len(stmt_words) >= 0.75:
                    return True

        # 2. Dense semantic similarity validation against context passages
        stmt_emb = self.embedder.embed_query(statement)
        chunk_embs = self.embedder.embed_texts(context_chunks)

        dots = np.dot(chunk_embs, stmt_emb)
        norms = np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(stmt_emb)
        sims = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)

        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
        return max_sim >= self.similarity_cutoff

    def evaluate(self, answer: str, context_chunks: List[str]) -> FaithfulnessResult:
        """Evaluate faithfulness by auditing all substantive claims made in the answer."""
        statements = self.split_into_statements(answer)
        if not statements:
            # If answer contains no substantive claims (or is a fallback refusal), consider safe
            return FaithfulnessResult(
                faithfulness_score=1.0,
                total_statements=0,
                supported_statements=0,
                unsupported_statements=[],
                is_faithful=True,
            )

        supported_count = 0
        unsupported: List[str] = []

        for stmt in statements:
            if self.is_statement_grounded(stmt, context_chunks):
                supported_count += 1
            else:
                unsupported.append(stmt)

        score = float(supported_count / len(statements))
        is_faithful = score >= self.threshold

        return FaithfulnessResult(
            faithfulness_score=round(score, 4),
            total_statements=len(statements),
            supported_statements=supported_count,
            unsupported_statements=unsupported,
            is_faithful=is_faithful,
        )