"""Orchestrator for executing multi-metric RAG evaluations over benchmark datasets."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.rag.pipeline import RAGPipeline
from src.utils.file_utils import save_json

logger = logging.getLogger(__name__)


class BenchmarkItem(BaseModel):
    """Schema representing an evaluation question in data/evaluation/questions.jsonl."""
    query_id: str
    query: str
    expected_answer: Optional[str] = None
    expected_doc_ids: List[str] = Field(default_factory=list)
    expected_keywords: List[str] = Field(default_factory=list)
    category: Optional[str] = "general"


class SingleEvaluationReport(BaseModel):
    """Evaluation summary for a single query."""
    query_id: str
    query: str
    latency_ms: float
    retrieval_hit_rate: float
    retrieval_recall: float
    retrieval_mrr: float
    token_f1: Optional[float] = None
    semantic_similarity: Optional[float] = None
    faithfulness_score: float
    is_faithful: bool
    is_fallback: bool


class BenchmarkSummaryReport(BaseModel):
    """Aggregated metrics across the entire evaluation test suite."""
    total_queries: int
    mean_latency_ms: float
    mean_hit_rate: float
    mean_recall: float
    mean_mrr: float
    mean_token_f1: float
    mean_semantic_similarity: float
    mean_faithfulness: float
    fallback_rate: float
    itemized_results: List[SingleEvaluationReport]


class EvaluationRunner:
    """Executes evaluation benchmarks and generates quantitative assessment reports."""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        retrieval_k: int = 5,
        faithfulness_threshold: float = 0.70,
    ):
        self.pipeline = rag_pipeline or RAGPipeline()
        self.retrieval_k = retrieval_k
        self.faithfulness_evaluator = FaithfulnessEvaluator(threshold=faithfulness_threshold)

    @staticmethod
    def load_benchmark_dataset(dataset_path: Path) -> List[BenchmarkItem]:
        """Parse questions.jsonl into validated BenchmarkItem objects."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {dataset_path}")

        items: List[BenchmarkItem] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                try:
                    payload = json.loads(clean_line)
                    items.append(BenchmarkItem(**payload))
                except Exception as e:
                    logger.warning(f"Skipping malformed benchmark item at line {line_idx}: {e}")

        logger.info(f"Loaded {len(items)} benchmark test queries from {dataset_path.name}")
        return items

    def run_benchmark(self, dataset_path: Path) -> BenchmarkSummaryReport:
        """Run complete RAG evaluation cycle across all benchmark queries."""
        items = self.load_benchmark_dataset(dataset_path)
        reports: List[SingleEvaluationReport] = []

        for item in items:
            t0 = time.time()
            # 1. Execute end-to-end RAG Pipeline
            response = self.pipeline.run(query=item.query, retrieval_k=self.retrieval_k)
            latency_ms = (time.time() - t0) * 1000

            # 2. Evaluate Retrieval Performance
            retrieved_chunk_ids = [c.chunk.chunk_id for c in response.retrieved_chunks]
            retrieval_res = RetrievalMetrics.evaluate_retrieval(
                retrieved_ids=retrieved_chunk_ids,
                ground_truth_ids=item.expected_doc_ids,
                k=self.retrieval_k,
            )

            # 3. Evaluate Answer Quality if Reference is Available
            token_f1 = None
            semantic_sim = None
            if item.expected_answer:
                ans_res = AnswerMetrics.evaluate_answer(
                    prediction=response.answer,
                    reference=item.expected_answer,
                )
                token_f1 = ans_res.token_f1
                semantic_sim = ans_res.semantic_similarity

            # 4. Evaluate Faithfulness against Retrieved Context Passages
            context_texts = [c.chunk.content for c in response.retrieved_chunks]
            faith_res = self.faithfulness_evaluator.evaluate(
                answer=response.answer,
                context_chunks=context_texts,
            )

            reports.append(
                SingleEvaluationReport(
                    query_id=item.query_id,
                    query=item.query,
                    latency_ms=round(latency_ms, 2),
                    retrieval_hit_rate=retrieval_res.hit_rate,
                    retrieval_recall=retrieval_res.recall,
                    retrieval_mrr=retrieval_res.mrr,
                    token_f1=round(token_f1, 4) if token_f1 is not None else None,
                    semantic_similarity=round(semantic_sim, 4) if semantic_sim is not None else None,
                    faithfulness_score=faith_res.faithfulness_score,
                    is_faithful=faith_res.is_faithful,
                    is_fallback=response.is_fallback,
                )
            )

        # 5. Synthesize Global Aggregate Metrics
        total = len(reports)
        if total == 0:
            raise ValueError("No valid benchmark items evaluated.")

        valid_f1 = [r.token_f1 for r in reports if r.token_f1 is not None]
        valid_sim = [r.semantic_similarity for r in reports if r.semantic_similarity is not None]

        summary = BenchmarkSummaryReport(
            total_queries=total,
            mean_latency_ms=round(float(np.mean([r.latency_ms for r in reports])), 2),
            mean_hit_rate=round(float(np.mean([r.retrieval_hit_rate for r in reports])), 4),
            mean_recall=round(float(np.mean([r.retrieval_recall for r in reports])), 4),
            mean_mrr=round(float(np.mean([r.retrieval_mrr for r in reports])), 4),
            mean_token_f1=round(float(np.mean(valid_f1)), 4) if valid_f1 else 0.0,
            mean_semantic_similarity=round(float(np.mean(valid_sim)), 4) if valid_sim else 0.0,
            mean_faithfulness=round(float(np.mean([r.faithfulness_score for r in reports])), 4),
            fallback_rate=round(float(sum(1 for r in reports if r.is_fallback) / total), 4),
            itemized_results=reports,
        )

        return summary