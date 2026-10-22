"""CLI script to run quantitative evaluation benchmarks over RAG pipeline."""

import argparse
import sys
from pathlib import Path

# Resolve project root dynamically
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table

from src.config.settings import settings
from src.evaluation.evaluation_runner import EvaluationRunner
from src.rag.pipeline import RAGPipeline
from src.utils.file_utils import ensure_directory, save_json

console = Console()


def ensure_sample_evaluation_data(dataset_path: Path) -> None:
    """Populate default benchmark questions if questions.jsonl is absent."""
    if dataset_path.exists() and dataset_path.stat().st_size > 0:
        return

    console.print(f"[yellow]Benchmark file not found. Creating starter benchmark at {dataset_path}...[/yellow]")
    ensure_directory(dataset_path.parent)

    sample_questions = [
        {
            "query_id": "BENCH-001",
            "query": "What is the standard probation period for new employees?",
            "expected_answer": "The standard probationary period is typically three to six months depending on company policy.",
            "expected_doc_ids": [],
            "expected_keywords": ["probation", "months", "employment"],
            "category": "probation",
        },
        {
            "query_id": "BENCH-002",
            "query": "How many days of paid annual leave are employees entitled to?",
            "expected_answer": "Employees are entitled to annual vacation days specified in their contract and HR policy.",
            "expected_doc_ids": [],
            "expected_keywords": ["annual leave", "vacation", "days"],
            "category": "leave",
        },
        {
            "query_id": "BENCH-003",
            "query": "Can an employee take advance annual leave during probation?",
            "expected_answer": "Advance leave during probation usually requires written approval from HR or senior management.",
            "expected_doc_ids": [],
            "expected_keywords": ["advance leave", "probation", "approval"],
            "category": "exceptions",
        },
        {
            "query_id": "BENCH-004",
            "query": "What is the policy for bringing pets into the company office?",
            "expected_answer": "The policy documentation does not authorize pets in the office.",
            "expected_doc_ids": [],
            "expected_keywords": ["pet", "dog", "office"],
            "category": "out_of_domain",
        },
    ]

    import json
    with open(dataset_path, "w", encoding="utf-8") as f:
        for q in sample_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Enterprise HR Policy RAG Pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl",
        help="Path to questions.jsonl evaluation dataset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top-k cutoff limit (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.ARTIFACTS_DIR / "evaluation_report.json",
        help="Destination path for output JSON report",
    )
    args = parser.parse_args()

    ensure_sample_evaluation_data(args.dataset)

    console.print("[bold cyan]Initializing Enterprise RAG Pipeline for Evaluation...[/bold cyan]")
    pipeline = RAGPipeline()
    runner = EvaluationRunner(rag_pipeline=pipeline, retrieval_k=args.top_k)

    console.print(f"[bold green]Executing evaluation on {args.dataset}...[/bold green]\n")
    report = runner.run_benchmark(args.dataset)

    # 1. Summary Metrics Display Table
    summary_table = Table(title="RAG Evaluation Benchmark: Global Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Score", style="bold green", justify="center")

    summary_table.add_row("Total Benchmark Queries", str(report.total_queries))
    summary_table.add_row("Mean Latency (ms)", f"{report.mean_latency_ms:.2f} ms")
    summary_table.add_row(f"Retrieval Hit Rate @ {args.top_k}", f"{report.mean_hit_rate * 100:.1f}%")
    summary_table.add_row(f"Retrieval MRR @ {args.top_k}", f"{report.mean_mrr:.3f}")
    summary_table.add_row("Answer Token F1", f"{report.mean_token_f1 * 100:.1f}%")
    summary_table.add_row("Answer Semantic Similarity", f"{report.mean_semantic_similarity * 100:.1f}%")
    summary_table.add_row("Faithfulness (Groundedness)", f"{report.mean_faithfulness * 100:.1f}%")
    summary_table.add_row("Safe Fallback Rate", f"{report.fallback_rate * 100:.1f}%")

    console.print(summary_table)

    # 2. Per-Query Itemized Breakdown
    items_table = Table(title="Itemized Benchmark Diagnostics")
    items_table.add_column("ID", style="cyan")
    items_table.add_column("Query Preview", style="white")
    items_table.add_column("Latency", justify="center")
    items_table.add_column("F1", justify="center")
    items_table.add_column("SemSim", justify="center")
    items_table.add_column("Faithful?", justify="center")
    items_table.add_column("Fallback?", justify="center")

    for item in report.itemized_results:
        f_style = "[green]Yes[/green]" if item.is_faithful else "[red]No[/red]"
        fb_style = "[yellow]Yes[/yellow]" if item.is_fallback else "[white]No[/white]"
        items_table.add_row(
            item.query_id,
            item.query[:35] + "...",
            f"{item.latency_ms:.0f}ms",
            f"{item.token_f1:.2f}" if item.token_f1 is not None else "N/A",
            f"{item.semantic_similarity:.2f}" if item.semantic_similarity is not None else "N/A",
            f_style,
            fb_style,
        )

    console.print(items_table)

    # 3. Persist Full Report
    save_json(report.model_dump(), args.output)
    console.print(f"\n[bold green]✓ Full evaluation report saved to {args.output}[/bold green]")


if __name__ == "__main__":
    main()