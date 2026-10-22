#!/usr/bin/env python3
"""CLI utility to inspect FAISS index artifacts, metadata consistency, and test queries."""

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.retrieval.retriever import PolicyRetriever
from src.vector_store.index_manager import IndexManager

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect existing FAISS index artifacts, metadata mapping, and probe retrieval."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=settings.VECTOR_INDEX_PATH,
        help="Path to FAISS index binary.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=settings.METADATA_STORE_PATH,
        help="Path to metadata store JSON file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional search query to probe index retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest neighbors to retrieve when querying.",
    )
    parser.add_argument(
        "--show-chunks",
        type=int,
        default=2,
        help="Number of stored chunks to display as sample previews.",
    )
    return parser.parse_args()


def display_index_health(index_manager: IndexManager, index_path: Path, metadata_path: Path) -> None:
    """Print index properties, size metrics, and metadata alignment."""
    stats = index_manager.get_stats()
    idx_size_kb = index_path.stat().st_size / 1024 if index_path.exists() else 0.0
    meta_size_kb = metadata_path.stat().st_size / 1024 if metadata_path.exists() else 0.0

    table = Table(title="FAISS Artifact Inspection")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("FAISS Index File", f"{index_path} ({idx_size_kb:.2f} KB)")
    table.add_row("Metadata File", f"{metadata_path} ({meta_size_kb:.2f} KB)")
    table.add_row("Indexed Vectors Count", str(stats["total_vectors"]))
    table.add_row("Vector Dimension", str(stats["vector_dimension"]))
    table.add_row("Underlying Model", str(stats["embedding_model"]))
    table.add_row("Metadata Payloads Count", str(len(index_manager.vector_store.metadata_store)))

    console.print(table)

    # Sanity verification
    vec_count = stats["total_vectors"]
    meta_count = len(index_manager.vector_store.metadata_store)
    if vec_count != meta_count:
        console.print(
            f"[bold red]WARNING: Vector count ({vec_count}) and metadata entries ({meta_count}) do not match![/bold red]"
        )
    else:
        console.print("[bold green]Integrity Check: FAISS vector space and metadata mappings are in 1:1 sync.[/bold green]\n")


def display_sample_chunks(index_manager: IndexManager, count: int) -> None:
    """Preview sample stored chunks from the metadata repository."""
    metadata_store = index_manager.vector_store.metadata_store
    if not metadata_store:
        console.print("[yellow]Metadata store is empty. No chunks to display.[/yellow]")
        return

    console.print(f"[bold cyan]Showing {min(count, len(metadata_store))} Sample Chunks:[/bold cyan]\n")
    sample_keys = list(metadata_store.keys())[:count]

    for key in sample_keys:
        payload = metadata_store[key]
        header = (
            f"Index Slot: {key} | Document: {payload.get('document_name')} | "
            f"Page: {payload.get('page_number')} | Chunk ID: {payload.get('chunk_id')}"
        )
        content_preview = payload.get("content", "")[:280] + ("..." if len(payload.get("content", "")) > 280 else "")
        console.print(Panel(content_preview, title=header, subtitle=f"Chars: {len(payload.get('content', ''))}"))


def probe_query(index_manager: IndexManager, query: str, top_k: int) -> None:
    """Execute vector retrieval probe and render matching chunks."""
    console.print(f"\n[bold magenta]Probing Retrieval with Query:[/bold magenta] '{query}' (top_k={top_k})")
    retriever = PolicyRetriever(index_manager=index_manager)
    results = retriever.retrieve(query=query, top_k=top_k)

    if not results:
        console.print("[red]No matching chunks retrieved.[/red]")
        return

    table = Table(title=f"Retrieval Hits for: '{query}'")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Source Document", style="magenta")
    table.add_column("Page", justify="center")
    table.add_column("Excerpt Preview", style="white")

    for rank, res in enumerate(results, start=1):
        chunk = res.chunk
        score_val = f"{res.similarity_score:.4f}"
        preview = chunk.content[:160].replace("\n", " ") + "..."
        page_val = str(chunk.metadata.page_number) if chunk.metadata.page_number is not None else "-"
        table.add_row(str(rank), score_val, chunk.metadata.document_name, page_val, preview)

    console.print(table)


def main() -> None:
    args = parse_args()
    index_path = args.index_path.resolve()
    metadata_path = args.metadata_path.resolve()

    if not index_path.exists() or not metadata_path.exists():
        console.print(
            f"[bold red]Error: Index artifacts missing at:\n  - {index_path}\n  - {metadata_path}[/bold red]\n"
            "Run [bold green]python scripts/build_index.py[/bold green] first."
        )
        sys.exit(1)

    index_manager = IndexManager(index_path=index_path, metadata_path=metadata_path)
    index_manager.load_or_create()

    # 1. Output health table
    display_index_health(index_manager, index_path, metadata_path)

    # 2. Preview sample chunks
    if args.show_chunks > 0:
        display_sample_chunks(index_manager, args.show_chunks)

    # 3. Probe search query if requested
    if args.query:
        probe_query(index_manager, args.query, args.top_k)


if __name__ == "__main__":
    main()