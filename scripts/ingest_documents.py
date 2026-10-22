#!/usr/bin/env python3
"""CLI script to ingest, clean, and record HR policy documents."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.config.settings import settings
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.loader_factory import LoaderFactory
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.metadata_extractor import MetadataExtractor
from src.utils.file_utils import ensure_directory, list_files_by_extension

console = Console()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("ingest_documents")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest HR documents, sanitize text, and update document registry."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=settings.DATA_RAW_DIR,
        help="Directory containing source PDF and DOCX files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.DATA_PROCESSED_DIR,
        help="Directory where cleaned document JSONs are saved.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=settings.ARTIFACTS_DIR / "document_registry.json",
        help="Path to the document registry JSON file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process files even if they already exist in the registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir: Path = args.source_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    registry_path: Path = args.registry_path.resolve()

    ensure_directory(source_dir)
    ensure_directory(output_dir)
    ensure_directory(registry_path.parent)

    logger.info(f"Scanning directory: {source_dir}")
    discovered_files = list_files_by_extension(source_dir, ["pdf", "docx"])

    if not discovered_files:
        logger.warning(f"No candidate PDF or DOCX files found in {source_dir}.")
        sys.exit(0)

    registry = DocumentRegistry(registry_path)
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    table = Table(title="Document Ingestion Summary")
    table.add_column("File Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Pages", justify="right", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for file_path in discovered_files:
        rel_name = file_path.name
        file_ext = file_path.suffix.lower().lstrip(".")

        if not args.force and registry.is_document_registered(file_path):
            skipped_count += 1
            table.add_row(rel_name, file_ext, "-", "[yellow]SKIPPED[/yellow]", "Unchanged in registry")
            continue

        try:
            # 1. Load document
            raw_doc = LoaderFactory.load_document(file_path)

            # 2. Clean text & normalize whitespace
            cleaned_doc = DocumentCleaner.clean_document(raw_doc)

            # 3. Enrich metadata (title, dates, department)
            enriched_doc = MetadataExtractor.enrich_document_metadata(cleaned_doc)

            # 4. Save processed document JSON
            out_file = output_dir / f"{enriched_doc.doc_id}.json"
            out_file.write_text(
                json.dumps(enriched_doc.model_dump(), default=str, indent=2),
                encoding="utf-8",
            )

            # 5. Register document
            registry.register(enriched_doc)

            processed_count += 1
            page_count = str(len(enriched_doc.pages))
            table.add_row(rel_name, file_ext, page_count, "[green]PROCESSED[/green]", str(out_file.name))
        except Exception as exc:
            failed_count += 1
            table.add_row(rel_name, file_ext, "0", "[red]FAILED[/red]", str(exc))
            logger.error(f"Error processing {rel_name}: {exc}", exc_info=True)

    console.print(table)
    logger.info(
        f"Ingestion complete: {processed_count} processed, "
        f"{skipped_count} skipped, {failed_count} failed."
    )


if __name__ == "__main__":
    main()