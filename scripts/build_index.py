#!/usr/bin/env python3
"""CLI script to chunk documents, generate embeddings, and build the FAISS vector index."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.chunking.text_splitter import ContextualTextSplitter
from src.config.model_config import ChunkingConfig, chunking_config
from src.config.settings import settings
from src.domain.documents import Document
from src.embeddings.embedding_model import EmbeddingModel
from src.ingestion.loader_factory import LoaderFactory
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.metadata_extractor import MetadataExtractor
from src.utils.file_utils import ensure_directory, list_files_by_extension
from src.vector_store.index_manager import IndexManager

console = Console()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk HR documents, generate embeddings, and build FAISS vector index."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=settings.DATA_RAW_DIR,
        help="Directory containing original raw policy documents.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=settings.DATA_PROCESSED_DIR,
        help="Directory containing processed document JSON files (optional shortcut).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=settings.VECTOR_INDEX_PATH,
        help="Output path for the serialized FAISS index binary.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=settings.METADATA_STORE_PATH,
        help="Output path for the serialized chunk metadata JSON.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=chunking_config.chunk_size,
        help="Chunk size limit in characters/tokens.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=chunking_config.chunk_overlap,
        help="Overlap size in characters/tokens between adjacent chunks.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Purge existing index and rebuild from scratch.",
    )
    return parser.parse_args()


def load_documents_from_processed(processed_dir: Path) -> List[Document]:
    """Load pre-cleaned Document objects from JSON files."""
    json_files = list(processed_dir.glob("*.json"))
    docs = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            docs.append(Document.model_validate(data))
        except Exception as err:
            logger.warning(f"Could not load processed doc from {jf.name}: {err}")
    return docs


def load_documents_from_raw(source_dir: Path) -> List[Document]:
    """Load, clean, and enrich documents directly from raw files."""
    files = list_files_by_extension(source_dir, ["pdf", "docx"])
    docs = []
    for fp in files:
        try:
            raw_doc = LoaderFactory.load_document(fp)
            cleaned = DocumentCleaner.clean_document(raw_doc)
            enriched = MetadataExtractor.enrich_document_metadata(cleaned)
            docs.append(enriched)
        except Exception as err:
            logger.error(f"Failed to load raw document {fp.name}: {err}")
    return docs


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    index_path = args.index_path.resolve()
    metadata_path = args.metadata_path.resolve()

    ensure_directory(index_path.parent)
    ensure_directory(metadata_path.parent)

    logger.info("Starting index construction workflow...")
    start_time = time.perf_counter()

    # 1. Discover documents
    documents: List[Document] = []
    if processed_dir.exists() and list(processed_dir.glob("*.json")):
        logger.info(f"Loading pre-processed documents from {processed_dir}")
        documents = load_documents_from_processed(processed_dir)

    if not documents:
        logger.info(f"Loading and processing documents directly from {source_dir}")
        documents = load_documents_from_raw(source_dir)

    if not documents:
        logger.error("No documents found to index. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} document(s) for indexing.")

    # 2. Build ChunkingConfig from CLI parameters
    custom_config = ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=chunking_config.separators,
    )
    custom_splitter = ContextualTextSplitter(config=custom_config)

    # 3. Setup IndexManager
    embedding_model = EmbeddingModel.get_instance()
    index_manager = IndexManager(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )
    index_manager.text_splitter = custom_splitter

    if args.rebuild:
        logger.info("Flag --rebuild provided. Clearing existing index artifacts...")
        index_manager.clear_index()
    else:
        index_manager.load_or_create()

    # 4. Execute indexing
    indexed_chunk_count = index_manager.index_documents(documents)
    elapsed = time.perf_counter() - start_time

    # 5. Summary presentation
    table = Table(title="FAISS Index Build Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source Documents", str(len(documents)))
    table.add_row("Total Chunks Created", str(indexed_chunk_count))
    table.add_row("Total Vectors in Index", str(index_manager.vector_store.total_vectors()))
    table.add_row("Embedding Dimension", str(embedding_model.dimension))
    table.add_row("Chunk Size", str(args.chunk_size))
    table.add_row("Chunk Overlap", str(args.chunk_overlap))
    table.add_row("FAISS Index File", str(index_path.name))
    table.add_row("Metadata Store File", str(metadata_path.name))
    table.add_row("Elapsed Time", f"{elapsed:.2f} s")

    console.print(table)
    logger.info("Index build and persistence completed successfully.")


if __name__ == "__main__":
    main()