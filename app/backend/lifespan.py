"""FastAPI Lifespan management handling resource initialization and graceful shutdown."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.backend.dependencies import get_index_manager
from src.config.settings import settings
from src.utils.file_utils import ensure_directory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Execute startup checks, initialize indices, and manage cleanup on termination."""
    logger.info("Initializing Enterprise HR Policy RAG API Service...")

    # 1. Ensure required storage directories exist
    ensure_directory(settings.DATA_RAW_DIR)
    ensure_directory(settings.DATA_PROCESSED_DIR)
    ensure_directory(settings.ARTIFACTS_DIR)
    ensure_directory(settings.LOGS_DIR)

    # 2. Warm up FAISS index & embeddings
    try:
        index_manager = get_index_manager()
        stats = index_manager.get_stats()
        logger.info(
            f"Vector index ready. Indexed vectors: {stats['total_vectors']}, "
            f"Embedding model: {stats['embedding_model']}"
        )
    except Exception as e:
        logger.error(f"Failed to load vector index on startup: {e}")

    yield

    logger.info("Shutting down HR Policy RAG API Service...")