"""FastAPI dependencies providing singletons and service instances to routers."""

from functools import lru_cache
from app.backend.services.chat_service import ChatService
from app.backend.services.document_service import DocumentService
from src.ingestion.document_registry import DocumentRegistry
from src.rag.pipeline import RAGPipeline
from src.vector_store.index_manager import IndexManager


@lru_cache()
def get_index_manager() -> IndexManager:
    """Retrieve or initialize the singleton IndexManager with cached FAISS vectors."""
    manager = IndexManager()
    manager.load_or_create()
    return manager


@lru_cache()
def get_document_registry() -> DocumentRegistry:
    """Provide DocumentRegistry instance."""
    return DocumentRegistry()


@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    """Provide initialized RAGPipeline instance."""
    return RAGPipeline()


def get_document_service() -> DocumentService:
    """Instantiate DocumentService injecting active registry and index manager."""
    return DocumentService(
        registry=get_document_registry(),
        index_manager=get_index_manager(),
    )


def get_chat_service() -> ChatService:
    """Instantiate ChatService injecting pipeline singleton."""
    return ChatService(rag_pipeline=get_rag_pipeline())