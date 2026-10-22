"""Health check and readiness inspection router."""

from fastapi import APIRouter, Depends
from app.backend.dependencies import get_index_manager
from app.backend.schemas.common import APIResponse
from src.vector_store.index_manager import IndexManager

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=APIResponse[dict])
def health_check(index_mgr: IndexManager = Depends(get_index_manager)):
    """Return system liveness status and vector store diagnostics."""
    stats = index_mgr.get_stats()
    return APIResponse(
        success=True,
        message="HR Policy RAG API is operational",
        data={
            "status": "healthy",
            "vectors_indexed": stats["total_vectors"],
            "embedding_model": stats["embedding_model"],
            "index_persisted": stats["is_persisted"],
        },
    )