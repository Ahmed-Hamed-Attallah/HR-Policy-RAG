"""Administrative router for index maintenance, rebuilding, and diagnostics."""

from fastapi import APIRouter, Depends
from app.backend.dependencies import get_index_manager
from app.backend.schemas.common import APIResponse
from src.config.settings import settings
from src.domain.documents import Document
from src.utils.file_utils import read_json
from src.vector_store.index_manager import IndexManager

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/rebuild-index", response_model=APIResponse[dict])
def rebuild_index(index_mgr: IndexManager = Depends(get_index_manager)):
    """Rebuild the vector store index from processed documents on disk."""
    index_mgr.clear_index()

    # Load all processed documents
    processed_dir = settings.DATA_PROCESSED_DIR
    documents = []
    for json_file in processed_dir.glob("*.json"):
        payload = read_json(json_file)
        documents.append(Document(**payload))

    indexed_count = index_mgr.index_documents(documents)
    return APIResponse(
        success=True,
        message=f"Reindexed {len(documents)} document(s) into {indexed_count} vector chunks.",
        data={"documents_processed": len(documents), "total_chunks": indexed_count},
    )


@router.post("/clear-index", response_model=APIResponse[dict])
def clear_index(index_mgr: IndexManager = Depends(get_index_manager)):
    """Clear all vector index data from memory and disk."""
    index_mgr.clear_index()
    return APIResponse(
        success=True,
        message="Vector store successfully cleared.",
        data={"total_vectors": 0},
    )