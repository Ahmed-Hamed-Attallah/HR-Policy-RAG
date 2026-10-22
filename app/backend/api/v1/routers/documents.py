"""Router exposing document management and ingestion endpoints."""

from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, status
from app.backend.dependencies import get_document_service
from app.backend.schemas.common import APIResponse
from app.backend.schemas.documents import (
    DocumentDetailResponse,
    DocumentSummaryResponse,
    DocumentUploadResponse,
)
from app.backend.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", response_model=APIResponse[List[DocumentSummaryResponse]])
def list_documents(service: DocumentService = Depends(get_document_service)):
    """List all registered and indexed HR policy documents."""
    docs = service.list_documents()
    return APIResponse(data=docs, message=f"Retrieved {len(docs)} documents.")


@router.get("/{doc_id}", response_model=APIResponse[DocumentDetailResponse])
def get_document_by_id(doc_id: str, service: DocumentService = Depends(get_document_service)):
    """Fetch metadata and extracted text preview for a document."""
    doc = service.get_document(doc_id)
    return APIResponse(data=doc)


@router.post(
    "/upload",
    response_model=APIResponse[DocumentUploadResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    """Upload a new policy document (.pdf, .docx), parse, extract metadata, and index in FAISS."""
    result = await service.upload_document(file)
    return APIResponse(data=result, message=result.message)