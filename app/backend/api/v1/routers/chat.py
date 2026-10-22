"""Router handling RAG conversational Q&A queries."""

from fastapi import APIRouter, Depends, status
from app.backend.dependencies import get_chat_service
from app.backend.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.backend.schemas.common import APIResponse
from app.backend.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat & RAG"])


@router.post(
    "/query",
    response_model=APIResponse[ChatQueryResponse],
    status_code=status.HTTP_200_OK,
)
def submit_query(
    request: ChatQueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Query the HR policy RAG engine with an inquiry, metadata filters, and retrieval tuning."""
    result = chat_service.process_query(request)
    return APIResponse(
        success=True,
        message="Query processed successfully",
        data=result,
    )