"""Centralized exception classes and global FastAPI exception handlers."""

import logging
from typing import Any, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.backend.schemas.common import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base exception for application-level errors."""
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class DocumentNotFoundError(AppException):
    """Raised when an ingested policy document cannot be located."""
    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document with ID '{doc_id}' not found.",
            code="DOCUMENT_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class InvalidFileTypeError(AppException):
    """Raised when an unsupported file format is uploaded."""
    def __init__(self, extension: str):
        super().__init__(
            message=f"File extension '{extension}' is not supported. Supported: .pdf, .docx",
            code="UNSUPPORTED_FILE_TYPE",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class GuardrailViolationError(AppException):
    """Raised when prompt injection or malicious input is intercepted."""
    def __init__(self, reason: str):
        super().__init__(
            message=f"Request violated system safety policies: {reason}",
            code="GUARDRAIL_VIOLATION",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class VectorStoreError(AppException):
    """Raised when FAISS indexing or similarity retrieval fails."""
    def __init__(self, message: str):
        super().__init__(
            message=f"Vector store operation error: {message}",
            code="VECTOR_STORE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom AppException instances."""
    logger.warning(f"Handled application exception on {request.url.path}: {exc.code} - {exc.message}")
    payload = ErrorResponse(
        error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details)
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled exceptions."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    payload = ErrorResponse(
        error=ErrorDetail(
            code="UNHANDLED_EXCEPTION",
            message="An unexpected server error occurred. Please contact the administrator.",
            details=str(exc) if logger.isEnabledFor(logging.DEBUG) else None,
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(mode="json"),
    )