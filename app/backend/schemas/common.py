"""Common reusable response schemas for FastAPI endpoints."""

from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API response envelope."""
    success: bool = Field(default=True, description="Indicates request success status")
    message: str = Field(default="Operation completed successfully", description="Status message")
    data: Optional[T] = Field(default=None, description="Response payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp")


class ErrorDetail(BaseModel):
    """Detailed error structure returned in exception payloads."""
    code: str = Field(description="Machine-readable error identifier")
    message: str = Field(description="Human-readable error description")
    details: Optional[Any] = Field(default=None, description="Supplemental debug or field details")


class ErrorResponse(BaseModel):
    """Unified error response model."""
    success: bool = Field(default=False)
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Query parameters for paginated listings."""
    page: int = Field(default=1, ge=1, description="Page number starting at 1")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Wrapper envelope for paginated collections."""
    items: List[T] = Field(default_factory=list)
    total_count: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=0)