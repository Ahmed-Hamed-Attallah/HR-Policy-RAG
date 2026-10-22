"""Frontend configuration and API endpoints routing."""

import os
from pydantic_settings import BaseSettings


class FrontendSettings(BaseSettings):
    """Configuration settings for Streamlit application."""
    BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")
    APP_TITLE: str = "Enterprise HR Policy Assistant"
    APP_ICON: str = "🏢"
    REQUEST_TIMEOUT_SECONDS: int = 60

    @property
    def health_url(self) -> str:
        return f"{self.BACKEND_API_URL}/health"

    @property
    def chat_query_url(self) -> str:
        return f"{self.BACKEND_API_URL}/chat/query"

    @property
    def documents_url(self) -> str:
        return f"{self.BACKEND_API_URL}/documents"

    @property
    def document_upload_url(self) -> str:
        return f"{self.BACKEND_API_URL}/documents/upload"

    @property
    def rebuild_index_url(self) -> str:
        return f"{self.BACKEND_API_URL}/admin/rebuild-index"

    @property
    def clear_index_url(self) -> str:
        return f"{self.BACKEND_API_URL}/admin/clear-index"


frontend_settings = FrontendSettings()