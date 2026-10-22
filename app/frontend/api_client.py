"""HTTP API Client connecting Streamlit UI to FastAPI endpoints."""

import logging
from typing import Any, Dict, List, Optional
import requests
from app.frontend.config import frontend_settings

logger = logging.getLogger(__name__)


class APIClient:
    """Manages all REST API interactions with the backend service."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or frontend_settings.BACKEND_API_URL
        self.timeout = frontend_settings.REQUEST_TIMEOUT_SECONDS

    def check_health(self) -> Dict[str, Any]:
        """Query the backend /health probe."""
        try:
            resp = requests.get(frontend_settings.health_url, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("data", {})
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def submit_chat_query(
        self,
        query: str,
        top_k: int = 5,
        rerank_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a natural language question to the RAG pipeline."""
        payload = {
            "query": query,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "filters": filters,
        }
        try:
            resp = requests.post(
                frontend_settings.chat_query_url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("data", {})
        except requests.exceptions.HTTPError as http_err:
            try:
                err_json = resp.json()
                msg = err_json.get("error", {}).get("message", str(http_err))
            except Exception:
                msg = str(http_err)
            raise RuntimeError(f"Backend API error: {msg}")
        except Exception as err:
            raise RuntimeError(f"Network error: {err}")

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieve list of all ingested documents."""
        try:
            resp = requests.get(frontend_settings.documents_url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as err:
            logger.error(f"Failed to list documents: {err}")
            return []

    def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """Fetch metadata and text preview of a specific document."""
        url = f"{frontend_settings.documents_url}/{doc_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {})

    def upload_document(self, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        """Upload a PDF/DOCX file for instant indexing."""
        files = {"file": (file_name, file_bytes)}
        resp = requests.post(
            frontend_settings.document_upload_url,
            files=files,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    def rebuild_index(self) -> Dict[str, Any]:
        """Trigger FAISS re-indexing of all processed files."""
        resp = requests.post(frontend_settings.rebuild_index_url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def clear_index(self) -> Dict[str, Any]:
        """Purge current vector store."""
        resp = requests.post(frontend_settings.clear_index_url, timeout=10)
        resp.raise_for_status()
        return resp.json()


api_client = APIClient()