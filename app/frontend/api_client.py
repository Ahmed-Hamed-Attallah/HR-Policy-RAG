"""In-memory client running RAGPipeline directly for Streamlit Cloud deployment."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

# Guarantee root path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.pipeline import RAGPipeline
from src.vector_store.index_manager import IndexManager


@st.cache_resource
def get_cached_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource
def get_cached_index_manager() -> IndexManager:
    manager = IndexManager()
    manager.load_or_create()
    return manager


class APIClient:
    """Direct in-process client replacing REST HTTP requests."""

    def __init__(self, base_url: Optional[str] = None):
        pass

    def check_health(self) -> Dict[str, Any]:
        try:
            manager = get_cached_index_manager()
            stats = manager.get_stats()
            return {
                "status": "healthy",
                "vectors_indexed": stats["total_vectors"],
                "embedding_model": stats["embedding_model"],
                "index_persisted": stats["is_persisted"],
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def submit_chat_query(
        self,
        query: str,
        top_k: int = 5,
        rerank_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pipeline = get_cached_pipeline()
        response = pipeline.run(query=query, retrieval_k=top_k, rerank_k=rerank_k)

        citations_data = [
            {
                "source_name": c.source_name,
                "page_number": c.page_number,
                "snippet": c.snippet,
                "score": c.score,
            }
            for c in response.citations
        ]

        passages_data = [
            {
                "document_name": rc.chunk.metadata.document_name,
                "page_number": rc.chunk.metadata.page_number,
                "similarity_score": rc.similarity_score,
                "rerank_score": rc.rerank_score,
                "content": rc.chunk.content,
            }
            for rc in response.retrieved_chunks
        ]

        return {
            "query": response.query,
            "answer": response.answer,
            "citations": citations_data,
            "retrieved_passages": passages_data,
            "model_name": response.model_name,
            "is_fallback": response.is_fallback,
            "latency_ms": getattr(response, "latency_ms", 0.0),
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        try:
            manager = get_cached_index_manager()
            metadata = manager.vector_store.metadata_store
            seen = set()
            docs = []
            for item in metadata.values():
                doc_name = item.get("document_name")
                if doc_name and doc_name not in seen:
                    seen.add(doc_name)
                    docs.append({
                        "doc_id": item.get("doc_id", doc_name),
                        "file_name": doc_name,
                        "file_type": Path(doc_name).suffix.lstrip(".").lower() or "pdf",
                        "file_size_bytes": 0,
                        "total_pages": item.get("page_number", 1),
                    })
            return docs
        except Exception:
            return []

    def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        return {"content_preview": "Document registered in index."}

    def upload_document(self, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        raise NotImplementedError("Direct uploads disabled in cloud deployment mode.")

    def rebuild_index(self) -> Dict[str, Any]:
        raise NotImplementedError("Rebuilding disabled in cloud deployment mode.")

    def clear_index(self) -> Dict[str, Any]:
        raise NotImplementedError("Clearing disabled in cloud deployment mode.")


api_client = APIClient()