"""API tests for the RAG chat query endpoint."""

from fastapi.testclient import TestClient
from app.backend.main import app
from app.backend.dependencies import get_chat_service
from app.backend.schemas.chat import ChatQueryResponse, CitationResponse
from unittest.mock import MagicMock

client = TestClient(app)


def test_submit_chat_query_success():
    mock_chat_service = MagicMock()
    mock_chat_service.process_query.return_value = ChatQueryResponse(
        query="What is probation period?",
        answer="Probation is 90 days [Policy.pdf, Page 1].",
        citations=[
            CitationResponse(
                source_name="Policy.pdf",
                page_number=1,
                snippet="Probation is 90 days.",
                score=0.95,
            )
        ],
        retrieved_passages=[],
        model_name="test-model",
        is_fallback=False,
        latency_ms=45.2,
    )
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    payload = {"query": "What is probation period?", "top_k": 3}
    response = client.post("/api/v1/chat/query", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert "90 days" in res_data["data"]["answer"]
    assert len(res_data["data"]["citations"]) == 1
    assert res_data["data"]["citations"][0]["source_name"] == "Policy.pdf"


def test_submit_chat_query_invalid_body():
    # Empty query violates min_length=2
    response = client.post("/api/v1/chat/query", json={"query": ""})
    assert response.status_code == 422