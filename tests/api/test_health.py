"""API tests for the /api/v1/health probe."""

from fastapi.testclient import TestClient
from app.backend.main import app
from app.backend.dependencies import get_index_manager
from unittest.mock import MagicMock

client = TestClient(app)


def test_health_endpoint():
    mock_idx = MagicMock()
    mock_idx.get_stats.return_value = {
        "total_vectors": 42,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "is_persisted": True,
    }
    app.dependency_overrides[get_index_manager] = lambda: mock_idx

    response = client.get("/api/v1/health")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "healthy"
    assert payload["data"]["vectors_indexed"] == 42
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers