"""Tests for the FastAPI application (endpoint integration)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from zora.app import app
from zora.settings import settings


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


GOOD = "This is a perfectly good and long enough draft response."


class TestHealthEndpoint:
    def test_health(self) -> None:
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestOrchestrateEndpoint:
    def test_valid_request_returns_200(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{settings.local_base_url}/chat/completions",
            json=_completion(GOOD),
        )
        body = {
            "task_type": "draft",
            "messages": [{"role": "user", "content": "Say hello"}],
        }
        with TestClient(app) as client:
            resp = client.post("/v1/orchestrate", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert "trace" in data
        assert data["tier_used"] == "local"

    def test_invalid_task_type_returns_422(self) -> None:
        body = {
            "task_type": "nonexistent",
            "messages": [{"role": "user", "content": "test"}],
        }
        with TestClient(app) as client:
            resp = client.post("/v1/orchestrate", json=body)
        assert resp.status_code == 422

    def test_missing_messages_returns_422(self) -> None:
        body = {"task_type": "draft"}
        with TestClient(app) as client:
            resp = client.post("/v1/orchestrate", json=body)
        assert resp.status_code == 422
