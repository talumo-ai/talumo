"""Tests for the FastAPI application (endpoint integration)."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from talumo.app import app
from talumo.settings import settings


def _completion(content: str, model: str = "test") -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


GOOD = "This is a perfectly good and long enough draft response."
CODE_GOOD = "```csharp\nConsole.WriteLine(\"FizzBuzz\");\n```"


class TestHealthEndpoint:
    def test_health(self) -> None:
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestOrchestrateEndpoint:
    def test_valid_request_returns_200(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{settings.litellm_base_url}/chat/completions",
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

    def test_local_model_hint_general_is_forwarded(self, httpx_mock: HTTPXMock) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["model"] == settings.litellm_local_model
            return httpx.Response(200, json=_completion(CODE_GOOD, model="local-general"))

        httpx_mock.add_callback(
            handler,
            url=f"{settings.litellm_base_url}/chat/completions",
        )
        body = {
            "task_type": "code",
            "local_model_hint": "general",
            "messages": [{"role": "user", "content": "Implement fizzbuzz"}],
        }
        with TestClient(app) as client:
            resp = client.post("/v1/orchestrate", json=body)
        assert resp.status_code == 200
        assert resp.json()["tier_used"] == "local"
