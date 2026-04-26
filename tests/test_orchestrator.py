"""Tests for the orchestration loop — proves the four MVP claims.

Uses pytest-httpx to mock backend HTTP calls so no real servers are needed.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from talumo.orchestrator import orchestrate
from talumo.schemas import (
    Finality,
    LocalModelHint,
    Message,
    OrchRequest,
    TaskType,
    Tier,
    Verdict,
)
from talumo.settings import settings


@pytest.fixture(autouse=True)
def _set_openai_key_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep historical test behavior unless a test overrides the key explicitly."""
    monkeypatch.setattr(settings, "openai_api_key", "test-openai-key")


def _req(**kw: object) -> OrchRequest:
    defaults: dict[str, object] = {
        "task_type": TaskType.DRAFT,
        "messages": [Message(role="user", content="Summarize this.")],
    }
    defaults.update(kw)
    return OrchRequest(**defaults)  # type: ignore[arg-type]


def _completion(content: str, model: str = "test-model") -> dict[str, object]:
    """Build a minimal OpenAI-compatible chat completion response body."""
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


GOOD_OUTPUT = "Here is a thorough draft response with sufficient content for the validators."
CODE_OUTPUT = "```csharp\nConsole.WriteLine(\"FizzBuzz\");\n```"


# ---- Claim 1: local-first routing works ----

@pytest.mark.asyncio
async def test_local_first_routing(httpx_mock: HTTPXMock) -> None:
    """A private draft request should route to the local backend and return PASS."""
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(GOOD_OUTPUT, model="local-model"),
    )
    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(), client)

    assert resp.tier_used == Tier.LOCAL
    assert resp.content == GOOD_OUTPUT
    assert len(resp.trace) == 1
    assert resp.trace[0].verdict == Verdict.PASS


# ---- Claim 2: validation catches obvious failures ----

@pytest.mark.asyncio
async def test_validation_catches_short_output(httpx_mock: HTTPXMock) -> None:
    """A short response triggers RETRY, then a good response should PASS."""
    # First call → too short
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion("no"),
    )
    # Retry → good
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(GOOD_OUTPUT),
    )
    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(), client)

    assert resp.tier_used == Tier.LOCAL
    assert len(resp.trace) == 2
    assert resp.trace[0].verdict == Verdict.RETRY_SAME_TIER
    assert resp.trace[1].verdict == Verdict.PASS


@pytest.mark.asyncio
async def test_validation_catches_refusal(httpx_mock: HTTPXMock) -> None:
    """A refusal triggers RETRY, then escalation on second failure."""
    refusal = "I'm sorry, I cannot help with that request."
    # First call → refusal
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(refusal),
    )
    # Retry → still a refusal
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(refusal),
    )
    # Escalation to cheap → good
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(GOOD_OUTPUT),
    )
    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(), client)

    assert resp.tier_used == Tier.REMOTE_CHEAP
    assert resp.trace[0].verdict == Verdict.RETRY_SAME_TIER
    assert resp.trace[1].verdict == Verdict.ESCALATE
    assert resp.trace[2].verdict == Verdict.PASS


# ---- Claim 3: escalation to remote works ----

@pytest.mark.asyncio
async def test_escalation_on_backend_error(httpx_mock: HTTPXMock) -> None:
    """If the local backend errors twice, we escalate to remote_cheap."""
    # Local → error
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        status_code=500,
    )
    # Local retry → error again
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        status_code=500,
    )
    # Escalate to cheap → success
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(GOOD_OUTPUT),
    )
    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(), client)

    assert resp.tier_used == Tier.REMOTE_CHEAP
    assert resp.trace[-1].verdict == Verdict.PASS


# ---- Claim 4: one endpoint hides backend complexity ----

@pytest.mark.asyncio
async def test_response_always_has_trace(httpx_mock: HTTPXMock) -> None:
    """Every response includes a request_id, content, and non-empty trace."""
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(GOOD_OUTPUT),
    )
    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(), client)

    assert resp.request_id
    assert resp.content
    assert resp.trace
    assert resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_final_goes_to_frontier(httpx_mock: HTTPXMock) -> None:
    """A request with finality=final should start at frontier tier."""
    httpx_mock.add_response(
        url=f"{settings.litellm_base_url}/chat/completions",
        json=_completion(GOOD_OUTPUT, model="frontier-model"),
    )
    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(finality=Finality.FINAL), client)

    assert resp.tier_used == Tier.REMOTE_FRONTIER
    assert resp.trace[0].tier == Tier.REMOTE_FRONTIER


@pytest.mark.asyncio
async def test_local_code_task_uses_local_code_alias(httpx_mock: HTTPXMock) -> None:
    """Code requests on local tier should use the local code alias."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == settings.litellm_local_code_model
        return httpx.Response(200, json=_completion(CODE_OUTPUT, model="local-code"))

    httpx_mock.add_callback(
        handler,
        url=f"{settings.litellm_base_url}/chat/completions",
    )

    async with httpx.AsyncClient() as client:
        resp = await orchestrate(_req(task_type=TaskType.CODE), client)

    assert resp.tier_used == Tier.LOCAL
    assert resp.trace[0].model == "local-code"


@pytest.mark.asyncio
async def test_local_model_hint_general_overrides_code_default(httpx_mock: HTTPXMock) -> None:
    """An explicit local_model_hint should override default code-model routing."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == settings.litellm_local_model
        return httpx.Response(200, json=_completion(CODE_OUTPUT, model="local-general"))

    httpx_mock.add_callback(
        handler,
        url=f"{settings.litellm_base_url}/chat/completions",
    )

    async with httpx.AsyncClient() as client:
        resp = await orchestrate(
            _req(task_type=TaskType.CODE, local_model_hint=LocalModelHint.GENERAL),
            client,
        )

    assert resp.tier_used == Tier.LOCAL
    assert resp.trace[0].model == "local-general"


@pytest.mark.asyncio
async def test_missing_openai_key_fails_only_when_remote_is_used() -> None:
    """Missing OPENAI_API_KEY should fail on remote-tier attempts without HTTP retries."""
    settings.openai_api_key = ""
    req = _req(task_type=TaskType.ANALYSIS, can_test=True)

    async with httpx.AsyncClient() as client:
        resp = await orchestrate(req, client)

    assert resp.tier_used == Tier.REMOTE_CHEAP
    assert resp.content == ""
    assert resp.finish_reason == "error"
    assert len(resp.trace) == 1
    assert resp.trace[0].tier == Tier.REMOTE_CHEAP
    assert "OPENAI_API_KEY is not set" in resp.trace[0].reasons[0]
