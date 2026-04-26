"""Model backend: thin wrapper around OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

import httpx

from talumo.schemas import LocalModelHint, Message, OrchRequest, TaskType, Tier
from talumo.settings import settings


class RemoteAPIKeyMissingError(RuntimeError):
    """Raised when a remote tier is requested without OPENAI_API_KEY configured."""


def _base_url_for(tier: Tier) -> str:
    return settings.litellm_base_url


def _local_model_for(req: OrchRequest | None) -> str:
    if req is None:
        return settings.litellm_local_model

    if req.local_model_hint == LocalModelHint.GENERAL:
        return settings.litellm_local_model

    if req.local_model_hint == LocalModelHint.CODE:
        return settings.litellm_local_code_model

    if req.task_type == TaskType.CODE:
        return settings.litellm_local_code_model

    return settings.litellm_local_model


def _model_for(tier: Tier, req: OrchRequest | None = None) -> str:
    if tier == Tier.LOCAL:
        return _local_model_for(req)
    if tier == Tier.REMOTE_CHEAP:
        return settings.litellm_cheap_model
    return settings.litellm_frontier_model


def _headers_for(tier: Tier) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.litellm_api_key}",
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ModelResult:
    __slots__ = ("content", "finish_reason", "model")

    def __init__(self, content: str, finish_reason: str, model: str) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.model = model


async def call_model(
    tier: Tier,
    messages: list[Message],
    *,
    client: httpx.AsyncClient,
    req: OrchRequest | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelResult:
    """Send a chat-completion request to the tier's backend and return the result."""
    if tier != Tier.LOCAL and not settings.openai_api_key:
        raise RemoteAPIKeyMissingError(
            "OPENAI_API_KEY is not set; remote tiers require a provider API key"
        )

    model_name = _model_for(tier, req=req)
    payload: dict[str, object] = {
        "model": model_name,
        "messages": [m.model_dump() for m in messages],
        "max_tokens": max_tokens or settings.max_tokens,
        "temperature": temperature if temperature is not None else settings.temperature,
    }

    resp = await client.post(
        f"{_base_url_for(tier)}/chat/completions",
        json=payload,
        headers=_headers_for(tier),
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    choice = body["choices"][0]
    return ModelResult(
        content=choice["message"]["content"],
        finish_reason=choice.get("finish_reason", "stop"),
        model=body.get("model", model_name),
    )
