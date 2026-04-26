"""Model backend: thin wrapper around OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

import httpx

from talumo.schemas import Message, Tier
from talumo.settings import settings


class RemoteAPIKeyMissingError(RuntimeError):
    """Raised when a remote tier is requested without OPENAI_API_KEY configured."""


def _base_url_for(tier: Tier) -> str:
    return settings.litellm_base_url


def _model_for(tier: Tier) -> str:
    if tier == Tier.LOCAL:
        return settings.litellm_local_model
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
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelResult:
    """Send a chat-completion request to the tier's backend and return the result."""
    if tier != Tier.LOCAL and not settings.openai_api_key:
        raise RemoteAPIKeyMissingError(
            "OPENAI_API_KEY is not set; remote tiers require a provider API key"
        )

    payload: dict[str, object] = {
        "model": _model_for(tier),
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
        model=body.get("model", _model_for(tier)),
    )
