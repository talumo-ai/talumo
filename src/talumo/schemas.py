"""Request/response schemas and shared types for the orchestrator."""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(StrEnum):
    DRAFT = "draft"
    SUMMARY = "summary"
    EXTRACT = "extract"
    CODE = "code"
    ANALYSIS = "analysis"
    FINAL = "final"


class Finality(StrEnum):
    DRAFT = "draft"
    FINAL = "final"


class Sensitivity(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"


class LocalModelHint(StrEnum):
    GENERAL = "general"
    CODE = "code"


class Tier(StrEnum):
    LOCAL = "local"
    REMOTE_CHEAP = "remote_cheap"
    REMOTE_FRONTIER = "remote_frontier"


class Verdict(StrEnum):
    PASS = "PASS"
    RETRY_SAME_TIER = "RETRY_SAME_TIER"
    ESCALATE = "ESCALATE"


# ---------------------------------------------------------------------------
# Chat message
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


# ---------------------------------------------------------------------------
# Request envelope
# ---------------------------------------------------------------------------

class RequestMetadata(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tags: list[str] = Field(default_factory=list)


class OrchRequest(BaseModel):
    task_type: TaskType
    finality: Finality = Finality.DRAFT
    sensitivity: Sensitivity = Sensitivity.PRIVATE
    local_model_hint: LocalModelHint | None = None
    needs_schema: bool = False
    can_test: bool = False
    messages: list[Message]
    response_schema: dict[str, Any] | None = None
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)


# ---------------------------------------------------------------------------
# Routing trace
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    tier: Tier
    model: str
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)
    latency_ms: float


class OrchResponse(BaseModel):
    request_id: str
    content: str
    finish_reason: str
    tier_used: Tier
    trace: list[TraceStep]
