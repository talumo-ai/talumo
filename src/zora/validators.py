"""Output validators.

Each validator is a callable:
    (content: str, request: OrchRequest) -> list[str]

It returns an empty list on success, or a list of human-readable failure reasons.
The orchestrator aggregates reasons to decide PASS / RETRY / ESCALATE.
"""

from __future__ import annotations

import json
from typing import Protocol

from zora.schemas import OrchRequest, TaskType
from zora.settings import settings


class Validator(Protocol):
    def __call__(self, content: str, request: OrchRequest) -> list[str]: ...


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------


def check_length(content: str, request: OrchRequest) -> list[str]:
    """Fail if the output is too short or too long."""
    reasons: list[str] = []
    if len(content) < settings.min_output_chars:
        reasons.append(f"Output too short ({len(content)} chars < {settings.min_output_chars})")
    if len(content) > settings.max_output_chars:
        reasons.append(f"Output too long ({len(content)} chars > {settings.max_output_chars})")
    return reasons


def check_not_refusal(content: str, request: OrchRequest) -> list[str]:
    """Catch common refusal patterns that indicate the model declined the task."""
    lowered = content.lower().strip()
    refusal_prefixes = [
        "i'm sorry",
        "i cannot",
        "i can't",
        "as an ai",
        "i am unable",
    ]
    for prefix in refusal_prefixes:
        if lowered.startswith(prefix):
            return [f"Model appears to have refused (starts with '{prefix}')"]
    return []


def check_json_if_schema(content: str, request: OrchRequest) -> list[str]:
    """If the request asked for structured output, verify the response is valid JSON."""
    if not request.needs_schema:
        return []
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return [f"Expected JSON output but got invalid JSON: {exc}"]
    return []


def check_code_fence(content: str, request: OrchRequest) -> list[str]:
    """For code tasks, verify a code fence is present."""
    if request.task_type != TaskType.CODE:
        return []
    if "```" not in content:
        return ["Code task response missing code fence (```)"]
    return []


# ---------------------------------------------------------------------------
# Validator pipeline
# ---------------------------------------------------------------------------

ALL_VALIDATORS: list[Validator] = [
    check_length,
    check_not_refusal,
    check_json_if_schema,
    check_code_fence,
]


def run_validators(content: str, request: OrchRequest) -> list[str]:
    """Run all validators and return aggregated failure reasons."""
    reasons: list[str] = []
    for v in ALL_VALIDATORS:
        reasons.extend(v(content, request))
    return reasons
