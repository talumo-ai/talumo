"""Tier classification: decide which backend to try first."""

from __future__ import annotations

from zora.schemas import Finality, OrchRequest, Sensitivity, TaskType, Tier

# ---------------------------------------------------------------------------
# Escalation ladder
# ---------------------------------------------------------------------------

ESCALATION_ORDER: list[Tier] = [Tier.LOCAL, Tier.REMOTE_CHEAP, Tier.REMOTE_FRONTIER]


def next_tier(current: Tier) -> Tier | None:
    """Return the next tier in the escalation ladder, or None if at the top."""
    idx = ESCALATION_ORDER.index(current)
    if idx + 1 < len(ESCALATION_ORDER):
        return ESCALATION_ORDER[idx + 1]
    return None


# ---------------------------------------------------------------------------
# Initial tier selection
# ---------------------------------------------------------------------------

def classify(req: OrchRequest) -> Tier:
    """Choose the starting tier for a request.

    Rules (evaluated in order):
    1. ``final`` finality  → remote_frontier
    2. ``analysis`` or ``code`` task with final finality → remote_frontier
    3. ``public`` sensitivity → remote_cheap  (data can leave the machine)
    4. everything else → local
    """
    # Rule 1 – final deliverables always go to frontier
    if req.finality == Finality.FINAL:
        return Tier.REMOTE_FRONTIER

    # Rule 2 – complex tasks that explicitly need schemas or tests
    if req.task_type in {TaskType.ANALYSIS, TaskType.CODE} and (
        req.needs_schema or req.can_test
    ):
        return Tier.REMOTE_CHEAP

    # Rule 3 – public data can leave the machine → use cheap remote
    if req.sensitivity == Sensitivity.PUBLIC:
        return Tier.REMOTE_CHEAP

    # Default – keep it local
    return Tier.LOCAL
