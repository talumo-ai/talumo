"""Tests for tier classification logic."""

from zora.classifier import classify, next_tier
from zora.schemas import (
    Finality,
    Message,
    OrchRequest,
    Sensitivity,
    TaskType,
    Tier,
)


def _req(**kw: object) -> OrchRequest:
    defaults: dict[str, object] = {
        "task_type": TaskType.DRAFT,
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(kw)
    return OrchRequest(**defaults)  # type: ignore[arg-type]


class TestClassify:
    def test_default_is_local(self) -> None:
        assert classify(_req()) == Tier.LOCAL

    def test_final_finality_goes_to_frontier(self) -> None:
        assert classify(_req(finality=Finality.FINAL)) == Tier.REMOTE_FRONTIER

    def test_public_goes_to_cheap(self) -> None:
        assert classify(_req(sensitivity=Sensitivity.PUBLIC)) == Tier.REMOTE_CHEAP

    def test_code_with_can_test_goes_to_cheap(self) -> None:
        assert classify(_req(task_type=TaskType.CODE, can_test=True)) == Tier.REMOTE_CHEAP

    def test_analysis_with_needs_schema_goes_to_cheap(self) -> None:
        assert (
            classify(_req(task_type=TaskType.ANALYSIS, needs_schema=True)) == Tier.REMOTE_CHEAP
        )

    def test_private_draft_stays_local(self) -> None:
        req = _req(sensitivity=Sensitivity.PRIVATE, finality=Finality.DRAFT)
        assert classify(req) == Tier.LOCAL


class TestNextTier:
    def test_local_escalates_to_cheap(self) -> None:
        assert next_tier(Tier.LOCAL) == Tier.REMOTE_CHEAP

    def test_cheap_escalates_to_frontier(self) -> None:
        assert next_tier(Tier.REMOTE_CHEAP) == Tier.REMOTE_FRONTIER

    def test_frontier_has_no_next(self) -> None:
        assert next_tier(Tier.REMOTE_FRONTIER) is None
