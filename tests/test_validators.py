"""Tests for output validators."""

from zora.schemas import Message, OrchRequest, TaskType
from zora.validators import (
    check_code_fence,
    check_json_if_schema,
    check_length,
    check_not_refusal,
    run_validators,
)


def _req(**kw: object) -> OrchRequest:
    defaults: dict[str, object] = {
        "task_type": TaskType.DRAFT,
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(kw)
    return OrchRequest(**defaults)  # type: ignore[arg-type]


class TestCheckLength:
    def test_too_short(self) -> None:
        assert check_length("hi", _req())

    def test_ok_length(self) -> None:
        assert not check_length("x" * 50, _req())

    def test_too_long(self) -> None:
        assert check_length("x" * 60_000, _req())


class TestCheckNotRefusal:
    def test_refusal_detected(self) -> None:
        assert check_not_refusal("I'm sorry, I cannot help with that.", _req())

    def test_normal_response(self) -> None:
        assert not check_not_refusal("Here is the analysis you requested...", _req())

    def test_as_an_ai_refusal(self) -> None:
        assert check_not_refusal("As an AI language model, I cannot", _req())


class TestCheckJsonIfSchema:
    def test_valid_json_when_needed(self) -> None:
        assert not check_json_if_schema('{"key": "value"}', _req(needs_schema=True))

    def test_invalid_json_when_needed(self) -> None:
        assert check_json_if_schema("not json", _req(needs_schema=True))

    def test_no_check_when_not_needed(self) -> None:
        assert not check_json_if_schema("not json", _req(needs_schema=False))


class TestCheckCodeFence:
    def test_code_task_with_fence(self) -> None:
        assert not check_code_fence("```python\nprint('hi')\n```", _req(task_type=TaskType.CODE))

    def test_code_task_without_fence(self) -> None:
        assert check_code_fence("print('hi')", _req(task_type=TaskType.CODE))

    def test_non_code_task_no_fence_ok(self) -> None:
        assert not check_code_fence("just text", _req(task_type=TaskType.DRAFT))


class TestRunValidators:
    def test_aggregates_multiple_failures(self) -> None:
        reasons = run_validators("hi", _req())  # too short
        assert len(reasons) >= 1

    def test_passes_good_output(self) -> None:
        good = "This is a perfectly reasonable draft response with enough content."
        reasons = run_validators(good, _req())
        assert reasons == []
