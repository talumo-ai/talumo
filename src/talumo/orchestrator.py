"""Core orchestration loop.

Flow:
  1. classify → initial tier
  2. call_model on that tier
  3. run_validators
  4. if failures are repairable → RETRY_SAME_TIER (once)
  5. if still failing → ESCALATE to next tier (once)
  6. return result + full trace
"""

from __future__ import annotations

import time

import httpx

from talumo.backend import ModelResult, RemoteAPIKeyMissingError, call_model
from talumo.classifier import classify, next_tier
from talumo.schemas import (
    OrchRequest,
    OrchResponse,
    Tier,
    TraceStep,
    Verdict,
)
from talumo.validators import run_validators

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decide(reasons: list[str], already_retried: bool) -> Verdict:
    """Translate validator reasons into a routing verdict."""
    if not reasons:
        return Verdict.PASS
    if not already_retried:
        return Verdict.RETRY_SAME_TIER
    return Verdict.ESCALATE


def _make_step(
    tier: Tier, model: str, verdict: Verdict, reasons: list[str], elapsed_ms: float
) -> TraceStep:
    return TraceStep(
        tier=tier, model=model, verdict=verdict, reasons=reasons, latency_ms=round(elapsed_ms, 1)
    )


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

async def orchestrate(req: OrchRequest, client: httpx.AsyncClient) -> OrchResponse:
    """Execute the routing / validation / escalation loop and return a response."""
    tier = classify(req)
    trace: list[TraceStep] = []
    retried = False
    result: ModelResult | None = None

    for _attempt in range(3):  # at most: initial + retry + escalate
        t0 = time.monotonic()
        try:
            result = await call_model(tier, req.messages, client=client)
        except RemoteAPIKeyMissingError as exc:
            elapsed = (time.monotonic() - t0) * 1000
            trace.append(_make_step(tier, "unknown", Verdict.ESCALATE, [str(exc)], elapsed))
            break
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
            elapsed = (time.monotonic() - t0) * 1000
            reasons = [f"Backend error: {exc!r}"]
            verdict = _decide(reasons, already_retried=retried)
            trace.append(_make_step(tier, "unknown", verdict, reasons, elapsed))
            if verdict == Verdict.ESCALATE:
                nxt = next_tier(tier)
                if nxt is None:
                    break  # top tier already; return whatever we have
                tier = nxt
                retried = False
                continue
            # RETRY_SAME_TIER
            retried = True
            continue

        elapsed = (time.monotonic() - t0) * 1000
        reasons = run_validators(result.content, req)
        verdict = _decide(reasons, already_retried=retried)
        trace.append(_make_step(tier, result.model, verdict, reasons, elapsed))

        if verdict == Verdict.PASS:
            break

        if verdict == Verdict.RETRY_SAME_TIER:
            retried = True
            continue

        # ESCALATE
        nxt = next_tier(tier)
        if nxt is None:
            break  # already at top tier; return best-effort
        tier = nxt
        retried = False

    content = result.content if result else ""
    finish = result.finish_reason if result else "error"

    return OrchResponse(
        request_id=req.metadata.request_id,
        content=content,
        finish_reason=finish,
        tier_used=tier,
        trace=trace,
    )
