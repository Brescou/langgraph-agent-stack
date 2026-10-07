"""api/guardrails.py — Apply the opt-in guardrail engine at the API boundaries.

``screen`` is the single entry point used by the typed pack routes (REST and
MCP) and the legacy pipeline routes. When ``GUARDRAILS_ENABLED`` is false the
lifespan leaves ``state.guardrail_rule_sets`` unset and ``screen`` returns
before evaluating anything.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, status

import api.state as state
from control_plane.enforce import guardrail_rule_sets
from core.guardrails import GuardrailPhase, Verdict, evaluate
from core.observability import outcome_from_http_status, record_guardrail_finding

logger = logging.getLogger(__name__)

#: Error code shared by the HTTP body and the streaming error event.
GUARDRAIL_BLOCKED_CODE = "guardrail_blocked"

#: Generic message; never echoes the matched text or the rule that fired.
GUARDRAIL_BLOCKED_MESSAGE = "Request blocked by a guardrail policy."

#: Outcome label recorded on ``pack_runs_total`` for blocked runs.
GUARDRAIL_BLOCKED_OUTCOME = "guardrail_blocked"

#: Pack stream event carrying the final result; the only one screened.
FINAL_STREAM_EVENT_TYPE = "pipeline_completed"


class GuardrailBlockedError(HTTPException):
    """A guardrail rule with action ``block`` matched.

    Input blocks map to 422 (the caller sent something refused); output blocks
    map to 502 (the model produced something the service will not return).
    """

    phase: GuardrailPhase

    def __init__(self, phase: GuardrailPhase) -> None:
        status_code = (
            status.HTTP_422_UNPROCESSABLE_CONTENT
            if phase == "input"
            else status.HTTP_502_BAD_GATEWAY
        )
        super().__init__(
            status_code=status_code,
            detail={
                "code": GUARDRAIL_BLOCKED_CODE,
                "phase": phase,
                "message": GUARDRAIL_BLOCKED_MESSAGE,
            },
        )
        self.phase = phase


def screen(
    pack_id: str,
    phase: GuardrailPhase,
    data: Any,
    *,
    run_id: str | None = None,
) -> Verdict | None:
    """Evaluate ``data`` against the rule sets bound to ``pack_id`` for ``phase``.

    Args:
        pack_id: Pack whose ``PackPolicy.guardrails`` selects the rule sets.
        phase: ``"input"`` for the request body, ``"output"`` for the result.
        data: JSON-like payload; every string field is scanned.
        run_id: Correlation id for the log event, when known.

    Returns:
        The verdict, or ``None`` when guardrails are disabled or the pack has
        no rule set for this phase.

    Raises:
        GuardrailBlockedError: when any matching rule has action ``block``.
    """
    loaded = state.guardrail_rule_sets
    if loaded is None:
        return None
    rule_sets = guardrail_rule_sets(pack_id, phase, loaded)
    if not rule_sets:
        return None

    verdict = evaluate(data, rule_sets)
    if not verdict.findings:
        return verdict

    for finding in verdict.findings:
        record_guardrail_finding(
            pack_id=pack_id,
            phase=phase,
            rule_set=finding.rule_set,
            rule_id=finding.rule_id,
            action=finding.action,
        )
        logger.warning(
            "guardrail finding",
            extra={
                "event": "guardrail_finding",
                "pack_id": pack_id,
                "run_id": run_id,
                "phase": phase,
                "rule_set": finding.rule_set,
                "rule_id": finding.rule_id,
                "field_path": finding.field_path,
                "action": finding.action,
            },
        )
    if verdict.action == "block":
        raise GuardrailBlockedError(phase)
    return verdict


def screen_stream_event(
    pack_id: str, event: dict[str, Any], *, run_id: str | None = None
) -> Verdict | None:
    """Screen a pack stream event if it carries the final result.

    Only ``pipeline_completed`` is screened; intermediate events (tokens,
    phase markers) pass through untouched.
    """
    if event.get("type") != FINAL_STREAM_EVENT_TYPE:
        return None
    payload = {key: value for key, value in event.items() if key != "type"}
    return screen(pack_id, "output", payload, run_id=run_id)


def screenable_payload(value: Any) -> Any:
    """Return the JSON form of ``value`` as the client would receive it."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def escalation_reason(
    input_verdict: Verdict | None, output_verdict: Verdict | None
) -> str | None:
    """Build the review-queue reason for escalating findings, if any.

    Only ``escalate`` findings are listed; ``flag`` findings stay in logs and
    metrics. Example: ``"guardrail: pii_basic/email (output)"``.
    """
    parts: list[str] = []
    for phase, verdict in (("input", input_verdict), ("output", output_verdict)):
        if verdict is None:
            continue
        refs = list(
            dict.fromkeys(
                f"{finding.rule_set}/{finding.rule_id}"
                for finding in verdict.findings
                if finding.action == "escalate"
            )
        )
        if refs:
            parts.append(f"{', '.join(refs)} ({phase})")
    if not parts:
        return None
    return "guardrail: " + "; ".join(parts)


def pack_run_outcome(exc: HTTPException) -> str:
    """Map an HTTPException to the ``pack_runs_total`` outcome label."""
    if isinstance(exc, GuardrailBlockedError):
        return GUARDRAIL_BLOCKED_OUTCOME
    return outcome_from_http_status(exc.status_code)


def blocked_stream_event(phase: GuardrailPhase) -> dict[str, str]:
    """SSE error event sent in place of ``done`` when the final output is blocked."""
    return {
        "type": "error",
        "code": GUARDRAIL_BLOCKED_CODE,
        "phase": phase,
        "message": GUARDRAIL_BLOCKED_MESSAGE,
    }
