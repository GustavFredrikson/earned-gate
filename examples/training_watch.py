"""A cron-style wake gate, modelled on the training-watch gate in the design note.

Run it with a JSON event on stdin:

    echo '{"trigger": "hrv_shift", "hrv_delta_pct": -3, "ts": "2026-10-24T08:00Z"}' \
        | python examples/training_watch.py --ledger /tmp/watch.db

It prints the decision as JSON and exits 0 only when the caller may act on a
cheap route (skip the agent). Any other exit status, including a crash, means
"invoke the agent", so the safe behaviour does not depend on this script
working.

The provider here is a toy heuristic so the example runs offline; swap in a
real typed decision model (see openai_compatible.py for an adapter sketch).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any

from earned_gate import (
    BooleanQuestion,
    ChoiceQuestion,
    Gate,
    GateConfig,
    Ledger,
    ProviderRequest,
    ProviderResponse,
    Question,
)

ROUTES = ("ignore", "persist", "ask", "defer", "agent")
#: Only these trigger types may reach the model at all; everything else wakes the agent.
GATED_TRIGGERS = frozenset({"hrv_shift", "sleep_shift"})
HIGH_RISK = frozenset({"chest_pain", "dizziness", "injury"})


def high_risk_symptom(ctx: Mapping[str, Any]) -> str | None:
    symptom = ctx.get("symptom")
    return f"high-risk symptom: {symptom}" if symptom in HIGH_RISK else None


QUESTIONS: dict[str, Question] = {
    "material": BooleanQuestion(
        "Could this change plausibly alter the next coaching decision?", fallback=True
    ),
    "route": ChoiceQuestion(
        "Which route best matches the required response?", options=ROUTES, fallback="agent"
    ),
}


class ToyProvider:
    """Small deltas are immaterial; large ones need the agent."""

    model_version = "toy-heuristic-1"

    def ask(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        delta = abs(float(request.context.get("hrv_delta_pct", 0)))
        if delta < 5:
            route, p_yes, p = "ignore", 0.04, 0.95
        elif delta < 10:
            route, p_yes, p = "persist", 0.3, 0.8
        else:
            route, p_yes, p = "agent", 0.9, 0.9
        rest = (1 - p) / (len(ROUTES) - 1)
        return ProviderResponse(
            {
                "material": {"p_yes": p_yes},
                "route": {"probs": {r: p if r == route else rest for r in ROUTES}},
            },
            tokens=0,
        )


def project(event: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit feature projection: only the fields the questions need."""
    keys = ("trigger", "hrv_delta_pct", "sleep_delta_min", "symptom", "ts")
    return {k: event[k] for k in keys if k in event}


def main(argv: list[str] | None = None, stdin: Any = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="training-watch.db")
    parser.add_argument("--suppress", action="store_true", help="stage B: allow cheap routes")
    args = parser.parse_args(argv)
    event = json.load(stdin or sys.stdin)
    if event.get("trigger") not in GATED_TRIGGERS and "symptom" not in event:
        print(json.dumps({"route": "agent", "reason": "trigger type is not gated"}))
        return 1
    config = GateConfig(
        feature="training-watch",
        routes=ROUTES,
        skippable=frozenset({"ignore", "persist"}),
        allowlist=frozenset({"ignore", "persist"}),
        route_question="route",
        materiality_question="material",
        suppression_enabled=args.suppress,
        escalation_rules=(high_risk_symptom,),
    )
    with Ledger(args.ledger) as ledger, Gate(config, ToyProvider(), ledger) as gate:
        decision = gate.evaluate(project(event), QUESTIONS)
    print(json.dumps(decision.to_json(), indent=2))
    return 0 if decision.skip_agent else 1


if __name__ == "__main__":
    raise SystemExit(main())
