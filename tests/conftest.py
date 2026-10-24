from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import pytest

from earned_gate import (
    BooleanQuestion,
    ChoiceQuestion,
    Gate,
    GateConfig,
    Ledger,
    Question,
    RolloutPolicy,
)
from earned_gate.testing import FakeClock, ScriptedProvider

ROUTES = ("ignore", "persist", "ask", "defer", "agent")
SKIPPABLE = frozenset({"ignore", "persist"})


def route_probs(route: str, p: float = 0.97) -> dict[str, Any]:
    rest = (1.0 - p) / (len(ROUTES) - 1)
    return {"probs": {r: (p if r == route else rest) for r in ROUTES}}


def questions(
    prompt: str = "Which route best matches the required response?",
) -> dict[str, Question]:
    return {
        "material": BooleanQuestion(
            "Could this change plausibly alter the next coaching decision?", fallback=True
        ),
        "route": ChoiceQuestion(prompt, options=ROUTES, fallback="agent"),
    }


def answers(route: str = "ignore", p: float = 0.97, p_material: float = 0.03) -> dict[str, Any]:
    return {"material": {"p_yes": p_material}, "route": route_probs(route, p)}


def base_config(**overrides: Any) -> GateConfig:
    cfg = GateConfig(
        feature="training-watch",
        routes=ROUTES,
        skippable=SKIPPABLE,
        route_question="route",
        materiality_question="material",
        allowlist=SKIPPABLE,
        rollout=RolloutPolicy(min_labelled=4, min_would_suppress=2, degraded_window=10),
        reuse_ttl_s=900,
    )
    return replace(cfg, **overrides)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ledger(clock: FakeClock) -> Iterator[Ledger]:
    with Ledger(":memory:", clock=clock) as led:
        yield led


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider(answers=answers())


GateFactory = Callable[..., Gate]


@pytest.fixture
def make_gate(ledger: Ledger, provider: ScriptedProvider) -> Iterator[GateFactory]:
    gates: list[Gate] = []

    def factory(**overrides: Any) -> Gate:
        prov = overrides.pop("provider", provider)
        gate = Gate(base_config(**overrides), prov, ledger)
        gates.append(gate)
        return gate

    yield factory
    for g in gates:
        g.close()


def seed_evidence(
    gate: Gate, clock: FakeClock, n: int, *, label: str = "ignore", start: int = 0
) -> list[str]:
    """Run n measurement-stage evaluations and label each one."""
    ids: list[str] = []
    for i in range(start, start + n):
        clock.advance(60)
        d = gate.evaluate({"event": "hrv_dip", "n": i}, questions())
        gate.label(d.eval_id, label)
        ids.append(d.eval_id)
    return ids
