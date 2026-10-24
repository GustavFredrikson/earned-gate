from __future__ import annotations

from typing import Any

import pytest
from conftest import ROUTES, SKIPPABLE, GateFactory, answers, questions, seed_evidence

from earned_gate import ConfigError, Ledger, RolloutPolicy, assess
from earned_gate.testing import FakeClock, ScriptedProvider

POLICY = RolloutPolicy(min_labelled=4, min_would_suppress=2, degraded_window=10)


def report(ledger: Ledger, policy: RolloutPolicy = POLICY, version: str = "scripted-1") -> Any:
    return assess(
        ledger,
        feature="training-watch",
        model_version=version,
        routes=ROUTES,
        skippable=SKIPPABLE,
        policy=policy,
    )


def test_empty_ledger_is_closed(ledger: Ledger) -> None:
    r = report(ledger)
    assert not r.open
    assert r.miss_score == 1.0
    assert any("labelled" in x for x in r.reasons)


def test_enough_clean_evidence_opens(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    seed_evidence(make_gate(), clock, 4)
    r = report(ledger)
    assert r.open, r.reasons
    assert (r.labelled, r.would_suppress, r.false_negatives, r.miss_score) == (4, 4, 0, 0.0)
    assert r.to_json()["open"] is True


def test_single_false_negative_closes(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    gate = make_gate()
    seed_evidence(gate, clock, 10)
    seed_evidence(gate, clock, 1, label="agent", start=100)
    r = report(ledger)
    assert r.false_negatives == 1
    assert not r.open
    assert r.miss_score == pytest.approx(3 / 11, abs=1e-6)


def test_over_escalation_counts_once_in_miss_score(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    gate = make_gate(provider=ScriptedProvider(answers=answers("defer")))
    seed_evidence(gate, clock, 4, label="ignore")
    r = report(ledger, RolloutPolicy(min_labelled=4, min_would_suppress=0))
    assert (r.false_negatives, r.mismatches, r.miss_score) == (0, 4, 1.0)
    assert not r.open


def test_unlabelled_evidence_does_not_count(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    gate = make_gate()
    for i in range(10):
        clock.advance(60)
        gate.evaluate({"n": i}, questions())
    r = report(ledger)
    assert (r.evaluations, r.labelled) == (10, 0)
    assert not r.open


def test_high_fallback_rate_closes(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    seed_evidence(make_gate(), clock, 4)
    broken = make_gate(provider=ScriptedProvider(raises=RuntimeError("down")))
    for i in range(2):
        clock.advance(60)
        broken.evaluate({"n": 50 + i}, questions())
    r = report(ledger)
    assert r.fallback_rate == pytest.approx(2 / 6, abs=1e-6)
    assert any("fallback rate" in x for x in r.reasons)


def test_degraded_provider_closes_even_for_a_new_version(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    seed_evidence(make_gate(), clock, 4)
    broken = make_gate(provider=ScriptedProvider(raises=RuntimeError("down"), version="v2"))
    for i in range(6):
        clock.advance(60)
        broken.evaluate({"n": 70 + i}, questions())
    r = report(
        ledger,
        RolloutPolicy(
            min_labelled=4, min_would_suppress=2, max_fallback_rate=1.0, degraded_window=10
        ),
    )
    assert any("provider degraded" in x for x in r.reasons)


def test_new_model_version_starts_from_zero(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    seed_evidence(make_gate(), clock, 4)
    assert report(ledger).open
    assert report(ledger, version="scripted-2").labelled == 0


def test_window_expiry(make_gate: GateFactory, clock: FakeClock, ledger: Ledger) -> None:
    seed_evidence(make_gate(), clock, 4)
    clock.advance(31 * 86400)
    assert not report(ledger).open


def test_fails_closed_on_errors(clock: FakeClock) -> None:
    led = Ledger(":memory:", clock=clock)
    led.close()
    r = report(led)
    assert not r.open
    assert r.reasons[0].startswith("error while assessing")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0},
        {"min_labelled": 0},
        {"max_false_negatives": -1},
        {"false_negative_weight": 0.5},
        {"max_miss_score": 2.0},
        {"max_fallback_rate": -0.1},
    ],
)
def test_policy_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        RolloutPolicy(**kwargs)
