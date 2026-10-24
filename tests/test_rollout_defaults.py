"""The default thresholds, tested at their exact boundaries."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
from conftest import ROUTES, SKIPPABLE

from earned_gate import (
    BoundaryError,
    Ledger,
    RolloutPolicy,
    RolloutReport,
    assess,
    validate_context,
)
from earned_gate.ledger import EvaluationRecord
from earned_gate.testing import FakeClock

_ids = itertools.count()


def add(
    led: Ledger,
    clock: FakeClock,
    n: int,
    *,
    shadow: str | None = "ignore",
    label: str | None = "ignore",
    fallback: bool = False,
) -> None:
    for _ in range(n):
        clock.advance(1)
        eval_id = f"e{next(_ids)}"
        led.record_evaluation(
            EvaluationRecord(
                eval_id=eval_id,
                ts=clock(),
                feature="f",
                model_version="m1",
                state_hash="h",
                status="provider_error" if fallback else "ok",
                final_route="agent",
                shadow_route=None if fallback else shadow,
                answers=(),
                tokens=0,
                fallback=fallback,
                suppressed=False,
            )
        )
        if label is not None and not fallback:
            led.label(eval_id, label)


def report(led: Ledger) -> RolloutReport:
    return assess(
        led,
        feature="f",
        model_version="m1",
        routes=ROUTES,
        skippable=SKIPPABLE,
        policy=RolloutPolicy(),
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def led(clock: FakeClock) -> Iterator[Ledger]:
    with Ledger(":memory:", clock=clock) as ledger:
        yield ledger


@pytest.mark.parametrize(("n", "is_open"), [(29, False), (30, True)])
def test_min_labelled_is_30(led: Ledger, clock: FakeClock, n: int, is_open: bool) -> None:
    add(led, clock, n)
    r = report(led)
    assert r.open is is_open, r.reasons
    assert any("labelled evaluations" in x for x in r.reasons) is not is_open


@pytest.mark.parametrize(("n", "is_open"), [(14, False), (15, True)])
def test_min_would_suppress_is_15(led: Ledger, clock: FakeClock, n: int, is_open: bool) -> None:
    add(led, clock, n)
    add(led, clock, 30 - n, shadow="agent", label="agent")
    r = report(led)
    assert r.open is is_open, r.reasons


@pytest.mark.parametrize(("mismatches", "is_open"), [(6, True), (7, False)])
def test_max_miss_score_is_0_15(
    led: Ledger, clock: FakeClock, mismatches: int, is_open: bool
) -> None:
    # 40 labelled, no false negatives: 6/40 = 0.15 passes, 7/40 = 0.175 does not.
    add(led, clock, 20)
    add(led, clock, 20 - mismatches, shadow="ask", label="ask")
    add(led, clock, mismatches, shadow="ask", label="defer")
    r = report(led)
    assert r.false_negatives == 0
    assert r.open is is_open, r.reasons


@pytest.mark.parametrize(("fallbacks", "is_open"), [(10, True), (11, False)])
def test_max_fallback_rate_is_0_2(
    led: Ledger, clock: FakeClock, fallbacks: int, is_open: bool
) -> None:
    # Fallbacks first, so the recent window (last 20) is clean: 10/50 = 0.2 passes.
    add(led, clock, fallbacks, fallback=True)
    add(led, clock, 40)
    r = report(led)
    assert r.recent_fallback_rate == 0.0
    assert r.open is is_open, r.reasons


@pytest.mark.parametrize(("fallbacks", "is_open"), [(10, True), (11, False)])
def test_max_recent_fallback_rate_is_0_5_over_20(
    led: Ledger, clock: FakeClock, fallbacks: int, is_open: bool
) -> None:
    add(led, clock, 60)
    add(led, clock, 20 - fallbacks)
    add(led, clock, fallbacks, fallback=True)
    r = report(led)
    assert r.fallback_rate <= 0.2
    assert r.open is is_open, r.reasons
    assert any("degraded" in x for x in r.reasons) is not is_open


@pytest.mark.parametrize(("size", "ok"), [(8192, True), (8193, False)])
def test_default_context_limit_is_8192_bytes(size: int, ok: bool) -> None:
    ctx = {"x": "a" * (size - len('{"x":""}'))}
    if ok:
        assert len(validate_context(ctx).encode()) == size
    else:
        with pytest.raises(BoundaryError, match="8193 bytes"):
            validate_context(ctx)
