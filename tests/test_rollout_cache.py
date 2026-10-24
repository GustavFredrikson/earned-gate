"""The rollout gate is not recomputed from 30 days of rows on every evaluate()."""

from __future__ import annotations

from typing import Any

from conftest import answers, base_config, questions, seed_evidence

from earned_gate import Gate, Ledger
from earned_gate.testing import FakeClock, ScriptedProvider

CTX = {"event": "hrv_dip", "delta": -4}


class CountingLedger(Ledger):
    scans = 0

    def evaluations(self, *args: Any, **kwargs: Any) -> Any:
        self.scans += 1
        return super().evaluations(*args, **kwargs)


def _open(clock: FakeClock, led: Ledger, **cfg: Any) -> Gate:
    with Gate(base_config(), ScriptedProvider(answers=answers()), led) as g:
        seed_evidence(g, clock, 4)
    return Gate(
        base_config(suppression_enabled=True, **cfg), ScriptedProvider(answers=answers()), led
    )


def test_repeated_evaluations_reuse_one_rollout_scan(clock: FakeClock) -> None:
    with CountingLedger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        led.scans = 0
        for i in range(10):
            clock.advance(1)
            assert gate.evaluate({**CTX, "i": i}, questions()).skip_agent
        assert led.scans == 1


def test_assess_scans_the_window_once(clock: FakeClock) -> None:
    with CountingLedger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        led.scans = 0
        gate.rollout_status()
        assert led.scans == 1


def test_cache_expires(clock: FakeClock) -> None:
    with CountingLedger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        gate.evaluate(CTX, questions())
        led.scans = 0
        clock.advance(61)
        gate.evaluate({**CTX, "i": 1}, questions())
        assert led.scans == 1


def test_label_through_the_gate_invalidates_immediately(clock: FakeClock) -> None:
    with Ledger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        first = gate.evaluate(CTX, questions())
        assert first.skip_agent
        gate.label(first.eval_id, "agent")  # a false negative
        clock.advance(1)
        d = gate.evaluate({**CTX, "i": 1}, questions())
        assert not d.skip_agent
        assert d.rollout is not None and not d.rollout.open


def test_fallbacks_invalidate_the_cache(clock: FakeClock) -> None:
    with Ledger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        assert gate.evaluate(CTX, questions()).skip_agent
        gate.provider = ScriptedProvider(raises=RuntimeError("HTTP 500"))
        for i in range(10):
            clock.advance(1)
            gate.evaluate({**CTX, "i": i}, questions())
        gate.provider = ScriptedProvider(answers=answers())
        clock.advance(1)
        d = gate.evaluate({**CTX, "i": 99}, questions())
        assert not d.skip_agent
        assert d.rollout is not None and any("fallback" in r for r in d.rollout.reasons)


def test_cache_can_be_disabled(clock: FakeClock) -> None:
    with (
        CountingLedger(":memory:", clock=clock) as led,
        _open(clock, led, rollout_cache_s=0) as gate,
    ):
        led.scans = 0
        for i in range(3):
            clock.advance(1)
            gate.evaluate({**CTX, "i": i}, questions())
        assert led.scans == 3


def test_label_from_another_connection_closes_the_gate_immediately(
    tmp_path: Any, clock: FakeClock
) -> None:
    path = tmp_path / "gate.db"
    with Ledger(path, clock=clock) as led, _open(clock, led) as gate:
        first = gate.evaluate(CTX, questions())
        assert first.skip_agent  # report is now cached as open
        # e.g. `earned-gate label` from a cron job or another process
        with Ledger(path, clock=clock, create=False) as other:
            other.label(first.eval_id, "agent")
        clock.advance(1)  # well inside rollout_cache_s
        d = gate.evaluate({**CTX, "i": 1}, questions())
        assert not d.skip_agent
        assert d.rollout is not None and "false negatives 1" in " ".join(d.rollout.reasons)


def test_label_on_the_shared_ledger_object_closes_the_gate_immediately(
    clock: FakeClock,
) -> None:
    with Ledger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        first = gate.evaluate(CTX, questions())
        assert first.skip_agent
        led.label(first.eval_id, "agent")  # bypasses Gate.label
        clock.advance(1)
        assert not gate.evaluate({**CTX, "i": 1}, questions()).skip_agent


def test_foreign_ok_evaluations_do_not_reopen_from_a_stale_report(
    tmp_path: Any, clock: FakeClock
) -> None:
    path = tmp_path / "gate.db"
    with CountingLedger(path, clock=clock) as led, _open(clock, led) as gate:
        gate.evaluate(CTX, questions())
        led.scans = 0
        with (
            Ledger(path, clock=clock, create=False) as other,
            Gate(base_config(), ScriptedProvider(answers=answers()), other) as g2,
        ):
            g2.evaluate({**CTX, "other": 1}, questions())
        clock.advance(1)
        gate.evaluate({**CTX, "i": 1}, questions())
        assert led.scans == 1  # any foreign commit forces a fresh assessment


class NoTokenLedger(CountingLedger):
    def change_token(self) -> tuple[int, int]:
        raise RuntimeError("probe failed")


def test_no_cache_without_a_working_change_token(clock: FakeClock) -> None:
    with NoTokenLedger(":memory:", clock=clock) as led, _open(clock, led) as gate:
        led.scans = 0
        for i in range(3):
            clock.advance(1)
            assert gate.evaluate({**CTX, "i": i}, questions()).skip_agent
        assert led.scans == 3
