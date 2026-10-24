"""evaluate() raises only BoundaryError/ConfigError; everything else fails safe."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from typing import Any

import pytest
from conftest import GateFactory, answers, base_config, questions, seed_evidence

from earned_gate import ConfigError, Gate, Ledger
from earned_gate.testing import FakeClock, ScriptedProvider

CTX = {"event": "hrv_dip", "delta": -4}


def broken_rule(context: Mapping[str, Any]) -> str | None:
    raise KeyError("symptom")


def test_escalation_rule_error_fails_safe_without_calling_the_provider(
    make_gate: GateFactory,
    provider: ScriptedProvider,
    ledger: Ledger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = make_gate(escalation_rules=(broken_rule,))
    with caplog.at_level(logging.WARNING, logger="earned_gate"):
        d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.status, d.recorded) == ("agent", False, "rule_error", True)
    assert "broken_rule" in d.reason and "KeyError" in d.reason
    assert provider.calls == []
    assert any("broken_rule" in r.getMessage() for r in caplog.records)
    [rec] = ledger.evaluations("training-watch", since=0)
    assert rec.status == "rule_error" and rec.fallback and rec.shadow_route is None


def test_rule_errors_close_the_gate_via_fallback_rate(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    seed_evidence(make_gate(), clock, 4)
    gate = make_gate(suppression_enabled=True, escalation_rules=(broken_rule,))
    for _ in range(10):
        clock.advance(60)
        gate.evaluate(CTX, questions())
    report = gate.rollout_status()
    assert not report.open
    assert report.fallback_rate > 0.2


def test_rule_error_does_not_label_earlier_suppressions(
    make_gate: GateFactory, clock: FakeClock, ledger: Ledger
) -> None:
    seed_evidence(make_gate(), clock, 4)
    make_gate(escalation_rules=(broken_rule,)).evaluate(CTX, questions())
    labels = [r.label for r in ledger.evaluations("training-watch", since=0) if r.label]
    assert labels == ["ignore"] * 4


def test_closed_ledger_fails_safe(
    clock: FakeClock, provider: ScriptedProvider, caplog: pytest.LogCaptureFixture
) -> None:
    led = Ledger(":memory:", clock=clock)
    led.close()
    with Gate(base_config(), provider, led) as gate, caplog.at_level(logging.WARNING):
        d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.status, d.recorded) == ("agent", False, "ledger_error", False)
    assert "ProgrammingError" in d.reason
    assert provider.calls == []
    assert any("ledger" in r.getMessage() for r in caplog.records)


class FailingWrites(Ledger):
    fail_writes = False

    def record_evaluation(self, record: Any) -> None:
        if self.fail_writes:
            raise sqlite3.OperationalError("database or disk is full")
        super().record_evaluation(record)


def test_ledger_write_error_after_an_open_gate_never_skips(
    clock: FakeClock, provider: ScriptedProvider
) -> None:
    with FailingWrites(":memory:", clock=clock) as led:
        with Gate(base_config(), provider, led) as g:
            seed_evidence(g, clock, 4)
        led.fail_writes = True
        with Gate(base_config(suppression_enabled=True), provider, led) as gate:
            assert gate.rollout_status().open
            d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.status, d.recorded) == ("agent", False, "ledger_error", False)
    assert d.shadow_route == "ignore"
    assert d.tokens == 100  # the provider was asked; its answers are kept
    assert d.answers["route"].value == "ignore"
    assert "disk is full" in d.reason


class FailingReads(Ledger):
    def find_reusable(self, *args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("database is locked")


def test_ledger_read_error_fails_safe(clock: FakeClock, provider: ScriptedProvider) -> None:
    with FailingReads(":memory:", clock=clock) as led, Gate(base_config(), provider, led) as g:
        d = g.evaluate(CTX, questions())
        assert (d.status, d.route, d.recorded) == ("ledger_error", "agent", False)
        assert "locked" in d.reason
        assert provider.calls == []
        with pytest.raises(ConfigError, match="unknown evaluation"):
            g.label(d.eval_id, "agent")


class FailingLabels(Ledger):
    def label_missed_suppressions(self, *args: Any, **kwargs: Any) -> list[str]:
        raise sqlite3.OperationalError("database is locked")


def test_ledger_error_during_escalation_still_escalates(
    clock: FakeClock, provider: ScriptedProvider
) -> None:
    with FailingLabels(":memory:", clock=clock) as led, Gate(base_config(), provider, led) as g:
        d = g.evaluate(CTX, questions(), escalate="chest pain")
    assert (d.status, d.route, d.skip_agent, d.recorded) == ("ledger_error", "agent", False, False)
    assert "chest pain" in d.reason and "locked" in d.reason


class BrokenVersion(ScriptedProvider):
    @property
    def model_version(self) -> str:
        raise RuntimeError("config not loaded")


def test_provider_model_version_error_fails_safe(ledger: Ledger) -> None:
    prov = BrokenVersion(answers=answers())
    with Gate(base_config(), prov, ledger) as g:
        d = g.evaluate(CTX, questions())
    assert (d.status, d.route, d.skip_agent, d.recorded) == (
        "provider_unavailable",
        "agent",
        False,
        False,
    )
    assert "model_version" in d.reason
    assert prov.calls == []
    assert ledger.evaluations("training-watch", since=0) == []


class NonStrVersion(ScriptedProvider):
    @property
    def model_version(self) -> str:
        return 3  # type: ignore[return-value]


def test_non_string_model_version_fails_safe(ledger: Ledger) -> None:
    with Gate(base_config(), NonStrVersion(answers=answers()), ledger) as g:
        d = g.evaluate(CTX, questions())
    assert (d.status, d.recorded) == ("provider_unavailable", False)
    assert "not str" in d.reason


def test_integration_bugs_still_raise(make_gate: GateFactory) -> None:
    with pytest.raises(ConfigError):
        make_gate().evaluate(CTX, {"route": questions()["route"]}, signals={"x": 1.0})


def test_rule_error_with_a_failing_ledger(clock: FakeClock, provider: ScriptedProvider) -> None:
    with FailingWrites(":memory:", clock=clock) as led:
        led.fail_writes = True
        with Gate(base_config(escalation_rules=(broken_rule,)), provider, led) as g:
            d = g.evaluate(CTX, questions())
    assert (d.status, d.route, d.recorded) == ("ledger_error", "agent", False)
    assert "broken_rule" in d.reason and "disk is full" in d.reason


# -- signal values are validated -------------------------------------------------------


def _floor_gate(make_gate: GateFactory) -> Gate:
    from earned_gate import SignalFloor

    floor = SignalFloor(weights={"sleep": 1.0, "soreness": 1.0}, threshold=0.8, raise_to="defer")
    return make_gate(signal_floor=floor)


@pytest.mark.parametrize(
    "value", ["high", None, float("nan"), float("inf"), float("-inf"), True, [0.5], {"v": 1}]
)
def test_bad_signal_values_raise_boundary_error(
    make_gate: GateFactory, provider: ScriptedProvider, value: object
) -> None:
    from earned_gate import BoundaryError

    gate = _floor_gate(make_gate)
    with pytest.raises(BoundaryError, match=r"signals\.sleep"):
        gate.evaluate(CTX, questions(), signals={"sleep": value, "soreness": 0.1})  # type: ignore[dict-item]
    assert provider.calls == []


def test_non_mapping_signals_raise_boundary_error(make_gate: GateFactory) -> None:
    from earned_gate import BoundaryError

    with pytest.raises(BoundaryError, match="signals must be a mapping"):
        _floor_gate(make_gate).evaluate(CTX, questions(), signals=[("sleep", 1.0)])  # type: ignore[arg-type]


def test_int_signals_are_fine(make_gate: GateFactory) -> None:
    d = _floor_gate(make_gate).evaluate(CTX, questions(), signals={"sleep": 1, "soreness": 0})
    assert d.shadow_route == "defer"


@pytest.mark.parametrize(
    ("weights", "threshold"),
    [
        ({"sleep": float("nan")}, 0.8),
        ({"sleep": 1.0}, float("nan")),
        ({"sleep": float("inf")}, 1.0),
    ],
)
def test_non_finite_signal_floor_config_is_rejected(
    weights: dict[str, float], threshold: float
) -> None:
    from earned_gate import SignalFloor

    with pytest.raises(ConfigError, match="finite"):
        base_config(
            signal_floor=SignalFloor(weights=weights, threshold=threshold, raise_to="defer")
        )


def test_non_string_signal_names_raise_boundary_error(make_gate: GateFactory) -> None:
    from earned_gate import BoundaryError

    with pytest.raises(BoundaryError, match="names must be strings"):
        _floor_gate(make_gate).evaluate(CTX, questions(), signals={1: 0.5})  # type: ignore[dict-item]


def test_tokens_spent_before_a_failed_write_count_against_the_budget(
    clock: FakeClock, provider: ScriptedProvider
) -> None:
    with FailingWrites(":memory:", clock=clock) as led:
        led.fail_writes = True
        with Gate(base_config(daily_token_budget=150), provider, led) as gate:
            lost = gate.evaluate(CTX, questions())
            assert (lost.status, lost.tokens) == ("ledger_error", 100)
            led.fail_writes = False
            clock.advance(60)
            assert gate.evaluate({**CTX, "i": 1}, questions()).status == "ok"  # 100 < 150
            clock.advance(60)
            # 100 recorded + 100 unrecorded >= 150
            assert gate.evaluate({**CTX, "i": 2}, questions()).status == "budget_exhausted"
            # ...and the unrecorded spend ages out after a day like recorded spend.
            clock.advance(86400 - 60)
            assert gate.evaluate({**CTX, "i": 3}, questions()).status == "ok"


def test_failed_write_without_token_spend_adds_nothing(clock: FakeClock) -> None:
    free = ScriptedProvider(answers=answers(), tokens=0)
    with FailingWrites(":memory:", clock=clock) as led:
        led.fail_writes = True
        with Gate(base_config(daily_token_budget=1), free, led) as gate:
            assert gate.evaluate(CTX, questions()).status == "ledger_error"
            led.fail_writes = False
            assert gate.evaluate({**CTX, "i": 1}, questions()).status == "ok"


# -- questions (and escalate) are validated --------------------------------------------


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (None, "questions must be a mapping"),
        (42, "questions must be a mapping"),
        ([("route", "x")], "questions must be a mapping"),
        ("route", "questions must be a mapping"),
        ({**questions(), "extra": "Is it bad?"}, r"questions\['extra'\] must be"),
        ({**questions(), "extra": None}, r"questions\['extra'\] must be"),
        ({**questions(), 1: questions()["material"]}, "question ids must be strings"),
    ],
)
def test_bad_questions_raise_config_error(
    make_gate: GateFactory, provider: ScriptedProvider, bad: object, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        make_gate().evaluate(CTX, bad)  # type: ignore[arg-type]
    assert provider.calls == []


@pytest.mark.parametrize("bad", [1, True, ["x"], {"why": "x"}])
def test_non_string_escalate_raises_config_error(
    make_gate: GateFactory, provider: ScriptedProvider, bad: object
) -> None:
    with pytest.raises(ConfigError, match="escalate must be a string or None"):
        make_gate().evaluate(CTX, questions(), escalate=bad)  # type: ignore[arg-type]
    assert provider.calls == []


@pytest.mark.parametrize("opts", [(1j, 2j), (b"a", b"b")])
def test_non_json_choice_options_raise_config_error(
    make_gate: GateFactory, provider: ScriptedProvider, opts: tuple[object, object]
) -> None:
    from earned_gate import ChoiceQuestion

    q = ChoiceQuestion("Which one?", options=opts, fallback=opts[0])  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match=r"questions\['x'\] is not JSON-serialisable"):
        make_gate().evaluate(CTX, {**questions(), "x": q})
    assert provider.calls == []
