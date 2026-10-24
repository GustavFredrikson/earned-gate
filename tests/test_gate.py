from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from conftest import ROUTES, SKIPPABLE, GateFactory, answers, base_config, questions, seed_evidence

from earned_gate import (
    BooleanQuestion,
    BoundaryError,
    ChoiceQuestion,
    ConfigError,
    Gate,
    Ledger,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailable,
    ScoreQuestion,
    SignalFloor,
)
from earned_gate.testing import FakeClock, ScriptedProvider

CTX = {"event": "hrv_dip", "delta": -4, "ts": "2026-10-24T08:00:00Z"}


def open_gate(make_gate: GateFactory, clock: FakeClock, **overrides: Any) -> Gate:
    """Seed passing evidence in measurement stage, then return a stage-B gate."""
    seed_evidence(make_gate(), clock, 4)
    return make_gate(suppression_enabled=True, **overrides)


# -- measurement stage -----------------------------------------------------------------


def test_measurement_stage_never_skips_but_records_shadow(
    make_gate: GateFactory, provider: ScriptedProvider, ledger: Ledger
) -> None:
    d = make_gate().evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.status, d.stage) == ("agent", False, "ok", "A")
    assert d.shadow_route == "ignore"
    assert d.tokens == 100
    assert "measurement stage" in d.reason
    [rec] = ledger.evaluations("training-watch", since=0)
    assert rec.shadow_route == "ignore" and rec.final_route == "agent" and not rec.fallback
    assert len(provider.calls) == 1
    req = provider.calls[0]
    assert req.feature == "training-watch"
    assert json.loads(req.context_json) == CTX
    assert set(req.questions) == {"material", "route"}


def test_decision_is_json_serializable(make_gate: GateFactory) -> None:
    d = make_gate().evaluate(CTX, questions())
    data = json.loads(json.dumps(d.to_json()))
    assert data["answers"]["route"]["value"] == "ignore"
    assert data["rollout"] is None


# -- failures resolve to the top route -------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ProviderUnavailable("API key is not set"), "provider_unavailable"),
        (RuntimeError("HTTP 500"), "provider_error"),
        (TimeoutError(), "timeout"),
    ],
)
def test_provider_failures_fall_back(
    make_gate: GateFactory, clock: FakeClock, exc: BaseException, status: str
) -> None:
    gate = open_gate(make_gate, clock, provider=ScriptedProvider(raises=exc))
    d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.status, d.shadow_route) == ("agent", False, status, None)
    assert all(a.status == status for a in d.answers.values())
    assert d.answers["route"].value == "agent"


def test_slow_provider_times_out(make_gate: GateFactory) -> None:
    gate = make_gate(provider=ScriptedProvider(answers=answers(), delay=0.5), timeout_s=0.05)
    d = gate.evaluate(CTX, questions())
    assert (d.status, d.route) == ("timeout", "agent")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "ignore",
        ProviderResponse(answers=["x"]),  # type: ignore[arg-type]
        ProviderResponse(answers={}, tokens=-5),
    ],
)
def test_malformed_provider_return(make_gate: GateFactory, bad: object) -> None:
    prov = ScriptedProvider(script=lambda _req: bad)  # type: ignore[arg-type,return-value]
    d = make_gate(provider=prov).evaluate(CTX, questions())
    assert d.route == "agent"
    assert d.tokens == 0
    assert d.status in ("provider_error", "partial")


def test_out_of_schema_route_answer_is_partial_and_falls_back(make_gate: GateFactory) -> None:
    prov = ScriptedProvider(answers={"material": {"p_yes": 0.01}, "route": {"probs": {"x": 1}}})
    d = make_gate(provider=prov).evaluate(CTX, questions())
    assert d.status == "partial"
    assert d.answers["route"].status == "out_of_schema"
    assert d.shadow_route == "agent"


# -- reuse and budget ------------------------------------------------------------------


def test_reuse_skips_the_provider_and_ignores_volatile_keys(
    make_gate: GateFactory, provider: ScriptedProvider, clock: FakeClock
) -> None:
    gate = make_gate()
    first = gate.evaluate(CTX, questions())
    clock.advance(60)
    second = gate.evaluate({**CTX, "ts": "2026-10-24T08:05:00Z"}, questions())
    assert len(provider.calls) == 1
    assert second.status == "reused" and second.tokens == 0
    assert second.shadow_route == first.shadow_route
    assert all(a.status == "reused" for a in second.answers.values())


def test_reuse_is_re_resolved_under_current_config(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    make_gate().evaluate(CTX, questions())
    clock.advance(60)
    d = make_gate(allowlist=frozenset({"persist"})).evaluate(CTX, questions())
    assert d.status == "reused"
    assert d.shadow_route == "ask"


def test_reuse_expires(
    make_gate: GateFactory, provider: ScriptedProvider, clock: FakeClock
) -> None:
    gate = make_gate()
    gate.evaluate(CTX, questions())
    clock.advance(901)
    assert gate.evaluate(CTX, questions()).status == "ok"
    assert len(provider.calls) == 2


def test_reuse_invalidated_by_question_or_model_change(
    make_gate: GateFactory, provider: ScriptedProvider, clock: FakeClock
) -> None:
    gate = make_gate()
    gate.evaluate(CTX, questions())
    gate.evaluate(CTX, questions(prompt="Which route now?"))
    assert len(provider.calls) == 2
    other = ScriptedProvider(answers=answers(), version="scripted-2")
    make_gate(provider=other).evaluate(CTX, questions())
    assert len(other.calls) == 1


def test_daily_token_budget(
    make_gate: GateFactory, provider: ScriptedProvider, clock: FakeClock
) -> None:
    gate = make_gate(daily_token_budget=150)
    assert gate.evaluate({"n": 1}, questions()).status == "ok"
    assert gate.evaluate({"n": 2}, questions()).status == "ok"
    d = gate.evaluate({"n": 3}, questions())
    assert (d.status, d.route) == ("budget_exhausted", "agent")
    assert len(provider.calls) == 2
    clock.advance(86_401)
    assert gate.evaluate({"n": 4}, questions()).status == "ok"


# -- policy once suppression is on -----------------------------------------------------


def test_closed_rollout_gate_keeps_the_old_behaviour(make_gate: GateFactory) -> None:
    d = make_gate(suppression_enabled=True).evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.stage) == ("agent", False, "B")
    assert d.rollout is not None and not d.rollout.open
    assert "rollout gate closed" in d.reason


def test_open_gate_lets_an_allowlisted_cheap_route_skip(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    d = open_gate(make_gate, clock).evaluate(CTX, questions())
    assert d.rollout is not None and d.rollout.open
    assert (d.route, d.skip_agent, d.status) == ("ignore", True, "ok")


@pytest.mark.parametrize(
    ("ans", "route", "why"),
    [
        (answers("ask"), "ask", "model chose"),
        (answers("defer"), "defer", "model chose"),
        (answers("ignore", p=0.85), "ask", "confidence"),
        (answers("ignore", p_material=0.7), "ask", "materiality"),
        ({"route": answers()["route"], "material": {"p_yes": 2}}, "ask", "materiality"),
    ],
)
def test_policy_raises_but_never_lowers(
    make_gate: GateFactory, clock: FakeClock, ans: dict[str, Any], route: str, why: str
) -> None:
    gate = open_gate(make_gate, clock, provider=ScriptedProvider(answers=ans))
    d = gate.evaluate(CTX, questions())
    assert d.route == route
    assert not d.skip_agent
    assert why in d.reason


def test_signal_floor(make_gate: GateFactory, clock: FakeClock) -> None:
    floor = SignalFloor(weights={"sleep": 1.0, "soreness": 1.0}, threshold=0.8, raise_to="defer")
    gate = open_gate(make_gate, clock, signal_floor=floor)
    assert gate.evaluate(CTX, questions(), signals={"sleep": 0.6, "soreness": 0.6}).route == "defer"
    weak = gate.evaluate({**CTX, "n": 2}, questions(), signals={"sleep": 0.3, "soreness": 0.2})
    assert weak.route == "ignore"


def test_signal_floor_overflow_raises_the_route(make_gate: GateFactory, clock: FakeClock) -> None:
    # Finite weights and signals can still overflow: inf + -inf is NaN, and
    # NaN >= threshold is False. A non-finite score must count as over.
    floor = SignalFloor(weights={"a": 1e308, "b": -1e308}, threshold=0.8, raise_to="defer")
    gate = open_gate(make_gate, clock, signal_floor=floor)
    d = gate.evaluate(CTX, questions(), signals={"a": 10.0, "b": 10.0})
    assert d.route == "defer"
    assert "signal floor" in d.reason


def test_signals_without_floor_or_unknown_are_bugs(make_gate: GateFactory) -> None:
    with pytest.raises(ConfigError, match="no signal_floor"):
        make_gate().evaluate(CTX, questions(), signals={"x": 1.0})
    floor = SignalFloor(weights={"sleep": 1.0}, threshold=1.0, raise_to="agent")
    with pytest.raises(ConfigError, match="unknown signals"):
        make_gate(signal_floor=floor).evaluate(CTX, questions(), signals={"x": 1.0})


# -- escalation ------------------------------------------------------------------------


def test_escalation_bypasses_the_model_and_labels_misses(
    make_gate: GateFactory, provider: ScriptedProvider, clock: FakeClock, ledger: Ledger
) -> None:
    def chest_pain(ctx: Any) -> str | None:
        return "high-risk symptom" if ctx.get("symptom") == "chest_pain" else None

    gate = open_gate(make_gate, clock, escalation_rules=(chest_pain,))
    clock.advance(60)
    skipped = gate.evaluate(CTX, questions())
    assert skipped.skip_agent
    calls = len(provider.calls)
    clock.advance(60)
    d = gate.evaluate({"symptom": "chest_pain"}, questions())
    assert (d.route, d.status, d.skip_agent, d.shadow_route) == ("agent", "escalated", False, None)
    assert "labelled" in d.reason
    assert len(provider.calls) == calls
    labels = {r.eval_id: r.label for r in ledger.evaluations("training-watch", since=0)}
    assert labels[skipped.eval_id] == "agent"
    # The miss now closes the rollout gate.
    clock.advance(60)
    after = gate.evaluate({**CTX, "n": 99}, questions())
    assert not after.skip_agent
    assert after.rollout is not None and "false negatives" in " ".join(after.rollout.reasons)


def test_explicit_escalate_argument(make_gate: GateFactory, provider: ScriptedProvider) -> None:
    d = make_gate().evaluate(CTX, questions(), escalate="constraint breach")
    assert d.status == "escalated" and d.reason.endswith("constraint breach")
    assert provider.calls == []


def test_escalations_do_not_count_as_fallbacks(make_gate: GateFactory, ledger: Ledger) -> None:
    make_gate().evaluate(CTX, questions(), escalate="x")
    [rec] = ledger.evaluations("training-watch", since=0)
    assert not rec.fallback


# -- integration bugs raise ------------------------------------------------------------


def test_boundary_errors_raise_before_the_provider(
    make_gate: GateFactory, provider: ScriptedProvider
) -> None:
    with pytest.raises(BoundaryError):
        make_gate().evaluate({"api_key": "x"}, questions())
    assert provider.calls == []


def test_label_rejects_unknown_route(make_gate: GateFactory) -> None:
    gate = make_gate()
    d = gate.evaluate(CTX, questions())
    with pytest.raises(ConfigError):
        gate.label(d.eval_id, "teleport")


@pytest.mark.parametrize(
    "qs",
    [
        {"material": questions()["material"]},
        {**questions(), "route": BooleanQuestion("route?", fallback=True)},
        {**questions(), "route": ChoiceQuestion("r", options=ROUTES[::-1], fallback="agent")},
        {**questions(), "route": ChoiceQuestion("r", options=ROUTES, fallback="ignore")},
        {**questions(), "material": ScoreQuestion("m", fallback=1.0)},
    ],
)
def test_question_set_must_match_config(make_gate: GateFactory, qs: Any) -> None:
    with pytest.raises(ConfigError):
        make_gate().evaluate(CTX, qs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature": ""},
        {"routes": ("agent",)},
        {"routes": ("a", "a", "agent")},
        {"skippable": frozenset({"agent"})},
        {"skippable": frozenset({"nope"})},
        {"allowlist": frozenset({"agent"})},
        {"denied_route": "nope"},
        {"denied_route": "ignore"},
        {"signal_floor": SignalFloor({}, 1.0, "nope")},
        {"min_route_confidence": 1.5},
        {"timeout_s": 0},
        {"daily_token_budget": -1},
        {"rollout_cache_s": -1},
    ],
)
def test_config_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        base_config(**overrides)


def test_stages_and_denied_route_default() -> None:
    cfg = base_config()
    assert cfg.stage == "A"
    assert cfg.effective_denied_route == "ask"
    assert replace(cfg, suppression_enabled=True).stage == "B"
    wide = replace(cfg, suppression_enabled=True, allowlist=SKIPPABLE | {"defer"})
    assert wide.stage == "C"
    assert replace(cfg, denied_route="defer").effective_denied_route == "defer"


def test_stage_c_lets_the_model_choose_a_wider_cheap_route(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    seed_evidence(make_gate(), clock, 4)
    gate = make_gate(
        suppression_enabled=True,
        allowlist=SKIPPABLE | {"ask"},
        denied_route="defer",
        provider=ScriptedProvider(answers=answers("ask")),
    )
    d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.stage) == ("ask", False, "C")


def test_provider_receives_a_copy_of_questions(make_gate: GateFactory) -> None:
    seen: list[ProviderRequest] = []

    def script(req: ProviderRequest) -> ProviderResponse:
        seen.append(req)
        return ProviderResponse(answers(), 7)

    make_gate(provider=ScriptedProvider(script=script)).evaluate(CTX, questions())
    assert seen[0].questions == questions()


def test_gate_as_context_manager(ledger: Ledger, provider: ScriptedProvider) -> None:
    with Gate(base_config(), provider, ledger) as gate:
        assert gate.evaluate(CTX, questions()).status == "ok"


def test_without_materiality_question(make_gate: GateFactory, clock: FakeClock) -> None:
    gate = open_gate(make_gate, clock, materiality_question=None)
    d = gate.evaluate(CTX, {"route": questions()["route"]})
    assert d.skip_agent


# -- empty allowlist (stage A without an allowlist yet) --------------------------------


def test_empty_allowlist_measures_shadow_as_if_skippable_were_allowed(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    gate = make_gate(allowlist=frozenset())
    d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.stage) == ("agent", False, "A")
    assert d.shadow_route == "ignore"  # not raised to "ask" by the empty allowlist
    # ...so stage-A traffic is usable calibration data and the gate can open.
    seed_evidence(gate, clock, 4)
    report = gate.rollout_status()
    assert report.open, report.reasons
    assert report.would_suppress >= 2


def test_empty_allowlist_with_suppression_on_still_never_skips(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    seed_evidence(make_gate(allowlist=frozenset()), clock, 4)
    gate = make_gate(allowlist=frozenset(), suppression_enabled=True)
    assert gate.rollout_status().open
    d = gate.evaluate(CTX, questions())
    assert (d.route, d.skip_agent, d.stage) == ("agent", False, "A")
    assert d.shadow_route == "ignore"
    assert "empty allowlist" in d.reason


def test_empty_allowlist_shadow_still_respects_confidence_and_materiality(
    make_gate: GateFactory,
) -> None:
    low = ScriptedProvider(answers=answers("ignore", p=0.6))
    assert (
        make_gate(allowlist=frozenset(), provider=low).evaluate(CTX, questions()).shadow_route
        == "ask"
    )
    material = ScriptedProvider(answers=answers("ignore", p_material=0.9))
    d = make_gate(allowlist=frozenset(), provider=material).evaluate(CTX, questions())
    assert d.shadow_route == "ask"


def test_partial_skips_when_only_an_extra_question_fell_back(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    gate = open_gate(make_gate, clock)
    qs = {**questions(), "extra": BooleanQuestion("Anything else?", fallback=True)}
    d = gate.evaluate(CTX, qs)  # the scripted answers have no "extra"
    assert d.status == "partial"
    assert d.route == "ignore" and d.skip_agent


def test_partial_with_every_answer_out_of_schema(make_gate: GateFactory, clock: FakeClock) -> None:
    gate = open_gate(make_gate, clock, provider=ScriptedProvider(answers={"x": 1}))
    d = gate.evaluate(CTX, questions())
    assert d.status == "partial"
    assert d.route == "agent" and not d.skip_agent
