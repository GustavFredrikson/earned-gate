"""Failure paths at the edges: exceptions that must not escape, numbers, the ledger."""

from __future__ import annotations

import math
import types
from collections import ChainMap
from typing import Any

import pytest
from conftest import ROUTES, GateFactory, answers, base_config, questions, route_probs

from earned_gate import BoundaryError, ConfigError, Gate, Ledger, validate_context
from earned_gate.testing import FakeClock, ScriptedProvider

CTX = {"event": "hrv_dip", "delta": -4}

# -- nothing but BoundaryError / ConfigError escapes evaluate() ------------------------


def test_non_dict_mappings_are_valid_contexts(make_gate: GateFactory) -> None:
    proxy = types.MappingProxyType({"event": "hrv_dip", "nested": ChainMap({"a": 1}, {"b": 2})})
    expected = validate_context({"event": "hrv_dip", "nested": {"a": 1, "b": 2}})
    assert validate_context(proxy) == expected
    d = make_gate().evaluate(proxy, questions())
    assert d.status == "ok"


def test_huge_int_in_context_is_a_boundary_error() -> None:
    with pytest.raises(BoundaryError):
        validate_context({"n": 10**5000}, max_bytes=10**9)


def test_huge_int_signal_is_a_boundary_error(make_gate: GateFactory) -> None:
    from earned_gate import SignalFloor

    gate = make_gate(signal_floor=SignalFloor({"x": 1.0}, threshold=1.0, raise_to="defer"))
    with pytest.raises(BoundaryError, match=r"signals\.x"):
        gate.evaluate(CTX, questions(), signals={"x": 10**400})


@pytest.mark.parametrize(
    "raw",
    [
        {"material": {"p_yes": 10**400}, "route": route_probs("ignore")},
        {"material": {"p_yes": 0.03}, "route": {"probs": {"ignore": 10**400}}},
        {
            "material": {"p_yes": 0.03},
            "route": {
                "probs": dict.fromkeys(("ignore", "persist", "ask", "defer", "agent"), 10**400)
            },
        },
    ],
)
def test_huge_int_provider_answers_fall_back(make_gate: GateFactory, raw: dict[str, Any]) -> None:
    d = make_gate(provider=ScriptedProvider(answers=raw)).evaluate(CTX, questions())
    assert d.status == "partial" and not d.skip_agent
    assert any(a.status == "out_of_schema" for a in d.answers.values())


def test_evaluate_after_close_is_a_config_error(make_gate: GateFactory) -> None:
    gate = make_gate()
    gate.close()
    with pytest.raises(ConfigError, match="closed"):
        gate.evaluate(CTX, questions())


@pytest.mark.parametrize("field", ["timeout_s", "reuse_ttl_s", "rollout_cache_s"])
@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_non_finite_durations_are_config_errors(field: str, value: float) -> None:
    with pytest.raises(ConfigError):
        base_config(**{field: value})


def test_pool_shut_down_under_a_live_gate_falls_back(make_gate: GateFactory) -> None:
    gate = make_gate()
    gate._pool.shutdown(wait=False)  # simulates close() racing an evaluate()
    d = gate.evaluate(CTX, questions())
    assert (d.status, d.skip_agent) == ("provider_error", False)


def test_resolve_raising_is_contained(
    make_gate: GateFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import earned_gate.gate as gate_mod

    def boom(*args: object) -> object:
        raise ArithmeticError("unexpected")

    monkeypatch.setattr(gate_mod, "resolve", boom)
    d = make_gate().evaluate(CTX, questions())
    assert d.status == "partial" and not d.skip_agent
    assert all(a.status == "out_of_schema" for a in d.answers.values())


# -- an int clock must not break verify() ----------------------------------------------


def test_int_clock_ledger_verifies(tmp_path: Any) -> None:
    from earned_gate import Gate, Ledger

    ticks = iter(range(1_800_000_000, 1_800_100_000, 7))
    path = tmp_path / "int-clock.db"
    with (
        Ledger(path, clock=lambda: next(ticks)) as led,
        Gate(base_config(), ScriptedProvider(answers=answers()), led) as gate,
    ):
        d = gate.evaluate(CTX, questions())
        gate.label(d.eval_id, "ignore")
        gate.evaluate(CTX, questions(), escalate="x")
        assert led.verify() == 4
    with Ledger(path, create=False) as reopened:
        assert reopened.verify(expected_head=reopened.head()) == 4


# -- explicit ledger operations raise LedgerError, never raw sqlite3 errors ------------


def _ledger_file(tmp_path: Any) -> tuple[Any, str]:
    from earned_gate import Gate, Ledger

    path = tmp_path / "gate.db"
    with (
        Ledger(path) as led,
        Gate(base_config(), ScriptedProvider(answers=answers()), led) as gate,
    ):
        eval_id = gate.evaluate(CTX, questions()).eval_id
    return path, eval_id


def test_cli_label_on_read_only_ledger_exits_2(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    from earned_gate.cli import main

    path, eval_id = _ledger_file(tmp_path)
    os.chmod(path, 0o444)
    try:
        assert main(["label", str(path), eval_id, "agent"]) == 2
        assert "LedgerError" in capsys.readouterr().err
    finally:
        os.chmod(path, 0o644)


def test_ledger_methods_on_a_closed_ledger_raise_ledger_error(tmp_path: Any) -> None:
    from earned_gate import Ledger, LedgerError

    path, eval_id = _ledger_file(tmp_path)
    led = Ledger(path, create=False)
    led.close()
    calls = [
        lambda: led.verify(),
        lambda: led.head(),
        lambda: led.label(eval_id, "agent"),
        lambda: led.evaluations("training-watch", since=0),
        lambda: led.recent_fallbacks("training-watch", since=0, limit=5),
        lambda: led.find_reusable("training-watch", "h", "m", since=0),
        lambda: led.tokens_since("training-watch", 0),
        lambda: led.change_token(),
        lambda: led.label_missed_suppressions(
            "training-watch", since=0, skippable={"ignore"}, actual_route="agent"
        ),
    ]
    for call in calls:
        with pytest.raises(LedgerError):
            call()


def test_creating_a_ledger_at_a_directory_raises_ledger_error(tmp_path: Any) -> None:
    from earned_gate import Ledger, LedgerError

    with pytest.raises(LedgerError):
        Ledger(tmp_path)


def test_corrupt_payload_raises_ledger_error(tmp_path: Any) -> None:
    import sqlite3

    from earned_gate import Ledger, LedgerError

    path, _ = _ledger_file(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE events SET payload = 'not json'")
    conn.commit()
    conn.close()
    with Ledger(path, create=False) as led, pytest.raises(LedgerError, match="unreadable"):
        led.evaluations("training-watch", since=0)


# -- non-finite or mistyped policy numbers are config errors ---------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"false_negative_weight": math.nan},
        {"false_negative_weight": math.inf},
        {"min_labelled": math.nan},
        {"min_labelled": 30.5},
        {"min_labelled": True},
        {"min_would_suppress": math.nan},
        {"max_false_negatives": math.inf},
        {"max_false_negatives": 0.5},
        {"degraded_window": 2.5},
        {"window_s": math.inf},
        {"window_s": math.nan},
        {"max_miss_score": math.nan},
        {"max_fallback_rate": math.nan},
        {"max_recent_fallback_rate": math.nan},
    ],
)
def test_rollout_policy_rejects_non_finite_or_mistyped_numbers(overrides: dict[str, Any]) -> None:
    from earned_gate import RolloutPolicy

    with pytest.raises(ConfigError):
        RolloutPolicy(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"daily_token_budget": math.nan},
        {"daily_token_budget": math.inf},
        {"daily_token_budget": 10.5},
        {"daily_token_budget": True},
        {"max_context_bytes": math.nan},
        {"max_context_bytes": 100.5},
        {"min_route_confidence": math.nan},
    ],
)
def test_gate_config_rejects_non_finite_or_mistyped_numbers(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        base_config(**overrides)


@pytest.mark.parametrize("value", ["30", None, 10**400])
def test_rollout_policy_rejects_non_numbers_and_overflow(value: object) -> None:
    from earned_gate import RolloutPolicy

    with pytest.raises(ConfigError):
        RolloutPolicy(window_s=value)  # type: ignore[arg-type]


def test_cli_rollout_with_nan_window_exits_2(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from earned_gate.cli import main

    path, _ = _ledger_file(tmp_path)
    argv = ["rollout", str(path), "--feature", "training-watch", "--model-version", "m"]
    argv += ["--routes", "ignore,agent", "--skippable", "ignore", "--window-days", "nan"]
    assert main(argv) == 2
    assert "ConfigError" in capsys.readouterr().err


# -- the hash chain detects naive edits, not a determined rewrite ----------------------


def test_a_full_rewrite_passes_verify_but_not_an_external_head(tmp_path: Any) -> None:
    import sqlite3

    from earned_gate import Ledger, LedgerIntegrityError
    from earned_gate.ledger import GENESIS, _row_hash

    path, _ = _ledger_file(tmp_path)
    with Ledger(path, create=False) as led:
        stored_head = led.head()
    # Anyone who can write the file can edit a row and recompute every hash.
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT seq, ts, kind, feature, model_version, eval_id, payload FROM events ORDER BY seq"
    ).fetchall()
    prev = GENESIS
    for seq, ts, kind, feature, mv, eval_id, payload in rows:
        payload = payload.replace('"tokens":100', '"tokens":0')
        digest = _row_hash(prev, ts, kind, feature, mv, eval_id, payload)
        conn.execute(
            "UPDATE events SET payload = ?, prev_hash = ?, hash = ? WHERE seq = ?",
            (payload, prev, digest, seq),
        )
        prev = digest
    conn.commit()
    conn.close()
    with Ledger(path, create=False) as led:
        assert led.verify() == len(rows)  # the chain alone cannot tell
        with pytest.raises(LedgerIntegrityError):
            led.verify(expected_head=stored_head)


def test_docs_do_not_overclaim_tamper_evidence() -> None:
    from pathlib import Path

    import earned_gate.ledger as ledger_mod

    root = Path(__file__).resolve().parent.parent
    doc = ledger_mod.__doc__ or ""
    assert "editing or deleting history is\ndetectable" not in doc
    for text in (doc, (root / "SECURITY.md").read_text(), (root / "docs/design.md").read_text()):
        assert "not tamper-proof" in text
        assert "head()" in text


@pytest.mark.parametrize("ctx", [{"f": "\udcff"}, {"\udcff": 1}, {"a": ["ok", {"b": "x\ud800"}]}])
def test_lone_surrogates_in_context_are_boundary_errors(
    make_gate: GateFactory, ctx: dict[str, Any]
) -> None:
    with pytest.raises(BoundaryError):
        make_gate().evaluate(ctx, questions())


def test_lone_surrogate_in_a_question_prompt_is_a_config_error(make_gate: GateFactory) -> None:
    from earned_gate import BooleanQuestion

    qs = {**questions(), "x": BooleanQuestion("Bad \udcff prompt?", fallback=True)}
    with pytest.raises(ConfigError, match="questions\\['x'\\]"):
        make_gate().evaluate(CTX, qs)


class _Truthless:
    def __bool__(self) -> bool:
        raise RuntimeError("no truth value")


def test_rule_result_whose_bool_raises_is_a_rule_error(make_gate: GateFactory) -> None:
    def odd_rule(ctx: Any) -> Any:
        return _Truthless()

    d = make_gate(escalation_rules=(odd_rule,)).evaluate(CTX, questions())
    assert (d.status, d.skip_agent) == ("rule_error", False)


# -- reuse re-resolves stored answers under the current questions ----------------------


def test_reuse_rechecks_the_current_min_confidence(ledger: Ledger, clock: FakeClock) -> None:
    from earned_gate import BooleanQuestion

    prov = ScriptedProvider(answers=answers("ignore", p_material=0.2))  # "no", confidence 0.6
    loose = {
        **questions(),
        "material": BooleanQuestion(
            "Could this change plausibly alter the next coaching decision?",
            fallback=True,
            min_confidence=0.5,
        ),
    }
    strict = {
        **loose,
        "material": BooleanQuestion(
            "Could this change plausibly alter the next coaching decision?",
            fallback=True,
            min_confidence=0.9,
        ),
    }
    with Gate(base_config(), prov, ledger) as gate:
        first = gate.evaluate(CTX, loose)
        assert first.status == "ok" and first.shadow_route == "ignore"
        clock.advance(60)
        again = gate.evaluate(CTX, strict)  # same spec hash, stricter threshold
    assert len(prov.calls) == 1  # reused, not re-asked
    assert again.answers["material"].status == "low_confidence"
    assert again.shadow_route == "ask"  # materiality fallback raises the route
    assert again.status == "partial" and not again.skip_agent


def test_reuse_with_unchanged_questions_stays_reused(ledger: Ledger, clock: FakeClock) -> None:
    prov = ScriptedProvider(answers=answers())
    with Gate(base_config(), prov, ledger) as gate:
        gate.evaluate(CTX, questions())
        clock.advance(60)
        d = gate.evaluate(CTX, questions())
    assert d.status == "reused" and len(prov.calls) == 1
    assert all(a.status == "reused" for a in d.answers.values())


# -- the ledger stores only validated answer fields ------------------------------------


def test_ledger_stores_only_validated_answer_fields(ledger: Ledger) -> None:
    raw = {
        "material": {"p_yes": 0.03, "reasoning": "the user said hrv_delta=-4", "x": object()},
        "route": {**route_probs("ignore"), "explanation": "echoed context"},
    }
    with Gate(base_config(), ScriptedProvider(answers=raw), ledger) as gate:
        d = gate.evaluate(CTX, questions())
    assert (d.status, d.recorded) == ("ok", True)  # extra non-JSON junk no longer breaks writes
    [rec] = ledger.evaluations("training-watch", since=0)
    stored = {a.question_id: a.raw for a in rec.answers}
    assert stored["material"] == {"p_yes": 0.03}
    assert set(stored["route"] or {}) == {"probs"}
    assert "echoed" not in str(stored) and "hrv_delta" not in str(stored)


# -- a failed COMMIT must not leave the connection inside a transaction ----------------


class _CommitFailsOnce:
    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.failed = False

    def execute(self, sql: str, *args: Any) -> Any:
        if sql == "COMMIT" and not self.failed:
            self.failed = True
            import sqlite3

            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def test_failed_commit_rolls_back_and_later_writes_work(tmp_path: Any) -> None:
    from earned_gate import LedgerError

    path, eval_id = _ledger_file(tmp_path)
    with Ledger(path, create=False) as led:
        real = led._conn
        led._conn = _CommitFailsOnce(real)  # type: ignore[assignment]
        with pytest.raises(LedgerError, match="disk I/O"):
            led.label(eval_id, "agent")
        assert not real.in_transaction
        led.label(eval_id, "ignore")  # the next write must not be wedged
        assert [r.label for r in led.evaluations("training-watch", since=0)] == ["ignore"]


# -- GateConfig: wrong types are a ConfigError, not a TypeError or a silent accept -----


@pytest.mark.parametrize(
    "overrides",
    [
        {"routes": "ignore,agent"},
        {"routes": ("ignore", 2)},
        {"skippable": "ignore"},
        {"allowlist": "ignore"},
        {"skippable": frozenset({1})},
        {"feature": 7},
        {"route_question": None},
        {"materiality_question": 3},
        {"denied_route": 1},
        {"suppression_enabled": "yes"},
        {"volatile_keys": "ts"},
        {"allow_keys": "token_count"},
        {"escalation_rules": lambda ctx: None},
        {"rollout": {"min_labelled": 4}},
        {"signal_floor": {"weights": {}, "threshold": 1.0, "raise_to": "agent"}},
    ],
    ids=lambda o: next(iter(o)),
)
def test_gate_config_rejects_wrong_types(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        base_config(**overrides)


@pytest.mark.parametrize(
    "floor",
    [
        {"weights": {"a": "1"}, "threshold": 1.0},
        {"weights": {"a": True}, "threshold": 1.0},
        {"weights": {1: 1.0}, "threshold": 1.0},
        {"weights": {"a": 1.0}, "threshold": "1"},
        {"weights": [("a", 1.0)], "threshold": 1.0},
    ],
)
def test_signal_floor_rejects_wrong_types(floor: dict[str, Any]) -> None:
    from earned_gate import SignalFloor

    with pytest.raises(ConfigError):
        base_config(signal_floor=SignalFloor(raise_to="agent", **floor))


def test_gate_config_accepts_lists_and_sets_for_collections() -> None:
    cfg = base_config(
        routes=["ignore", "ask", "agent"],
        skippable={"ignore"},
        allowlist=["ignore"],
    )
    assert cfg.routes == ("ignore", "ask", "agent")
    assert cfg.skippable == frozenset({"ignore"})
    assert cfg.allowlist == frozenset({"ignore"})


# -- shared references: a small object graph can expand to millions of nodes -----------


def test_shared_reference_blowup_stops_at_the_node_cap() -> None:
    node: Any = [1]
    for _ in range(17):  # 2**17 leaves through shared references, depth 18
        node = [node, node]
    with pytest.raises(BoundaryError, match="more than 8192 values"):
        validate_context({"x": node})


def test_node_cap_never_rejects_a_context_that_fits() -> None:
    # Every value costs at least one byte of JSON, so the cap is max_bytes.
    ctx = {"x": [0] * 4000}
    assert len(validate_context(ctx)) <= 8192


# -- seeded fuzz: evaluate() raises only BoundaryError or ConfigError -------------------

_ODD_VALUES: list[Any] = [
    0,
    -1,
    10**5000,
    1.5,
    math.nan,
    math.inf,
    True,
    None,
    "",
    "\udcff",
    "x" * 9000,
    "ghp_" + "a" * 36,
    b"bytes",
    {1, 2},
    (1, 2),
    object(),
    types.MappingProxyType({"a": 1}),
    ChainMap({"b": 2}),
]
_ODD_KEYS: list[Any] = ["a", "token", "token_count", "\udcff", 1, None, ("t",), "x" * 200]


def _odd(rng: Any, depth: int = 0) -> Any:
    roll = rng.random()
    if depth < 3 and roll < 0.25:
        return {rng.choice(_ODD_KEYS): _odd(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    if depth < 3 and roll < 0.4:
        return [_odd(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return rng.choice(_ODD_VALUES)


def test_seeded_fuzz_only_two_exceptions_escape(make_gate: GateFactory) -> None:
    import random

    from earned_gate import SignalFloor

    rng = random.Random(20261024)  # noqa: S311 - reproducible test data, not crypto
    floor = SignalFloor(weights={"a": 1.0, "s": 1e308}, threshold=0.5, raise_to="agent")
    decided = 0
    with make_gate(signal_floor=floor) as gate:
        for i in range(600):
            context = {"i": i, "v": _odd(rng)} if rng.random() < 0.7 else _odd(rng)
            signals: Any = None
            if rng.random() < 0.5:
                signals = {"a": rng.choice(_ODD_VALUES), "s": rng.choice([0.5, 10.0, 1e10])}
            try:
                d = gate.evaluate(context, questions(), signals=signals)
            except (BoundaryError, ConfigError):
                continue
            decided += 1
            assert d.route in ROUTES
    assert decided > 50  # the fuzz must reach the provider path, not only the boundary
