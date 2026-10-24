"""Config holes that would fail open, and caller objects that must still fail safe."""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import SKIPPABLE, GateFactory, answers, base_config, questions, seed_evidence

from earned_gate import BoundaryError, ConfigError, Gate, Ledger, SignalFloor
from earned_gate.cli import main
from earned_gate.testing import FakeClock, ScriptedProvider

CTX = {"event": "hrv_dip", "delta": -4}

# -- a denied or raised-to route must not be skippable -------------------------


@pytest.mark.parametrize("denied", ["ignore", "persist"])
def test_denied_route_in_skippable_is_a_config_error(denied: str) -> None:
    with pytest.raises(ConfigError, match="denied_route"):
        base_config(allowlist=frozenset({"ignore", "persist"} - {denied}), denied_route=denied)


def test_signal_floor_raise_to_skippable_is_a_config_error() -> None:
    floor = SignalFloor(weights={"x": 1.0}, threshold=1.0, raise_to="persist")
    with pytest.raises(ConfigError, match="raise_to"):
        base_config(signal_floor=floor)


# -- escalation_lookback_s must be positive -------------------------------------


@pytest.mark.parametrize("value", [-1.0, -3600, 0, 0.0])
def test_non_positive_escalation_lookback_is_a_config_error(value: float) -> None:
    with pytest.raises(ConfigError, match="escalation_lookback_s"):
        base_config(escalation_lookback_s=value)


# -- wrong-typed rollout_cache_s is a ConfigError, not a TypeError ---------------


@pytest.mark.parametrize("value", ["60", None, [1]])
def test_rollout_cache_s_wrong_type_is_a_config_error(value: object) -> None:
    with pytest.raises(ConfigError, match="rollout_cache_s"):
        base_config(rollout_cache_s=value)


# -- stage C with the default denied_route changes no decision -------------------


def test_stage_c_with_default_denied_route_decides_like_stage_b(
    make_gate: GateFactory, clock: FakeClock
) -> None:
    seed_evidence(make_gate(), clock, 4)
    prov = ScriptedProvider(answers=answers("ask"))
    b = make_gate(suppression_enabled=True, provider=prov).evaluate(CTX, questions())
    c = make_gate(suppression_enabled=True, allowlist=SKIPPABLE | {"ask"}, provider=prov).evaluate(
        CTX, questions()
    )
    assert (b.stage, c.stage) == ("B", "C")
    assert (b.route, b.skip_agent) == (c.route, c.skip_agent) == ("ask", False)


def test_readme_does_not_overclaim_stage_c() -> None:
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert "pick cheaper non-agent routes too" not in text
    assert "only changes a decision when `denied_route`" in text


# -- hostile caller objects inside evaluate() ---------------------------------


class _BadStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("str failed")


def test_rule_raising_exception_with_broken_str_fails_safe(make_gate: GateFactory) -> None:
    def rule(ctx: Any) -> str | None:
        raise _BadStr

    d = make_gate(escalation_rules=(rule,)).evaluate(CTX, questions())
    assert (d.status, d.route, d.skip_agent) == ("rule_error", "agent", False)


class _Rule:
    def __call__(self, ctx: Any) -> str | None:
        raise ValueError("boom")

    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


def test_rule_without_qualname_and_broken_repr_fails_safe(make_gate: GateFactory) -> None:
    rule = _Rule()
    assert not hasattr(rule, "__qualname__")
    d = make_gate(escalation_rules=(rule,)).evaluate(CTX, questions())
    assert (d.status, d.route, d.skip_agent) == ("rule_error", "agent", False)


class _BadVersion(ScriptedProvider):
    @property
    def model_version(self) -> str:
        raise _BadStr


def test_model_version_raising_broken_str_fails_safe(ledger: Ledger) -> None:
    with Gate(base_config(), _BadVersion(answers=answers()), ledger) as g:
        d = g.evaluate(CTX, questions())
    assert (d.status, d.route, d.skip_agent) == ("provider_unavailable", "agent", False)


class _BadInt(int):
    def __float__(self) -> float:
        raise ValueError("no float")


def test_int_subclass_signal_with_broken_float_is_a_boundary_error(
    make_gate: GateFactory,
) -> None:
    gate = make_gate(signal_floor=SignalFloor({"x": 1.0}, threshold=1.0, raise_to="defer"))
    with pytest.raises(BoundaryError, match=r"signals\.x"):
        gate.evaluate(CTX, questions(), signals={"x": _BadInt(1)})


# -- unrecorded tokens only accumulate when a budget needs them ----------------


class _FailingWrites(Ledger):
    def record_evaluation(self, record: Any) -> None:
        raise sqlite3.OperationalError("database or disk is full")


def test_unrecorded_tokens_do_not_pile_up_without_a_budget(
    clock: FakeClock, provider: ScriptedProvider
) -> None:
    with _FailingWrites(":memory:", clock=clock) as led, Gate(base_config(), provider, led) as g:
        for i in range(5):
            d = g.evaluate({"event": "x", "n": i}, questions())
            assert d.status == "ledger_error" and d.tokens == 100
        assert g._unrecorded_tokens == []  # pyright: ignore[reportPrivateUsage]


# -- the CLI tells an integrity failure apart from a bad path -------------------


def test_cli_verify_exits_1_on_a_broken_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "gate.db"
    with Ledger(path) as led, Gate(base_config(), ScriptedProvider(answers=answers()), led) as g:
        g.evaluate(CTX, questions())
    conn = sqlite3.connect(path)
    conn.execute("UPDATE events SET payload = replace(payload, '\"tokens\":100', '\"tokens\":0')")
    conn.commit()
    conn.close()
    assert main(["verify", str(path)]) == 1
    assert "LedgerIntegrityError" in capsys.readouterr().err


# -- credentials embedded in a URL are rejected ---------------------------------


@pytest.mark.parametrize(
    "url", ["postgres://user:hunter2@db.internal:5432/app", "https://u:p@example.com/x"]
)
def test_url_with_credentials_is_a_boundary_error(make_gate: GateFactory, url: str) -> None:
    with pytest.raises(BoundaryError, match="credentials"):
        make_gate().evaluate({"dsn_hint": url}, questions())


def test_url_without_credentials_passes(make_gate: GateFactory) -> None:
    d = make_gate().evaluate({"link": "https://example.com/a:b@c"}, questions())
    assert d.status == "ok"


# -- the example adapter sends the validated encoding ---------------------------


def test_example_adapter_accepts_a_non_dict_mapping_context(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import openai_compatible

    content = {"material": {"p_yes": 0.02}, "route": {"probs": {"ignore": 0.96, "agent": 0.04}}}
    sent: list[dict[str, Any]] = []

    def transport(url: str, body: bytes, headers: Any, timeout: float) -> bytes:
        sent.append(json.loads(body))
        payload = {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {}}
        return json.dumps(payload).encode()

    prov = openai_compatible.ChatCompletionsProvider("m", api_key="k", transport=transport)
    ctx = types.MappingProxyType({"event": "hrv"})
    with Ledger(tmp_path / "a.db") as led, Gate(base_config(), prov, led) as gate:
        d = gate.evaluate(ctx, questions())
    assert d.status != "provider_error"
    user = json.loads(sent[0]["messages"][1]["content"])
    assert user["state"] == {"event": "hrv"}
