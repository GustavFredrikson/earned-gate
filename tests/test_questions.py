from __future__ import annotations

import math

import pytest

from earned_gate import BooleanQuestion, ChoiceQuestion, ConfigError, ScoreQuestion
from earned_gate.questions import Answer, resolve

B = BooleanQuestion("Is it material?", fallback=True)
C = ChoiceQuestion("Which?", options=("a", "b", "c"), fallback="c")
S = ScoreQuestion("How urgent?", fallback=1.0)


def test_boolean_confident_no_is_not_a_weak_answer() -> None:
    a = resolve("m", B, {"p_yes": 0.02})
    assert a.value is False
    assert a.confidence == pytest.approx(0.96)
    assert a.status == "ok"


def test_boolean_fifty_fifty_has_zero_confidence() -> None:
    a = resolve("m", B, {"p_yes": 0.5})
    assert a.value is True
    assert a.confidence == 0.0


def test_choice_picks_max_and_reports_its_probability() -> None:
    a = resolve("r", C, {"probs": {"a": 0.1, "b": 0.85, "c": 0.05}})
    assert (a.value, a.confidence, a.status) == ("b", 0.85, "ok")


def test_score_maps_levels_to_unit_interval() -> None:
    a = resolve("s", S, {"probs": [0.0, 0.1, 0.1, 0.7, 0.1]})
    assert a.value == 0.75
    assert a.confidence == 0.7


@pytest.mark.parametrize(
    ("question", "raw"),
    [
        (B, None),
        (B, "yes"),
        (B, {}),
        (B, {"p_yes": 1.2}),
        (B, {"p_yes": -0.1}),
        (B, {"p_yes": math.nan}),
        (B, {"p_yes": True}),
        (B, {"p_yes": "0.4"}),
        (C, {"probs": {"a": 0.5, "b": 0.5}}),
        (C, {"probs": {"a": 0.5, "b": 0.5, "c": 0.5}}),
        (C, {"probs": {"a": 0.5, "b": 0.5, "d": 0.0}}),
        (C, {"probs": {"a": 0.5, "b": "x", "c": 0.5}}),
        (C, {"probs": [0.2, 0.3, 0.5]}),
        (S, {"probs": [0.2, 0.2, 0.2, 0.4]}),
        (S, {"probs": [0.2, 0.2, 0.2, 0.2, 0.3]}),
        (S, {"probs": [0.2, 0.2, 0.2, 0.2, "0.2"]}),
        (S, {"probs": {"1": 1.0}}),
    ],
)
def test_out_of_schema_resolves_to_fallback(question: object, raw: object) -> None:
    a = resolve("q", question, raw)  # type: ignore[arg-type]
    assert a.status == "out_of_schema"
    assert a.value == question.fallback  # type: ignore[attr-defined]
    assert a.confidence == 0.0
    assert a.is_fallback


def test_low_confidence_resolves_to_fallback() -> None:
    q = ChoiceQuestion("Which?", options=("a", "b"), fallback="b", min_confidence=0.9)
    a = resolve("r", q, {"probs": {"a": 0.8, "b": 0.2}})
    assert (a.value, a.status) == ("b", "low_confidence")


@pytest.mark.parametrize(
    "build",
    [
        lambda: BooleanQuestion("  ", fallback=True),
        lambda: BooleanQuestion("q", fallback=True, min_confidence=1.5),
        lambda: ChoiceQuestion("q", options=("a",), fallback="a"),
        lambda: ChoiceQuestion("q", options=("a", "a"), fallback="a"),
        lambda: ChoiceQuestion("q", options=("a", "b"), fallback="z"),
        lambda: ChoiceQuestion("", options=("a", "b"), fallback="a"),
        lambda: ScoreQuestion("q", fallback=2.0),
        lambda: ScoreQuestion("", fallback=0.0),
    ],
)
def test_question_config_errors(build: object) -> None:
    with pytest.raises(ConfigError):
        build()  # type: ignore[operator]


def test_answer_json_roundtrip() -> None:
    a = resolve("r", C, {"probs": {"a": 0.1, "b": 0.85, "c": 0.05}})
    assert Answer.from_json(a.to_json()) == a
    f = resolve("r", C, None)
    assert Answer.from_json(f.to_json()) == f


def test_specs_are_stable() -> None:
    assert B.spec() == {"kind": "boolean", "prompt": "Is it material?"}
    assert C.spec()["options"] == ["a", "b", "c"]
    assert S.spec()["levels"] == 5
