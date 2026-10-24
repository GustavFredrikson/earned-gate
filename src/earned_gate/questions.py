"""Typed questions and the answers they resolve to.

A question is bounded on purpose: a boolean, a choice from options the caller
supplies, or a score on a fixed five-level scale. There is no free-text answer
type, so nothing the provider returns can become generated prose in the caller.

Every question carries its own fallback, which should be whatever the feature
did before a model was involved, and its own minimum confidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .errors import ConfigError

SCORE_LEVELS = 5

AnswerStatus = Literal[
    "ok",
    "reused",
    "out_of_schema",
    "low_confidence",
    "provider_error",
    "provider_unavailable",
    "timeout",
    "budget_exhausted",
    "skipped",
]

#: Statuses where the answer came from the provider (now or earlier) and passed checks.
GOOD_STATUSES: frozenset[str] = frozenset({"ok", "reused"})


def _check_unit(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0) or math.isnan(value):
        raise ConfigError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class BooleanQuestion:
    """A yes/no question. The provider returns ``{"p_yes": float}``.

    Confidence is distance from 50/50 rescaled to [0, 1], so ``p_yes=0.02`` is a
    confident *no* (confidence 0.96), not a weak answer.
    """

    prompt: str
    fallback: bool
    min_confidence: float = 0.0
    kind: Literal["boolean"] = field(default="boolean", init=False)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ConfigError("question prompt must not be empty")
        _check_unit("min_confidence", self.min_confidence)

    def spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "prompt": self.prompt}


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one of ``options``. The provider returns ``{"probs": {option: p}}``.

    Confidence is the probability of the winning option.
    """

    prompt: str
    options: tuple[str, ...]
    fallback: str
    min_confidence: float = 0.0
    kind: Literal["choice"] = field(default="choice", init=False)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ConfigError("question prompt must not be empty")
        if len(self.options) < 2:
            raise ConfigError("a choice question needs at least two options")
        if len(set(self.options)) != len(self.options):
            raise ConfigError("choice options must be unique")
        if self.fallback not in self.options:
            raise ConfigError(f"fallback {self.fallback!r} is not one of the options")
        _check_unit("min_confidence", self.min_confidence)

    def spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "prompt": self.prompt, "options": list(self.options)}


@dataclass(frozen=True)
class ScoreQuestion:
    """An ordinal on a fixed five-level scale, normalized to [0, 1].

    The provider returns ``{"probs": [p1, p2, p3, p4, p5]}``. The value is the
    most likely level mapped to 0, 0.25, 0.5, 0.75 or 1.0, and confidence is
    that level's probability.
    """

    prompt: str
    fallback: float
    min_confidence: float = 0.0
    kind: Literal["score"] = field(default="score", init=False)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ConfigError("question prompt must not be empty")
        _check_unit("fallback", self.fallback)
        _check_unit("min_confidence", self.min_confidence)

    def spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "prompt": self.prompt, "levels": SCORE_LEVELS}


Question = BooleanQuestion | ChoiceQuestion | ScoreQuestion
AnswerValue = bool | str | float


@dataclass(frozen=True)
class Answer:
    """The resolved answer to one question.

    ``value`` is always usable: when ``status`` is not ``ok``/``reused`` it is
    the question's fallback, and ``confidence`` is 0.
    """

    question_id: str
    value: AnswerValue
    confidence: float
    status: AnswerStatus
    raw: Mapping[str, Any] | None = None

    @property
    def is_fallback(self) -> bool:
        return self.status not in GOOD_STATUSES

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "value": self.value,
            "confidence": self.confidence,
            "status": self.status,
            "raw": dict(self.raw) if self.raw is not None else None,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Answer:
        raw: object = data.get("raw")
        raw_map: Mapping[str, Any] | None = None
        if isinstance(raw, Mapping):
            items = cast("Mapping[object, Any]", raw)
            raw_map = {str(k): v for k, v in items.items()}
        return cls(
            question_id=str(data["question_id"]),
            value=data["value"],
            confidence=float(data["confidence"]),
            status=data["status"],
            raw=raw_map,
        )


def fallback_answer(question_id: str, question: Question, status: AnswerStatus) -> Answer:
    return Answer(question_id, question.fallback, 0.0, status)


def _prob(x: object) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except OverflowError:  # e.g. a 400-digit JSON integer from the provider
        return None
    if math.isnan(f) or f < 0.0 or f > 1.0:
        return None
    return f


def _distribution(values: list[float]) -> bool:
    return abs(sum(values) - 1.0) <= 0.02


def resolve(question_id: str, question: Question, raw: object) -> Answer:
    """Validate a provider's raw answer against the question's schema.

    Anything out of schema resolves to the fallback with ``out_of_schema``; an
    in-schema answer below ``min_confidence`` resolves to the fallback with
    ``low_confidence``. Never raises.
    """
    if not isinstance(raw, Mapping):
        return fallback_answer(question_id, question, "out_of_schema")
    raw_map: Mapping[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    value: AnswerValue
    # Only the validated fields are kept: the ledger must not store anything
    # else a provider sends back (reasoning text, echoed context, non-JSON junk).
    validated: dict[str, Any]
    if isinstance(question, BooleanQuestion):
        p = _prob(raw_map.get("p_yes"))
        if p is None:
            return fallback_answer(question_id, question, "out_of_schema")
        value, confidence = p >= 0.5, abs(p - 0.5) * 2.0
        validated = {"p_yes": p}
    elif isinstance(question, ChoiceQuestion):
        probs_obj = raw_map.get("probs")
        if not isinstance(probs_obj, Mapping):
            return fallback_answer(question_id, question, "out_of_schema")
        probs_map: Mapping[Any, Any] = probs_obj  # pyright: ignore[reportUnknownVariableType]
        if set(probs_map.keys()) != set(question.options):
            return fallback_answer(question_id, question, "out_of_schema")
        parsed = {opt: _prob(probs_map[opt]) for opt in question.options}
        clean = {k: v for k, v in parsed.items() if v is not None}
        if len(clean) != len(question.options) or not _distribution(list(clean.values())):
            return fallback_answer(question_id, question, "out_of_schema")
        # Ties resolve to the option listed first, which callers order cheapest-first
        # only if they want to; the gate's policy never relies on tie order.
        value = max(question.options, key=lambda o: clean[o])
        confidence = clean[value]
        validated = {"probs": dict(clean)}
    else:
        levels_obj = raw_map.get("probs")
        if not isinstance(levels_obj, list):
            return fallback_answer(question_id, question, "out_of_schema")
        levels_list: list[Any] = levels_obj  # pyright: ignore[reportUnknownVariableType]
        levels = [_prob(x) for x in levels_list]
        clean_levels = [x for x in levels if x is not None]
        if len(clean_levels) != SCORE_LEVELS or len(levels) != SCORE_LEVELS:
            return fallback_answer(question_id, question, "out_of_schema")
        if not _distribution(clean_levels):
            return fallback_answer(question_id, question, "out_of_schema")
        best = max(range(SCORE_LEVELS), key=lambda i: clean_levels[i])
        value, confidence = best / (SCORE_LEVELS - 1), clean_levels[best]
        validated = {"probs": clean_levels}
    if confidence < question.min_confidence:
        return fallback_answer(question_id, question, "low_confidence")
    return Answer(question_id, value, confidence, "ok", validated)
