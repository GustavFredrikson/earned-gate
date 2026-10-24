"""The rollout gate: a model earns authority from its own labelled track record.

Suppression (letting a cheap route end a wake without the agent) is only
allowed while :func:`assess` says the gate is open. It fails closed: too little
evidence, any false negative, a high weighted miss score, a high fallback rate,
a degraded provider, or an exception while computing any of it all mean
*closed*. Nothing here loosens a threshold automatically, and evidence is
counted per model version, so a new model version starts from zero.

Definitions (route ranks come from the ordered route list, cheapest first):

* *labelled*: an evaluation with a shadow route and a label.
* *would suppress*: a labelled evaluation whose shadow route is skippable.
* *false negative*: shadow route skippable, label not skippable.
* *miss score*: ``(w * FN + other mismatches) / labelled`` where ``w`` is
  ``false_negative_weight`` (default 3) and a mismatch is any labelled
  evaluation whose shadow route differs from its label.
* *fallback rate*: evaluations whose status is not ``ok`` or ``reused`` /
  all evaluations except deterministic escalations. The denominator therefore
  includes reused answers and evaluations where the provider was never
  called (``budget_exhausted``, ``rule_error``), so an exhausted budget also
  raises the fallback rate and can show up as "provider degraded".
* *degraded*: the fallback rate over the last ``degraded_window``
  non-escalation evaluations of the feature (any model version) exceeds
  ``max_recent_fallback_rate``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ._checks import check_int, check_number
from .errors import ConfigError
from .ledger import EvaluationRecord, Ledger

DAY = 86400.0


@dataclass(frozen=True)
class RolloutPolicy:
    window_s: float = 30 * DAY
    min_labelled: int = 30
    min_would_suppress: int = 15
    max_false_negatives: int = 0
    max_miss_score: float = 0.15
    false_negative_weight: float = 3.0
    max_fallback_rate: float = 0.2
    degraded_window: int = 20
    max_recent_fallback_rate: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "min_labelled",
            "min_would_suppress",
            "max_false_negatives",
            "degraded_window",
        ):
            check_int(name, getattr(self, name))
        for name in (
            "window_s",
            "max_miss_score",
            "false_negative_weight",
            "max_fallback_rate",
            "max_recent_fallback_rate",
        ):
            check_number(name, getattr(self, name))
        if self.window_s <= 0 or self.degraded_window <= 0:
            raise ConfigError("window_s and degraded_window must be positive")
        if self.min_labelled < 1 or self.min_would_suppress < 0 or self.max_false_negatives < 0:
            raise ConfigError("evidence thresholds must be non-negative (min_labelled >= 1)")
        if self.false_negative_weight < 1:
            raise ConfigError("false_negative_weight must be >= 1")
        for name in ("max_miss_score", "max_fallback_rate", "max_recent_fallback_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RolloutReport:
    open: bool
    reasons: tuple[str, ...]
    model_version: str
    evaluations: int = 0
    labelled: int = 0
    would_suppress: int = 0
    false_negatives: int = 0
    mismatches: int = 0
    miss_score: float = 0.0
    fallback_rate: float = 0.0
    recent_fallback_rate: float = 0.0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def check_routes(routes: Sequence[str], skippable: Collection[str]) -> None:
    """Raise ConfigError unless routes/skippable form a meaningful gate.

    Same rules as :class:`~earned_gate.GateConfig`: at least two unique,
    non-empty route names ordered cheapest first; at least one skippable
    route; every skippable route known; the top route never skippable.
    """
    if len(routes) < 2:
        raise ConfigError("routes must be at least two names, cheapest first")
    if len(set(routes)) != len(routes) or not all(routes):
        raise ConfigError("routes must be unique, non-empty names")
    if not skippable:
        raise ConfigError("at least one skippable route is required")
    unknown = set(skippable) - set(routes)
    if unknown:
        raise ConfigError(f"unknown skippable routes: {sorted(unknown)}")
    if routes[-1] in skippable:
        raise ConfigError(f"the top route {routes[-1]!r} is the fallback; it cannot be skippable")


def _asked(records: Sequence[EvaluationRecord]) -> list[EvaluationRecord]:
    return [r for r in records if r.status != "escalated"]


def assess(
    ledger: Ledger,
    *,
    feature: str,
    model_version: str,
    routes: Sequence[str],
    skippable: Collection[str],
    policy: RolloutPolicy,
) -> RolloutReport:
    """Compute whether suppression may be enabled right now. Never raises."""
    try:
        return _assess(ledger, feature, model_version, routes, skippable, policy)
    except Exception as exc:  # fail closed on anything unexpected
        return RolloutReport(
            False, (f"error while assessing: {type(exc).__name__}: {exc}",), model_version
        )


def _assess(
    ledger: Ledger,
    feature: str,
    model_version: str,
    routes: Sequence[str],
    skippable: Collection[str],
    policy: RolloutPolicy,
) -> RolloutReport:
    now = ledger.now()
    window = _asked(
        ledger.evaluations(feature, since=now - policy.window_s, model_version=model_version)
    )
    rank = {r: i for i, r in enumerate(routes)}
    top = len(routes) - 1

    labelled = [r for r in window if r.label is not None and r.shadow_route is not None]
    would = [r for r in labelled if r.shadow_route in skippable]
    fn = sum(1 for r in would if r.label not in skippable)
    mismatches = sum(
        1 for r in labelled if rank.get(r.shadow_route or "", top) != rank.get(r.label or "", top)
    )
    weighted = policy.false_negative_weight * fn + (mismatches - fn)
    miss = weighted / len(labelled) if labelled else 1.0
    fallback_rate = sum(r.fallback for r in window) / len(window) if window else 0.0

    recent = ledger.recent_fallbacks(
        feature, since=now - policy.window_s, limit=policy.degraded_window
    )
    recent_rate = sum(recent) / len(recent) if recent else 0.0

    reasons: list[str] = []
    if len(labelled) < policy.min_labelled:
        reasons.append(f"labelled evaluations {len(labelled)} < {policy.min_labelled}")
    if len(would) < policy.min_would_suppress:
        reasons.append(f"would-suppress evaluations {len(would)} < {policy.min_would_suppress}")
    if fn > policy.max_false_negatives:
        reasons.append(f"false negatives {fn} > {policy.max_false_negatives}")
    if miss > policy.max_miss_score:
        reasons.append(f"weighted miss score {miss:.3f} > {policy.max_miss_score}")
    if fallback_rate > policy.max_fallback_rate:
        reasons.append(f"fallback rate {fallback_rate:.3f} > {policy.max_fallback_rate}")
    if recent_rate > policy.max_recent_fallback_rate:
        reasons.append(
            f"provider degraded: recent fallback rate {recent_rate:.3f}"
            f" > {policy.max_recent_fallback_rate}"
        )
    return RolloutReport(
        open=not reasons,
        reasons=tuple(reasons),
        model_version=model_version,
        evaluations=len(window),
        labelled=len(labelled),
        would_suppress=len(would),
        false_negatives=fn,
        mismatches=mismatches,
        miss_score=round(miss, 6),
        fallback_rate=round(fallback_rate, 6),
        recent_fallback_rate=round(recent_rate, 6),
    )
