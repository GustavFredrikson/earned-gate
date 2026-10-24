"""The gate: deterministic code around a bounded model judgment.

Order of operations for :meth:`Gate.evaluate`::

    projected context
       -> boundary (JSON-only, secret-free, size)       raises BoundaryError
       -> question/signal config checks                 raises ConfigError
       -> deterministic escalation                      -> top route, no model call
          (a rule that raises                           -> top route, "rule_error")
       -> reuse (semantic hash, TTL, same model)        -> zero tokens
       -> token budget                                  -> fallback
       -> provider (one request, many questions)        -> failures become fallbacks
       -> per-question schema + confidence checks       -> failures become fallbacks
       -> policy (suppression on, allowlist, confidence, materiality, rollout gate)
       -> multi-signal floor
       -> append to the hash-chained ledger
       -> Decision

Only ``BoundaryError`` and ``ConfigError`` (integration bugs) propagate out of
:meth:`Gate.evaluate`. A failing escalation rule, a failing ledger read or
write, or a provider whose ``model_version`` raises all resolve to the top
route with ``skip_agent=False``, a status (``rule_error``, ``ledger_error``,
``provider_unavailable``) and a reason, and are logged as warnings on the
``earned_gate`` logger.

The gate never writes anything except its own ledger, and it must never be
used as a security, authorization, idempotency, approval, freshness or mutation
boundary. For state-changing actions use preview/apply/undo (for MCP tools,
mcp-changeset) behind whatever route the gate picks.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ._checks import check_int, check_number
from .boundary import (
    DEFAULT_MAX_BYTES,
    canonical_json,
    semantic_hash,
    validate_context,
    validate_signals,
)
from .errors import ConfigError, ProviderUnavailable
from .ledger import EvaluationRecord, Ledger, new_eval_id
from .provider import Provider, ProviderRequest, ProviderResponse
from .questions import (
    Answer,
    AnswerStatus,
    BooleanQuestion,
    ChoiceQuestion,
    Question,
    ScoreQuestion,
    fallback_answer,
    resolve,
)
from .rollout import DAY, RolloutPolicy, RolloutReport, assess

log = logging.getLogger("earned_gate")

DEFAULT_VOLATILE_KEYS = frozenset(
    {"ts", "timestamp", "time", "created_at", "updated_at", "observed_at", "request_id"}
)

#: A deterministic escalation rule: returns a reason to escalate, or None.
EscalationRule = Callable[[Mapping[str, Any]], "str | None"]

DecisionStatus = Literal[
    "ok",
    "reused",
    "partial",
    "escalated",
    "provider_error",
    "provider_unavailable",
    "timeout",
    "budget_exhausted",
    "rule_error",
    "ledger_error",
]


@dataclass(frozen=True)
class SignalFloor:
    """Combine individually weak signals; above ``threshold`` raise the route.

    The score is ``sum(weights[name] * signals[name])`` over the signals the
    caller passes to :meth:`Gate.evaluate`. A score that overflows to an
    infinity or NaN counts as over the threshold.
    """

    weights: Mapping[str, float]
    threshold: float
    raise_to: str


@dataclass(frozen=True)
class GateConfig:
    """Static configuration for one gated feature.

    ``routes`` are ordered cheapest first; the last route is "do what you did
    before the gate existed" (typically: invoke the agent) and is the fallback
    for every failure. ``skippable`` routes end the wake without the agent.
    ``allowlist`` is the set of cheap routes the model may choose directly once
    suppression is enabled and the rollout gate is open; anything else is
    raised to at least ``denied_route``.

    An empty ``allowlist`` means "no authority yet": the gate stays in stage A
    (it never skips, even with ``suppression_enabled=True``), and the shadow
    route is computed as if the allowlist were ``skippable``, so measurement
    traffic still produces would-suppress evidence and the rollout gate can
    open. See :attr:`shadow_allowlist`.
    """

    feature: str
    routes: tuple[str, ...]
    skippable: frozenset[str]
    route_question: str
    allowlist: frozenset[str] = frozenset()
    suppression_enabled: bool = False
    min_route_confidence: float = 0.90
    materiality_question: str | None = None
    denied_route: str | None = None
    signal_floor: SignalFloor | None = None
    escalation_rules: tuple[EscalationRule, ...] = ()
    escalation_lookback_s: float = 6 * 3600.0
    rollout: RolloutPolicy = field(default_factory=RolloutPolicy)
    rollout_cache_s: float = 60.0
    reuse_ttl_s: float = 900.0
    volatile_keys: frozenset[str] = DEFAULT_VOLATILE_KEYS
    max_context_bytes: int = DEFAULT_MAX_BYTES
    allow_keys: frozenset[str] = frozenset()
    daily_token_budget: int | None = None
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        self._check_types()
        if not self.feature:
            raise ConfigError("feature must not be empty")
        if len(self.routes) < 2 or len(set(self.routes)) != len(self.routes):
            raise ConfigError("routes must be at least two unique names, cheapest first")
        top = self.routes[-1]
        unknown = (self.skippable | self.allowlist) - set(self.routes)
        if unknown:
            raise ConfigError(f"unknown routes: {sorted(unknown)}")
        if top in self.skippable or top in self.allowlist:
            raise ConfigError(f"the top route {top!r} is the fallback; it cannot be skippable")
        if self.denied_route is not None and self.denied_route not in self.routes:
            raise ConfigError(f"denied_route {self.denied_route!r} is not a route")
        if self.denied_route in self.allowlist:
            raise ConfigError("denied_route must not be on the allowlist")
        if self.denied_route in self.skippable:
            # A denied answer is raised to denied_route; if that route skips the
            # agent, low-confidence or material answers would still skip.
            raise ConfigError("denied_route must not be skippable")
        if self.signal_floor is not None:
            if self.signal_floor.raise_to not in self.routes:
                raise ConfigError(
                    f"signal_floor.raise_to {self.signal_floor.raise_to!r} is not a route"
                )
            if self.signal_floor.raise_to in self.skippable:
                raise ConfigError("signal_floor.raise_to must not be skippable")
        check_number("min_route_confidence", self.min_route_confidence)
        check_int("max_context_bytes", self.max_context_bytes)
        if self.daily_token_budget is not None:
            check_int("daily_token_budget", self.daily_token_budget)
        if not 0.0 <= self.min_route_confidence <= 1.0:
            raise ConfigError("min_route_confidence must be in [0, 1]")
        for name in ("timeout_s", "reuse_ttl_s", "rollout_cache_s", "escalation_lookback_s"):
            check_number(name, getattr(self, name))
        if self.rollout_cache_s < 0:
            raise ConfigError("rollout_cache_s must be >= 0")
        if self.escalation_lookback_s <= 0:
            # zero or negative would silently stop escalations labelling misses
            raise ConfigError("escalation_lookback_s must be > 0")
        if self.timeout_s <= 0 or self.reuse_ttl_s < 0 or self.max_context_bytes <= 0:
            raise ConfigError("timeout_s, max_context_bytes must be positive; reuse_ttl_s >= 0")
        if self.daily_token_budget is not None and self.daily_token_budget < 0:
            raise ConfigError("daily_token_budget must be >= 0")

    def _check_types(self) -> None:
        """Wrong types raise ConfigError; lists and sets are normalised.

        Without this a string passed as ``routes`` or ``skippable`` is iterated
        character by character, and other wrong types surface later as a
        TypeError from inside :meth:`Gate.evaluate`.
        """

        def names(name: str, value: object) -> list[str]:
            if not isinstance(value, (list, tuple, set, frozenset)):
                raise ConfigError(
                    f"{name} must be a collection of names, not {type(value).__name__}"
                )
            items = cast("list[object]", list(value))  # pyright: ignore[reportUnknownArgumentType]
            if not all(isinstance(v, str) for v in items):
                raise ConfigError(f"{name} must contain only strings")
            return cast("list[str]", items)

        def text(name: str, value: object, *, optional: bool = False) -> None:
            if not (isinstance(value, str) or (optional and value is None)):
                raise ConfigError(f"{name} must be a string, not {type(value).__name__}")

        def of(attr: str) -> object:
            return cast(object, getattr(self, attr))

        set_ = object.__setattr__
        set_(self, "routes", tuple(names("routes", of("routes"))))
        for name in ("skippable", "allowlist", "volatile_keys", "allow_keys"):
            set_(self, name, frozenset(names(name, of(name))))
        text("feature", of("feature"))
        text("route_question", of("route_question"))
        text("materiality_question", of("materiality_question"), optional=True)
        text("denied_route", of("denied_route"), optional=True)
        if not isinstance(of("suppression_enabled"), bool):
            raise ConfigError("suppression_enabled must be a bool")
        if not isinstance(of("rollout"), RolloutPolicy):
            raise ConfigError("rollout must be a RolloutPolicy")
        rules = of("escalation_rules")
        if not isinstance(rules, (list, tuple)):
            raise ConfigError("escalation_rules must be a tuple of callables")
        rule_list = cast("list[object]", list(rules))  # pyright: ignore[reportUnknownArgumentType]
        if not all(callable(r) for r in rule_list):
            raise ConfigError("escalation_rules must be a tuple of callables")
        set_(self, "escalation_rules", tuple(rule_list))
        floor = of("signal_floor")
        if floor is None:
            return
        if not isinstance(floor, SignalFloor):
            raise ConfigError("signal_floor must be a SignalFloor")
        weights = cast(object, floor.weights)
        if not isinstance(weights, Mapping):
            raise ConfigError("signal_floor.weights must be a mapping of name -> number")
        pairs = cast("list[tuple[object, object]]", list(weights.items()))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        for key, weight in pairs:
            if not isinstance(key, str):
                raise ConfigError("signal_floor.weights keys must be strings")
            check_number(f"signal_floor.weights[{key!r}]", weight)
        check_number("signal_floor.threshold", floor.threshold)
        text("signal_floor.raise_to", cast(object, floor.raise_to))

    @property
    def top_route(self) -> str:
        return self.routes[-1]

    @property
    def effective_denied_route(self) -> str:
        if self.denied_route is not None:
            return self.denied_route
        return next(r for r in self.routes if r not in self.skippable)

    @property
    def shadow_allowlist(self) -> frozenset[str]:
        """The allowlist used to compute the shadow route.

        ``allowlist`` if set; otherwise ``skippable``, so that an empty
        allowlist in stage A measures what stage B would do instead of raising
        every shadow route to the denied route.
        """
        return self.allowlist or self.skippable

    @property
    def stage(self) -> Literal["A", "B", "C"]:
        """A: measurement only. B: only skippable routes may act. C: wider allowlist."""
        if not self.suppression_enabled or not self.allowlist:
            return "A"
        return "B" if self.allowlist <= self.skippable else "C"

    def rank(self, route: str) -> int:
        return self.routes.index(route)

    def check_questions(self, questions: Mapping[str, Question]) -> None:
        raw = cast(object, questions)  # public input; do not trust the annotation
        if not isinstance(raw, Mapping):
            raise ConfigError(
                "questions must be a mapping of question ids to BooleanQuestion,"
                f" ChoiceQuestion or ScoreQuestion, not {type(raw).__name__}"
            )
        items: Mapping[object, object] = raw  # pyright: ignore[reportUnknownVariableType]
        for qid, q in items.items():
            if not isinstance(qid, str):
                raise ConfigError(f"question ids must be strings, not {type(qid).__name__}")
            if not isinstance(q, (BooleanQuestion, ChoiceQuestion, ScoreQuestion)):
                raise ConfigError(
                    f"questions[{qid!r}] must be a BooleanQuestion, ChoiceQuestion or"
                    f" ScoreQuestion, not {type(q).__name__}"
                )
            try:
                canonical_json(q.spec()).encode("utf-8")  # also rejects lone surrogates
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"questions[{qid!r}] is not JSON-serialisable: {exc}") from exc
        rq = questions.get(self.route_question)
        if not isinstance(rq, ChoiceQuestion):
            raise ConfigError(f"route question {self.route_question!r} must be a ChoiceQuestion")
        if rq.options != self.routes:
            raise ConfigError("route question options must equal the configured routes, in order")
        if rq.fallback != self.top_route:
            raise ConfigError(f"route question fallback must be the top route {self.top_route!r}")
        if self.materiality_question is not None and not isinstance(
            questions.get(self.materiality_question), BooleanQuestion
        ):
            raise ConfigError(
                f"materiality question {self.materiality_question!r} must be a BooleanQuestion"
            )


@dataclass(frozen=True)
class Decision:
    """The outcome of one evaluation.

    Act on a cheap route only if ``skip_agent`` is true; every failure path
    sets ``route`` to the top route and ``skip_agent`` to false.

    ``recorded`` is false when the evaluation could not be written to the
    ledger (status ``ledger_error``, or ``provider_unavailable`` because the
    provider's ``model_version`` raised); such an ``eval_id`` cannot be
    labelled.
    """

    eval_id: str
    route: str
    skip_agent: bool
    status: DecisionStatus
    reason: str
    answers: Mapping[str, Answer]
    shadow_route: str | None
    model_version: str
    tokens: int
    stage: str
    rollout: RolloutReport | None = None
    recorded: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "route": self.route,
            "skip_agent": self.skip_agent,
            "status": self.status,
            "reason": self.reason,
            "answers": {k: a.to_json() for k, a in self.answers.items()},
            "shadow_route": self.shadow_route,
            "model_version": self.model_version,
            "tokens": self.tokens,
            "stage": self.stage,
            "rollout": self.rollout.to_json() if self.rollout else None,
            "recorded": self.recorded,
        }


class Gate:
    """Gate one feature's model calls. Thread-safe; share one per feature."""

    def __init__(
        self,
        config: GateConfig,
        provider: Provider,
        ledger: Ledger,
        *,
        max_workers: int = 4,
    ) -> None:
        self.config = config
        self.provider = provider
        self.ledger = ledger
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="earned-gate"
        )
        self._cache_lock = threading.Lock()
        self._closed = False
        self._rollout_cache: tuple[str, float, tuple[int, int], RolloutReport] | None = None
        # Tokens the provider spent on evaluations whose ledger write failed;
        # they still count against daily_token_budget (in this process).
        self._unrecorded_tokens: list[tuple[float, int]] = []

    def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Gate:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public -------------------------------------------------------------------

    def rollout_status(self) -> RolloutReport:
        """Assess the rollout gate now, from the ledger (never cached)."""
        return self._assess(self.provider.model_version, self.ledger.now())

    def _assess(self, model_version: str, now: float) -> RolloutReport:
        cfg = self.config
        token = self._change_token()
        report = assess(
            self.ledger,
            feature=cfg.feature,
            model_version=model_version,
            routes=cfg.routes,
            skippable=cfg.skippable,
            policy=cfg.rollout,
        )
        if token is not None:
            with self._cache_lock:
                self._rollout_cache = (model_version, now, token, report)
        return report

    def _change_token(self) -> tuple[int, int] | None:
        try:
            return self.ledger.change_token()
        except Exception:
            return None  # cannot validate a cache, so do not keep one

    def _cached_rollout(self, model_version: str, now: float) -> RolloutReport:
        """The rollout report, reused while nothing that could close the gate
        was written, for at most ``rollout_cache_s``.

        Before a cached report is served, :meth:`Ledger.change_token` is
        compared (one ``PRAGMA data_version`` plus an in-process counter). Any
        commit from another connection or process (for example the CLI's
        ``label``), and any label, fallback or escalation written through this
        ledger, forces a fresh assessment. ``rollout_cache_s`` then only
        bounds how long evidence ageing out of the window can go unnoticed.
        """
        if self.config.rollout_cache_s > 0:
            with self._cache_lock:
                cached = self._rollout_cache
            if (
                cached is not None
                and cached[0] == model_version
                and 0 <= now - cached[1] < self.config.rollout_cache_s
                and self._change_token() == cached[2]
            ):
                return cached[3]
        return self._assess(model_version, now)

    def _unrecorded_tokens_since(self, since: float) -> int:
        with self._cache_lock:
            self._unrecorded_tokens = [(t, n) for t, n in self._unrecorded_tokens if t >= since]
            return sum(n for _, n in self._unrecorded_tokens)

    def _invalidate_rollout(self) -> None:
        with self._cache_lock:
            self._rollout_cache = None

    def label(self, eval_id: str, actual_route: str, *, source: str = "agent") -> None:
        """Record which route was actually needed for an earlier evaluation."""
        if actual_route not in self.config.routes:
            raise ConfigError(f"unknown route {actual_route!r}")
        self.ledger.label(eval_id, actual_route, source=source)
        self._invalidate_rollout()

    def evaluate(
        self,
        context: Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        escalate: str | None = None,
        signals: Mapping[str, float] | None = None,
    ) -> Decision:
        """Evaluate one projected context.

        Raises only :class:`BoundaryError` and :class:`ConfigError`
        (integration bugs). Every other failure resolves to the top route.
        """
        cfg = self.config
        if self._closed:
            raise ConfigError("this Gate is closed; create a new one")
        context_json = validate_context(
            context, max_bytes=cfg.max_context_bytes, allow_keys=cfg.allow_keys
        )
        cfg.check_questions(questions)
        if escalate is not None and not isinstance(cast(object, escalate), str):
            raise ConfigError(
                f"escalate must be a string or None, not {type(cast(object, escalate)).__name__}"
            )
        signals = validate_signals(signals)
        self._check_signals(signals)
        try:
            raw_version = cast(object, self.provider.model_version)  # third-party code
            if not isinstance(raw_version, str):
                raise TypeError(f"model_version is {type(raw_version).__name__}, not str")
            model_version = raw_version
        except Exception as exc:
            return self._unrecorded(
                questions,
                "provider_unavailable",
                "",
                f"provider.model_version raised {_describe(exc)}",
                answer_status="provider_unavailable",
            )
        specs = {qid: q.spec() for qid, q in sorted(questions.items())}
        state_hash = semantic_hash(context, cfg.volatile_keys, specs, model_version)

        try:
            reason = escalate or self._escalation_reason(context)
        except _RuleError as err:
            return self._rule_error(err, questions, state_hash, model_version)
        try:
            now = self.ledger.now()
            if reason is not None:
                return self._escalate(reason, questions, state_hash, model_version, now)
            reused = self.ledger.find_reusable(
                cfg.feature, state_hash, model_version, since=now - cfg.reuse_ttl_s
            )
            over_budget = reused is None and (
                cfg.daily_token_budget is not None
                and self.ledger.tokens_since(cfg.feature, now - DAY)
                + self._unrecorded_tokens_since(now - DAY)
                >= cfg.daily_token_budget
            )
        except Exception as exc:
            what = f"deterministic escalation: {reason}; " if reason else ""
            return self._unrecorded(
                questions, "ledger_error", model_version, f"{what}ledger error: {_describe(exc)}"
            )

        tokens = 0
        if reused is not None:
            answers = self._re_resolve(reused, questions)
            fell_back = any(a.is_fallback for a in answers.values())
            status: DecisionStatus = "partial" if fell_back else "reused"
        elif over_budget:
            answers = self._all_fallback(questions, "budget_exhausted")
            status = "budget_exhausted"
        else:
            answers, status, tokens = self._ask(context, context_json, questions)

        return self._decide(answers, status, signals, state_hash, model_version, now, tokens)

    # -- internals ------------------------------------------------------------------

    def _check_signals(self, signals: Mapping[str, float]) -> None:
        if not signals:
            return
        floor = self.config.signal_floor
        if floor is None:
            raise ConfigError("signals were passed but no signal_floor is configured")
        unknown = set(signals) - set(floor.weights)
        if unknown:
            raise ConfigError(f"unknown signals: {sorted(unknown)}")

    def _escalation_reason(self, context: Mapping[str, Any]) -> str | None:
        for rule in self.config.escalation_rules:
            try:
                result = rule(context)
                reason = str(result) if result else None  # bool()/str() are rule code too
            except Exception as exc:
                raise _RuleError(rule, exc) from exc
            if reason is not None:
                return reason
        return None

    def _unrecorded(
        self,
        questions: Mapping[str, Question],
        status: DecisionStatus,
        model_version: str,
        reason: str,
        *,
        answer_status: AnswerStatus = "skipped",
        answers: Mapping[str, Answer] | None = None,
        shadow_route: str | None = None,
        tokens: int = 0,
    ) -> Decision:
        """A fail-safe decision that could not be written to the ledger."""
        log.warning(
            "earned-gate %s: %s; falling back to the top route", self.config.feature, reason
        )
        return Decision(
            eval_id=new_eval_id(),
            route=self.config.top_route,
            skip_agent=False,
            status=status,
            reason=reason,
            answers=answers
            if answers is not None
            else self._all_fallback(questions, answer_status),
            shadow_route=shadow_route,
            model_version=model_version,
            tokens=tokens,
            stage=self.config.stage,
            recorded=False,
        )

    def _rule_error(
        self,
        err: _RuleError,
        questions: Mapping[str, Question],
        state_hash: str,
        model_version: str,
    ) -> Decision:
        """An escalation rule raised: escalate without labelling, record as a fallback."""
        cfg = self.config
        reason = f"escalation rule {err.rule_name} raised {_describe(err.cause)}"
        answers = self._all_fallback(questions, "skipped")
        try:
            record = EvaluationRecord(
                eval_id=new_eval_id(),
                ts=self.ledger.now(),
                feature=cfg.feature,
                model_version=model_version,
                state_hash=state_hash,
                status="rule_error",
                final_route=cfg.top_route,
                shadow_route=None,
                answers=tuple(answers.values()),
                tokens=0,
                fallback=True,
                suppressed=False,
            )
            self.ledger.record_evaluation(record)
        except Exception as exc:
            return self._unrecorded(
                questions,
                "ledger_error",
                model_version,
                f"{reason}; ledger error: {_describe(exc)}",
            )
        self._invalidate_rollout()
        log.warning("earned-gate %s: %s; falling back to the top route", cfg.feature, reason)
        return Decision(
            eval_id=record.eval_id,
            route=cfg.top_route,
            skip_agent=False,
            status="rule_error",
            reason=reason,
            answers=answers,
            shadow_route=None,
            model_version=model_version,
            tokens=0,
            stage=cfg.stage,
        )

    @staticmethod
    def _re_resolve(
        reused: EvaluationRecord, questions: Mapping[str, Question]
    ) -> dict[str, Answer]:
        """Re-validate a reused evaluation's stored raw answers against the
        *current* questions, so a raised ``min_confidence`` applies to reused
        answers too. The spec hash already guarantees same kind, prompt and
        options."""
        prior = {a.question_id: a for a in reused.answers}
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            old = prior.get(qid)
            fresh = _safe_resolve(qid, q, old.raw if old is not None else None)
            if fresh.status == "ok":
                fresh = Answer(qid, fresh.value, fresh.confidence, "reused", fresh.raw)
            out[qid] = fresh
        return out

    @staticmethod
    def _all_fallback(questions: Mapping[str, Question], status: AnswerStatus) -> dict[str, Answer]:
        return {qid: fallback_answer(qid, q, status) for qid, q in questions.items()}

    def _ask(
        self, context: Mapping[str, Any], context_json: str, questions: Mapping[str, Question]
    ) -> tuple[dict[str, Answer], DecisionStatus, int]:
        cfg = self.config
        request = ProviderRequest(cfg.feature, context, context_json, dict(questions))
        try:
            future = self._pool.submit(self.provider.ask, request, timeout=cfg.timeout_s)
        except RuntimeError:  # closed concurrently with this call
            return self._all_fallback(questions, "provider_error"), "provider_error", 0
        try:
            response = future.result(timeout=cfg.timeout_s)
        except (concurrent.futures.TimeoutError, TimeoutError):
            future.cancel()
            return self._all_fallback(questions, "timeout"), "timeout", 0
        except ProviderUnavailable:
            return self._all_fallback(questions, "provider_unavailable"), "provider_unavailable", 0
        except Exception:
            return self._all_fallback(questions, "provider_error"), "provider_error", 0
        # Providers are third-party code; do not trust their return type.
        resp = cast(object, response)
        if not isinstance(resp, ProviderResponse):
            return self._all_fallback(questions, "provider_error"), "provider_error", 0
        raw_answers = cast(object, resp.answers)
        if not isinstance(raw_answers, Mapping):
            return self._all_fallback(questions, "provider_error"), "provider_error", 0
        answers_map: Mapping[str, object] = raw_answers  # pyright: ignore[reportUnknownVariableType]
        raw_tokens = cast(object, resp.tokens)
        tokens = raw_tokens if isinstance(raw_tokens, int) and raw_tokens > 0 else 0
        answers = {qid: _safe_resolve(qid, q, answers_map.get(qid)) for qid, q in questions.items()}
        any_fallback = any(a.is_fallback for a in answers.values())
        return answers, ("partial" if any_fallback else "ok"), tokens

    def _policy_route(
        self, answers: Mapping[str, Answer], signals: Mapping[str, float]
    ) -> tuple[str, str]:
        """The route the model's answers justify if suppression were on and the gate open."""
        cfg = self.config
        route_answer = answers[cfg.route_question]
        if route_answer.is_fallback:
            return cfg.top_route, f"route answer fell back ({route_answer.status})"
        proposed = str(route_answer.value)
        route, why = proposed, f"model chose {proposed!r} ({route_answer.confidence:.2f})"
        denied = cfg.effective_denied_route

        def raise_to(target: str, because: str) -> None:
            nonlocal route, why
            if cfg.rank(target) > cfg.rank(route):
                route, why = target, because

        if proposed not in cfg.shadow_allowlist:
            raise_to(denied, f"{proposed!r} is not on the allowlist")
        elif route_answer.confidence < cfg.min_route_confidence:
            raise_to(
                denied, f"confidence {route_answer.confidence:.2f} < {cfg.min_route_confidence}"
            )
        if cfg.materiality_question is not None:
            m = answers[cfg.materiality_question]
            if m.is_fallback or m.value is True:
                raise_to(denied, "the change could plausibly matter (materiality)")
        floor = cfg.signal_floor
        if floor is not None and signals:
            score = sum(floor.weights[k] * v for k, v in signals.items())
            # Finite inputs can still overflow to inf or NaN; treat that as over.
            if not math.isfinite(score) or score >= floor.threshold:
                raise_to(floor.raise_to, f"signal floor {score:.2f} >= {floor.threshold}")
        return route, why

    def _decide(
        self,
        answers: dict[str, Answer],
        status: DecisionStatus,
        signals: Mapping[str, float],
        state_hash: str,
        model_version: str,
        now: float,
        tokens: int,
    ) -> Decision:
        cfg = self.config
        shadow, why = self._policy_route(answers, signals)
        report: RolloutReport | None = None
        if status not in ("ok", "reused", "partial"):
            route, reason = cfg.top_route, f"fallback: {status}"
        elif not cfg.suppression_enabled:
            route, reason = cfg.top_route, f"measurement stage; shadow: {why}"
        elif not cfg.allowlist:
            route, reason = (
                cfg.top_route,
                f"measurement stage (empty allowlist, nothing may skip); shadow: {why}",
            )
        else:
            report = self._cached_rollout(model_version, now)
            if not report.open:
                route, reason = cfg.top_route, "rollout gate closed: " + "; ".join(report.reasons)
            else:
                route, reason = shadow, why
        skip = route in cfg.skippable and status in ("ok", "reused", "partial")
        record = EvaluationRecord(
            eval_id=new_eval_id(),
            ts=now,
            feature=cfg.feature,
            model_version=model_version,
            state_hash=state_hash,
            status=status,
            final_route=route,
            shadow_route=shadow if status in ("ok", "reused", "partial") else None,
            answers=tuple(answers.values()),
            tokens=tokens,
            fallback=status not in ("ok", "reused"),
            suppressed=skip,
        )
        try:
            self.ledger.record_evaluation(record)
        except Exception as exc:
            if tokens and cfg.daily_token_budget is not None:
                with self._cache_lock:
                    self._unrecorded_tokens.append((now, tokens))
            return self._unrecorded(
                {},
                "ledger_error",
                model_version,
                f"ledger error: {_describe(exc)}; would have been {route!r} ({reason})",
                answers=answers,
                shadow_route=record.shadow_route,
                tokens=tokens,
            )
        if record.fallback:
            self._invalidate_rollout()
        return Decision(
            eval_id=record.eval_id,
            route=route,
            skip_agent=skip,
            status=status,
            reason=reason,
            answers=answers,
            shadow_route=record.shadow_route,
            model_version=model_version,
            tokens=tokens,
            stage=cfg.stage,
            rollout=report,
        )

    def _escalate(
        self,
        reason: str,
        questions: Mapping[str, Question],
        state_hash: str,
        model_version: str,
        now: float,
    ) -> Decision:
        cfg = self.config
        missed = self.ledger.label_missed_suppressions(
            cfg.feature,
            since=now - cfg.escalation_lookback_s,
            skippable=cfg.skippable,
            actual_route=cfg.top_route,
        )
        answers = self._all_fallback(questions, "skipped")
        record = EvaluationRecord(
            eval_id=new_eval_id(),
            ts=now,
            feature=cfg.feature,
            model_version=model_version,
            state_hash=state_hash,
            status="escalated",
            final_route=cfg.top_route,
            shadow_route=None,
            answers=tuple(answers.values()),
            tokens=0,
            fallback=False,
            suppressed=False,
        )
        self.ledger.record_evaluation(record)
        self._invalidate_rollout()
        suffix = f"; labelled {len(missed)} earlier suppression(s) as missed" if missed else ""
        return Decision(
            eval_id=record.eval_id,
            route=cfg.top_route,
            skip_agent=False,
            status="escalated",
            reason=f"deterministic escalation: {reason}{suffix}",
            answers=answers,
            shadow_route=None,
            model_version=model_version,
            tokens=0,
            stage=cfg.stage,
        )


class _RuleError(Exception):
    """Internal: an escalation rule raised."""

    def __init__(self, rule: EscalationRule, cause: Exception) -> None:
        super().__init__(_safe_str(cause))
        self.rule_name = _rule_name(rule)
        self.cause = cause


def _safe_resolve(qid: str, question: Question, raw: object) -> Answer:
    """resolve() is written never to raise; this is defence in depth for
    provider data (third-party, possibly prompt-injected)."""
    try:
        return resolve(qid, question, raw)
    except Exception:
        return fallback_answer(qid, question, "out_of_schema")


def _safe_str(value: object) -> str:
    """str() of caller-supplied objects, which may themselves raise."""
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}: str() raised>"


def _rule_name(rule: object) -> str:
    name = getattr(rule, "__qualname__", None)
    if isinstance(name, str) and name:
        return name
    try:
        return repr(rule)
    except Exception:
        return f"<{type(rule).__name__} instance>"


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_safe_str(exc)}"
