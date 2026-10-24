"""earned-gate: let a cheap typed model call skip an expensive agent turn,
but only once it has earned the right to on your own traffic."""

from __future__ import annotations

from .boundary import semantic_hash, validate_context
from .errors import (
    BoundaryError,
    ConfigError,
    EarnedGateError,
    LedgerError,
    LedgerIntegrityError,
    ProviderUnavailable,
)
from .gate import Decision, EscalationRule, Gate, GateConfig, SignalFloor
from .ledger import EvaluationRecord, Ledger
from .provider import Provider, ProviderRequest, ProviderResponse
from .questions import Answer, BooleanQuestion, ChoiceQuestion, Question, ScoreQuestion
from .rollout import RolloutPolicy, RolloutReport, assess

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "BooleanQuestion",
    "BoundaryError",
    "ChoiceQuestion",
    "ConfigError",
    "Decision",
    "EarnedGateError",
    "EscalationRule",
    "EvaluationRecord",
    "Gate",
    "GateConfig",
    "Ledger",
    "LedgerError",
    "LedgerIntegrityError",
    "Provider",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderUnavailable",
    "Question",
    "RolloutPolicy",
    "RolloutReport",
    "ScoreQuestion",
    "SignalFloor",
    "__version__",
    "assess",
    "semantic_hash",
    "validate_context",
]
