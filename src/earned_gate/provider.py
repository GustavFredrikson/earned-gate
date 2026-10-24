"""The provider protocol: one projected state, several typed questions, one call.

A provider can be anything that answers typed questions with probabilities: a
typed decision model, a small classifier, or an LLM behind an
adapter. earned-gate ships no network client in its core; see
``earned_gate.testing.ScriptedProvider`` for tests and
``examples/openai_compatible.py`` for an adapter sketch.

Providers may raise anything. The gate turns
:class:`~earned_gate.errors.ProviderUnavailable` into status
``provider_unavailable``, :class:`TimeoutError` into ``timeout`` and any other
exception into ``provider_error``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .questions import Question


@dataclass(frozen=True)
class ProviderRequest:
    """What the provider sees. ``context_json`` is the validated canonical encoding."""

    feature: str
    context: Mapping[str, Any]
    context_json: str
    questions: Mapping[str, Question]


@dataclass(frozen=True)
class ProviderResponse:
    """Raw per-question answers (validated later by the gate) and tokens spent."""

    answers: Mapping[str, object]
    tokens: int = 0


class Provider(Protocol):
    @property
    def model_version(self) -> str:
        """Identifies the model; rollout evidence is counted per version."""
        ...

    def ask(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse: ...
