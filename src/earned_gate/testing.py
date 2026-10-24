"""Test helpers: a scripted provider and a controllable clock."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .provider import ProviderRequest, ProviderResponse

Script = Callable[[ProviderRequest], ProviderResponse]


@dataclass
class ScriptedProvider:
    """Returns fixed answers, or whatever ``script`` returns, and counts calls.

    Set ``raises`` to make every call raise, or ``delay`` to make it slow.
    """

    answers: Mapping[str, object] = field(default_factory=dict[str, object])
    tokens: int = 100
    version: str = "scripted-1"
    script: Script | None = None
    raises: BaseException | None = None
    delay: float = 0.0
    calls: list[ProviderRequest] = field(default_factory=list[ProviderRequest])

    @property
    def model_version(self) -> str:
        return self.version

    def ask(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        if self.script is not None:
            return self.script(request)
        return ProviderResponse(dict(self.answers), self.tokens)


@dataclass
class FakeClock:
    now: float = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
