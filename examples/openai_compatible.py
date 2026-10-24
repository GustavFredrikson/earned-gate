"""Adapter sketch: answer typed questions with any OpenAI-compatible chat endpoint.

This is an *example*, not part of the library, and it is deliberately honest
about its weakness: a general LLM asked to emit probabilities is not a
calibrated classifier. That is exactly why earned-gate starts every provider
in measurement mode and only lets it suppress anything after its own labelled
track record clears the rollout gate.

The API key is passed in or read from an environment variable (default
OPENAI_API_KEY), kept in a private attribute and sent only in the request's
Authorization header; it never enters the context, the ledger or a
Decision. A missing key raises ProviderUnavailable, which the gate turns into
a fallback, and the fallback rate shows up in the rollout report.

``base_url`` must be https (plain http is accepted only for localhost), and
the default transport does not follow redirects, because urllib would copy
the Authorization header to the redirect target.

Rollout evidence is counted per ``model_version`` (``chat:<model>``). Pin a
dated model snapshot rather than a moving alias: if the alias moves to a new
snapshot, the gate keeps using evidence earned by the old one.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, cast

from earned_gate import (
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailable,
    Question,
)

Transport = Callable[[str, bytes, Mapping[str, str], float], bytes]

SYSTEM = (
    "You answer typed questions about a JSON state. Reply with one JSON object only. "
    'For each question id: boolean -> {"p_yes": number}; choice -> '
    '{"probs": {option: number}} summing to 1; score -> {"probs": [5 numbers]} summing to 1. '
    "Never include any other text."
)


LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn every 3xx into an HTTPError instead of following it."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _urllib_transport(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310
    with _OPENER.open(req, timeout=timeout) as resp:
        data: bytes = resp.read()
        return data


def _check_base_url(base_url: str) -> None:
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK):
        return
    raise ValueError(f"base_url must be https (http only for localhost), got {base_url!r}")


def describe(questions: Mapping[str, Question]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for qid, q in questions.items():
        out.append({"id": qid, **q.spec()})
    return out


class ChatCompletionsProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        transport: Transport = _urllib_transport,
    ) -> None:
        _check_base_url(base_url)
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key if api_key is not None else os.environ.get(api_key_env)
        self._transport = transport

    @property
    def model_version(self) -> str:
        return f"chat:{self._model}"

    def ask(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        if not self._key:
            raise ProviderUnavailable("API key is not set")
        body = json.dumps(
            {
                "model": self._model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "state": json.loads(request.context_json),
                                "questions": describe(request.questions),
                            }
                        ),
                    },
                ],
            }
        ).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        raw = json.loads(self._transport(self._url, body, headers, timeout))
        content = raw["choices"][0]["message"]["content"]
        tokens = int(raw.get("usage", {}).get("total_tokens", 0))
        parsed: object = json.loads(content)
        answers: dict[str, Any] = {}
        if isinstance(parsed, dict):
            answers = cast("dict[str, Any]", parsed)
        return ProviderResponse(answers, tokens)
