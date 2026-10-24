from __future__ import annotations

import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import openai_compatible
import training_watch
from conftest import ROUTES, base_config, questions

from earned_gate import Gate, Ledger


def run(tmp_path: Path, event: dict[str, object], *flags: str) -> tuple[int, dict[str, object]]:
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        code = training_watch.main(
            ["--ledger", str(tmp_path / "w.db"), *flags], io.StringIO(json.dumps(event))
        )
    finally:
        sys.stdout = old
    return code, json.loads(out.getvalue())


def test_training_watch_measurement_stage_wakes_the_agent(tmp_path: Path) -> None:
    code, d = run(tmp_path, {"trigger": "hrv_shift", "hrv_delta_pct": -2, "ts": "x"})
    assert code == 1
    assert d["route"] == "agent" and d["shadow_route"] == "ignore"


def test_training_watch_ungated_trigger_and_escalation(tmp_path: Path) -> None:
    assert run(tmp_path, {"trigger": "new_activity"})[0] == 1
    code, d = run(tmp_path, {"trigger": "hrv_shift", "symptom": "chest_pain"}, "--suppress")
    assert code == 1 and d["status"] == "escalated"


def fake_transport(content: object, tokens: int = 42) -> openai_compatible.Transport:
    def transport(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> bytes:
        req = json.loads(body)
        assert url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer test-key"
        assert req["response_format"] == {"type": "json_object"}
        payload = {
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"total_tokens": tokens},
        }
        return json.dumps(payload).encode()

    return transport


def test_chat_adapter_roundtrip(tmp_path: Path) -> None:
    rest = 0.01
    content = {
        "material": {"p_yes": 0.02},
        "route": {"probs": {r: (0.96 if r == "ignore" else rest) for r in ROUTES}},
    }
    prov = openai_compatible.ChatCompletionsProvider(
        "small-model", api_key="test-key", transport=fake_transport(content)
    )
    with Ledger(tmp_path / "a.db") as led, Gate(base_config(), prov, led) as gate:
        d = gate.evaluate({"event": "hrv"}, questions())
    assert (d.status, d.shadow_route, d.tokens) == ("ok", "ignore", 42)
    assert d.model_version == "chat:small-model"


def test_chat_adapter_missing_key_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prov = openai_compatible.ChatCompletionsProvider("m", transport=fake_transport({}))
    with Ledger(tmp_path / "a.db") as led, Gate(base_config(), prov, led) as gate:
        d = gate.evaluate({"event": "hrv"}, questions())
    assert (d.status, d.route) == ("provider_unavailable", "agent")


def test_chat_adapter_non_object_content(tmp_path: Path) -> None:
    prov = openai_compatible.ChatCompletionsProvider(
        "m", api_key="test-key", transport=fake_transport([1, 2])
    )
    with Ledger(tmp_path / "a.db") as led, Gate(base_config(), prov, led) as gate:
        d = gate.evaluate({"event": "hrv"}, questions())
    assert d.status == "partial" and d.route == "agent"


def test_describe_questions() -> None:
    desc = openai_compatible.describe(questions())
    assert [q["id"] for q in desc] == ["material", "route"]
    assert desc[1]["options"] == list(ROUTES)


@pytest.mark.parametrize(
    "url", ["http://api.example.com/v1", "ftp://api.example.com", "api.example.com/v1"]
)
def test_chat_adapter_requires_https_for_remote_hosts(url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        openai_compatible.ChatCompletionsProvider("m", base_url=url, api_key="k")


@pytest.mark.parametrize("url", ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1"])
def test_chat_adapter_allows_plain_http_on_loopback(url: str) -> None:
    openai_compatible.ChatCompletionsProvider("m", base_url=url, api_key="k")


def test_chat_adapter_does_not_follow_redirects() -> None:
    # urllib follows a 302 and copies the Authorization header to the new
    # location, which may be another host. The example transport refuses.
    import threading
    import urllib.error
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        with pytest.raises(urllib.error.HTTPError) as refused:
            openai_compatible._urllib_transport(url, b"{}", {"Authorization": "Bearer k"}, 5.0)
        refused.value.close()  # release the response body (ResourceWarning on 3.14)
    finally:
        server.shutdown()
        server.server_close()
    assert seen == ["/v1/chat/completions"]
