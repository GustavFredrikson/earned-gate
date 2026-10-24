from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import answers, base_config, seed_evidence

from earned_gate import Gate, Ledger
from earned_gate.cli import main
from earned_gate.testing import FakeClock, ScriptedProvider

ROLLOUT = [
    "--feature",
    "training-watch",
    "--model-version",
    "scripted-1",
    "--routes",
    "ignore,persist,ask,defer,agent",
    "--skippable",
    "ignore,persist",
]


@pytest.fixture
def db(tmp_path: Path) -> tuple[Path, list[str]]:
    import time

    path = tmp_path / "gate.db"
    clock = FakeClock(now=time.time() - 3600)
    with (
        Ledger(path, clock=clock) as led,
        Gate(base_config(), ScriptedProvider(answers=answers()), led) as g,
    ):
        ids = seed_evidence(g, clock, 30)
    return path, ids


def test_verify(db: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", str(db[0])]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "events": 60}


def test_rollout_exit_codes(db: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rollout", str(db[0]), *ROLLOUT]) == 0
    assert json.loads(capsys.readouterr().out)["open"] is True
    assert main(["label", str(db[0]), db[1][0], "agent"]) == 0
    capsys.readouterr()
    assert main(["rollout", str(db[0]), *ROLLOUT]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["false_negatives"] == 1


def test_errors_exit_2(db: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["label", str(db[0]), "nope", "agent"]) == 2
    assert "unknown evaluation" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["verify", "rollout", "label"])
def test_missing_ledger_fails_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cmd: str
) -> None:
    missing = tmp_path / "typo.db"
    argv = {
        "verify": ["verify", str(missing)],
        "rollout": ["rollout", str(missing), *ROLLOUT],
        "label": ["label", str(missing), "some-id", "agent"],
    }[cmd]
    assert main(argv) == 2
    assert "no ledger" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_non_ledger_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = tmp_path / "other.db"
    other.write_bytes(b"this is not sqlite at all, just some bytes" * 4)
    assert main(["verify", str(other)]) == 2
    assert "LedgerError" in capsys.readouterr().err


def test_sqlite_file_without_events_table_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3

    other = tmp_path / "other.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE t (x)")
    conn.close()
    assert main(["verify", str(other)]) == 2
    assert "not an earned-gate ledger" in capsys.readouterr().err
    conn = sqlite3.connect(other)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert tables == {"t"}


@pytest.mark.parametrize(
    ("routes", "skippable", "match"),
    [
        ("ignore,persist,ask,defer,agent", "ignore,agent", "top route"),
        ("ignore,persist,ask,defer,agent", "ignore,nope", "unknown"),
        ("ignore,ignore,agent", "ignore", "unique"),
        ("agent", "", "at least two"),
        ("ignore,persist,ask,defer,agent", "", "skippable"),
    ],
)
def test_rollout_rejects_bad_routes(
    db: tuple[Path, list[str]],
    capsys: pytest.CaptureFixture[str],
    routes: str,
    skippable: str,
    match: str,
) -> None:
    argv = ["rollout", str(db[0]), "--feature", "training-watch", "--model-version"]
    argv += ["scripted-1", "--routes", routes, "--skippable", skippable]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "ConfigError" in err and match in err
