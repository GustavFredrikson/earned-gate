from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from earned_gate import ConfigError, EvaluationRecord, Ledger, LedgerIntegrityError
from earned_gate.ledger import new_eval_id
from earned_gate.questions import Answer
from earned_gate.testing import FakeClock


def record(
    clock: FakeClock,
    *,
    status: str = "ok",
    shadow: str | None = "ignore",
    state_hash: str = "h1",
    tokens: int = 10,
    feature: str = "f",
    model_version: str = "m1",
) -> EvaluationRecord:
    return EvaluationRecord(
        eval_id=new_eval_id(),
        ts=clock(),
        feature=feature,
        model_version=model_version,
        state_hash=state_hash,
        status=status,
        final_route="agent",
        shadow_route=shadow,
        answers=(Answer("route", "ignore", 0.97, "ok", {"probs": {"ignore": 0.97}}),),
        tokens=tokens,
        fallback=status not in ("ok", "reused"),
        suppressed=False,
    )


def test_roundtrip_and_latest_label_wins(ledger: Ledger, clock: FakeClock) -> None:
    r = record(clock)
    ledger.record_evaluation(r)
    ledger.label(r.eval_id, "ask")
    ledger.label(r.eval_id, "agent", source="escalation")
    [got] = ledger.evaluations("f", since=0)
    assert got == r
    assert (got.label, got.label_source) == ("agent", "escalation")
    assert ledger.verify() == 3


def test_label_unknown_evaluation(ledger: Ledger) -> None:
    with pytest.raises(ConfigError):
        ledger.label("nope", "agent")


def test_filters_by_feature_version_and_time(ledger: Ledger, clock: FakeClock) -> None:
    ledger.record_evaluation(record(clock, model_version="m1"))
    clock.advance(100)
    ledger.record_evaluation(record(clock, model_version="m2"))
    ledger.record_evaluation(record(clock, feature="other"))
    assert len(ledger.evaluations("f", since=0)) == 2
    assert len(ledger.evaluations("f", since=0, model_version="m2")) == 1
    assert len(ledger.evaluations("f", since=clock() - 50)) == 1


def test_label_written_after_window_start_still_attaches(ledger: Ledger, clock: FakeClock) -> None:
    r = record(clock)
    ledger.record_evaluation(r)
    clock.advance(10_000)
    ledger.label(r.eval_id, "persist")
    [got] = ledger.evaluations("f", since=r.ts)
    assert got.label == "persist"


def test_find_reusable_only_matches_ok_within_ttl(ledger: Ledger, clock: FakeClock) -> None:
    ledger.record_evaluation(record(clock, status="timeout"))
    assert ledger.find_reusable("f", "h1", "m1", since=0) is None
    ok = record(clock)
    ledger.record_evaluation(ok)
    assert ledger.find_reusable("f", "h1", "m1", since=0) == ok
    assert ledger.find_reusable("f", "h2", "m1", since=0) is None
    assert ledger.find_reusable("f", "h1", "m2", since=0) is None
    clock.advance(1000)
    assert ledger.find_reusable("f", "h1", "m1", since=clock() - 900) is None


def test_tokens_since(ledger: Ledger, clock: FakeClock) -> None:
    ledger.record_evaluation(record(clock, tokens=40))
    clock.advance(100)
    ledger.record_evaluation(record(clock, tokens=2))
    assert ledger.tokens_since("f", 0) == 42
    assert ledger.tokens_since("f", clock() - 10) == 2


def test_label_missed_suppressions(ledger: Ledger, clock: FakeClock) -> None:
    old = record(clock, shadow="ignore")
    ledger.record_evaluation(old)
    clock.advance(10_000)
    recent_skip = record(clock, shadow="persist")
    recent_ask = record(clock, shadow="ask")
    recent_none = record(clock, shadow=None, status="timeout")
    for r in (recent_skip, recent_ask, recent_none):
        ledger.record_evaluation(r)
    ids = ledger.label_missed_suppressions(
        "f", since=clock() - 100, skippable={"ignore", "persist"}, actual_route="agent"
    )
    assert ids == [recent_skip.eval_id]
    labels = {r.eval_id: r.label for r in ledger.evaluations("f", since=0)}
    assert labels == {
        old.eval_id: None,
        recent_skip.eval_id: "agent",
        recent_ask.eval_id: None,
        recent_none.eval_id: None,
    }


def _tamper(path: Path, sql: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(sql)
    conn.commit()
    conn.close()


def test_verify_detects_edited_payload(tmp_path: Path, clock: FakeClock) -> None:
    db = tmp_path / "l.db"
    with Ledger(db, clock=clock) as led:
        for _ in range(3):
            led.record_evaluation(record(clock))
    _tamper(
        db,
        "UPDATE events SET payload = replace(payload, '\"tokens\":10', '\"tokens\":0')"
        " WHERE seq = 2",
    )
    with Ledger(db, clock=clock) as led, pytest.raises(LedgerIntegrityError, match="event 2"):
        led.verify()


def test_verify_detects_deleted_row(tmp_path: Path, clock: FakeClock) -> None:
    db = tmp_path / "l.db"
    with Ledger(db, clock=clock) as led:
        for _ in range(3):
            led.record_evaluation(record(clock))
    _tamper(db, "DELETE FROM events WHERE seq = 2")
    with Ledger(db, clock=clock) as led, pytest.raises(LedgerIntegrityError, match="prev_hash"):
        led.verify()


def test_head_anchors_against_tail_truncation(tmp_path: Path, clock: FakeClock) -> None:
    db = tmp_path / "l.db"
    with Ledger(db, clock=clock) as led:
        for _ in range(3):
            led.record_evaluation(record(clock))
        head = led.head()
    _tamper(db, "DELETE FROM events WHERE seq = 3")
    with Ledger(db, clock=clock) as led:
        assert led.verify() == 2
        with pytest.raises(LedgerIntegrityError, match="head"):
            led.verify(expected_head=head)


def test_concurrent_appends_keep_the_chain_valid(tmp_path: Path, clock: FakeClock) -> None:
    db = tmp_path / "l.db"
    ledgers = [Ledger(db, clock=clock) for _ in range(4)]

    def work(led: Ledger) -> None:
        for _ in range(25):
            led.record_evaluation(record(clock))

    threads = [threading.Thread(target=work, args=(led,)) for led in ledgers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledgers[0].verify() == 100
    for led in ledgers:
        led.close()


def test_label_missed_suppressions_skips_evaluations_already_labelled_as_misses(
    ledger: Ledger, clock: FakeClock
) -> None:
    already_agent = record(clock, shadow="ignore")
    already_ask = record(clock, shadow="persist")
    wrongly_ignore = record(clock, shadow="ignore")
    unlabelled = record(clock, shadow="ignore")
    for r in (already_agent, already_ask, wrongly_ignore, unlabelled):
        ledger.record_evaluation(r)
    ledger.label(already_agent.eval_id, "agent")
    ledger.label(already_ask.eval_id, "ask")
    ledger.label(wrongly_ignore.eval_id, "ignore")
    skippable = {"ignore", "persist"}
    ids = ledger.label_missed_suppressions("f", since=0, skippable=skippable, actual_route="agent")
    # Only evaluations not yet labelled as needing more than a skippable route.
    assert ids == [wrongly_ignore.eval_id, unlabelled.eval_id]
    # A second escalation appends nothing: every miss is already recorded.
    assert (
        ledger.label_missed_suppressions("f", since=0, skippable=skippable, actual_route="agent")
        == []
    )
    labels = {r.eval_id: r.label for r in ledger.evaluations("f", since=0)}
    assert labels[already_ask.eval_id] == "ask"
