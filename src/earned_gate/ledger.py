"""A hash-chained SQLite event log of evaluations and labels.

Every evaluation the gate makes is appended here (route, answers, model
version, tokens, status), and so is every label attached later: what the agent
actually did, or an automatic false-negative label from a deterministic
escalation. The rollout gate reads its evidence from this log only.

Each row stores the hash of the previous row, so :meth:`Ledger.verify` detects
accidental or naive edits and deletions in the middle of the log. The chain is
not tamper-proof: it has no key, so anyone who can write the file can rewrite
rows and recompute every hash. For tamper-evidence, store :meth:`Ledger.head()
<Ledger.head>` somewhere the writer cannot change and pass it to
``verify(expected_head=...)`` (or add an HMAC). The log is append-only through
this API;
a later label for the same evaluation supersedes an earlier one when evidence
is computed, but both stay in the chain.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Collection, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from .boundary import canonical_json
from .errors import ConfigError, LedgerError, LedgerIntegrityError
from .questions import Answer

GENESIS = "0" * 64

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _sqlite_errors(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Re-raise sqlite3 errors and undecodable rows from a public ledger method
    as LedgerError, so callers (and the CLI) see one exception type."""

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return fn(*args, **kwargs)
        except sqlite3.Error as exc:
            raise LedgerError(f"{fn.__name__}: {type(exc).__name__}: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:  # a row we cannot decode
            raise LedgerError(f"{fn.__name__}: unreadable ledger data: {exc!r}") from exc

    return wrapper


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    kind          TEXT    NOT NULL CHECK (kind IN ('evaluation', 'label')),
    feature       TEXT    NOT NULL,
    model_version TEXT    NOT NULL,
    eval_id       TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    prev_hash     TEXT    NOT NULL,
    hash          TEXT    NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS events_feature_ts ON events (feature, ts);
CREATE INDEX IF NOT EXISTS events_eval_id ON events (eval_id);
"""


def new_eval_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class EvaluationRecord:
    """One gate evaluation as stored in the log.

    ``shadow_route`` is the route the model would have produced with suppression
    on and the rollout gate open. Labels are scored against it, which is what
    lets measurement-stage traffic count as calibration data.
    """

    eval_id: str
    ts: float
    feature: str
    model_version: str
    state_hash: str
    status: str
    final_route: str
    shadow_route: str | None
    answers: tuple[Answer, ...]
    tokens: int
    fallback: bool
    suppressed: bool
    label: str | None = field(default=None, compare=False)
    label_source: str | None = field(default=None, compare=False)

    def payload(self) -> dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "status": self.status,
            "final_route": self.final_route,
            "shadow_route": self.shadow_route,
            "answers": [a.to_json() for a in self.answers],
            "tokens": self.tokens,
            "fallback": self.fallback,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_row(
        cls, ts: float, feature: str, model_version: str, eval_id: str, payload: dict[str, Any]
    ) -> EvaluationRecord:
        return cls(
            eval_id=eval_id,
            ts=ts,
            feature=feature,
            model_version=model_version,
            state_hash=str(payload["state_hash"]),
            status=str(payload["status"]),
            final_route=str(payload["final_route"]),
            shadow_route=payload.get("shadow_route"),
            answers=tuple(Answer.from_json(a) for a in payload["answers"]),
            tokens=int(payload["tokens"]),
            fallback=bool(payload["fallback"]),
            suppressed=bool(payload["suppressed"]),
        )


def _row_hash(
    prev_hash: str,
    ts: float,
    kind: str,
    feature: str,
    model_version: str,
    eval_id: str,
    payload: str,
) -> str:
    material = canonical_json([prev_hash, repr(ts), kind, feature, model_version, eval_id, payload])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Ledger:
    """SQLite-backed event log. Safe to share between threads and processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        create: bool = True,
    ) -> None:
        """Open (and by default create) the ledger at ``path``.

        With ``create=False`` the file must already exist and contain an
        earned-gate ledger; otherwise :class:`LedgerError` is raised and nothing
        is created or modified. Use it for tools that only inspect or label an
        existing ledger, so a typo in the path cannot silently "pass".
        """
        self.path = str(path)
        self._clock = clock
        self._lock = threading.Lock()
        self._evidence_writes = 0
        if create:
            try:
                self._conn = sqlite3.connect(
                    self.path, isolation_level=None, check_same_thread=False
                )
                self._conn.execute("PRAGMA busy_timeout = 5000")
                if self.path != ":memory:":
                    self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.executescript(_SCHEMA)
            except sqlite3.Error as exc:
                raise LedgerError(f"cannot open ledger {self.path!r}: {exc}") from exc
        else:
            self._conn = self._open_existing(Path(path))

    @staticmethod
    def _open_existing(path: Path) -> sqlite3.Connection:
        if not path.is_file():
            raise LedgerError(f"no ledger at {str(path)!r}")
        # mode=rw never creates the file, unlike a plain connect().
        uri = path.resolve().as_uri() + "?mode=rw"
        try:
            conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - raced with a delete
            raise LedgerError(f"cannot open {str(path)!r}: {exc}") from exc
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'"
            ).fetchone()
        except sqlite3.Error as exc:
            conn.close()
            raise LedgerError(f"cannot read {str(path)!r}: {exc}") from exc
        if row is None:
            conn.close()
            raise LedgerError(f"{str(path)!r} is not an earned-gate ledger (no events table)")
        return conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def now(self) -> float:
        return self._clock()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:  # COMMIT itself may have failed
                    self._conn.execute("ROLLBACK")
                raise

    def _append(
        self,
        conn: sqlite3.Connection,
        *,
        ts: float,
        kind: str,
        feature: str,
        model_version: str,
        eval_id: str,
        payload: dict[str, Any],
    ) -> None:
        # The ts column is REAL, so an int clock value is read back as a float;
        # hash the float so verify() recomputes the same digest.
        ts = float(ts)
        row = conn.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        prev = str(row[0]) if row else GENESIS
        body = canonical_json(payload)
        digest = _row_hash(prev, ts, kind, feature, model_version, eval_id, body)
        conn.execute(
            "INSERT INTO events"
            " (ts, kind, feature, model_version, eval_id, payload, prev_hash, hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, kind, feature, model_version, eval_id, body, prev, digest),
        )

    # -- writes -----------------------------------------------------------------

    @_sqlite_errors
    def change_token(self) -> tuple[int, int]:
        """A cheap token that changes whenever rollout evidence may have changed.

        It combines SQLite's ``PRAGMA data_version`` (which changes when *any
        other* connection, in this process or another, commits) with a counter
        of this connection's own label, fallback and escalation writes. Two
        equal tokens mean nothing that could close the rollout gate was written
        in between. Used by :class:`~earned_gate.Gate` to validate its cached
        rollout report.
        """
        with self._lock:
            row = self._conn.execute("PRAGMA data_version").fetchone()
            return int(row[0]), self._evidence_writes

    @_sqlite_errors
    def record_evaluation(self, record: EvaluationRecord) -> None:
        with self._tx() as conn:
            self._append(
                conn,
                ts=record.ts,
                kind="evaluation",
                feature=record.feature,
                model_version=record.model_version,
                eval_id=record.eval_id,
                payload=record.payload(),
            )
            if record.fallback or record.status == "escalated":
                self._evidence_writes += 1

    @_sqlite_errors
    def label(self, eval_id: str, actual_route: str, *, source: str = "agent") -> None:
        """Attach what actually happened (the route that was really needed)."""
        with self._tx() as conn:
            row = conn.execute(
                "SELECT feature, model_version FROM events WHERE kind = 'evaluation'"
                " AND eval_id = ?",
                (eval_id,),
            ).fetchone()
            if row is None:
                raise ConfigError(f"unknown evaluation {eval_id!r}")
            self._append(
                conn,
                ts=self._clock(),
                kind="label",
                feature=str(row[0]),
                model_version=str(row[1]),
                eval_id=eval_id,
                payload={"actual_route": actual_route, "source": source},
            )
            self._evidence_writes += 1

    @_sqlite_errors
    def label_missed_suppressions(
        self,
        feature: str,
        *,
        since: float,
        skippable: Collection[str],
        actual_route: str,
        source: str = "escalation",
    ) -> list[str]:
        """Label every evaluation since ``since`` whose shadow route would have
        skipped the agent as needing ``actual_route``.

        Called when a deterministic escalation fires: whatever was suppressed (or
        would have been) in the lookback window is evidence of a miss.
        Evaluations whose latest label is already non-skippable are left alone:
        they already count as a miss, so relabelling them would only append
        duplicates. A skippable label (someone judged the wake unnecessary) is
        overridden, which is the conservative choice.
        """
        labelled: list[str] = []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT eval_id, model_version, payload FROM events WHERE kind = 'evaluation'"
                " AND feature = ? AND ts >= ? ORDER BY seq",
                (feature, since),
            ).fetchall()
            latest: dict[str, str] = {}
            for eval_id, body in conn.execute(
                "SELECT eval_id, payload FROM events WHERE kind = 'label' AND feature = ?"
                " AND eval_id IN (SELECT eval_id FROM events WHERE kind = 'evaluation'"
                " AND feature = ? AND ts >= ?) ORDER BY seq",
                (feature, feature, since),
            ):
                latest[str(eval_id)] = str(json.loads(body)["actual_route"])
            now = self._clock()
            for eval_id, model_version, body in rows:
                shadow = json.loads(body).get("shadow_route")
                prior = latest.get(str(eval_id))
                if shadow in skippable and (prior is None or prior in skippable):
                    self._append(
                        conn,
                        ts=now,
                        kind="label",
                        feature=feature,
                        model_version=str(model_version),
                        eval_id=str(eval_id),
                        payload={"actual_route": actual_route, "source": source},
                    )
                    labelled.append(str(eval_id))
            if labelled:
                self._evidence_writes += 1
        return labelled

    # -- reads ------------------------------------------------------------------

    @_sqlite_errors
    def evaluations(
        self, feature: str, *, since: float, model_version: str | None = None
    ) -> list[EvaluationRecord]:
        """Evaluations since ``since`` (oldest first), each with its latest label."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, model_version, eval_id, payload FROM events"
                " WHERE kind = 'evaluation' AND feature = ? AND ts >= ?"
                " AND (? IS NULL OR model_version = ?) ORDER BY seq",
                (feature, since, model_version, model_version),
            ).fetchall()
        # Labels can be written long after their evaluation, so they are fetched
        # by evaluation id rather than by their own timestamp.
        records: dict[str, EvaluationRecord] = {}
        for ts, kind, mv, eval_id, body in rows:
            if kind == "evaluation":
                records[eval_id] = EvaluationRecord.from_row(
                    ts, feature, mv, eval_id, json.loads(body)
                )
        if records:
            with self._lock:
                label_rows = self._conn.execute(
                    "SELECT eval_id, payload FROM events WHERE kind = 'label' AND feature = ?"
                    " AND eval_id IN (SELECT eval_id FROM events WHERE kind = 'evaluation'"
                    " AND feature = ? AND ts >= ?) ORDER BY seq",
                    (feature, feature, since),
                ).fetchall()
            for eval_id, body in label_rows:
                if eval_id in records:
                    data = json.loads(body)
                    records[eval_id] = replace(
                        records[eval_id],
                        label=str(data["actual_route"]),
                        label_source=str(data["source"]),
                    )
        return list(records.values())

    @_sqlite_errors
    def recent_fallbacks(self, feature: str, *, since: float, limit: int) -> list[bool]:
        """``fallback`` flags of the newest ``limit`` provider-asked evaluations
        (any model version, deterministic escalations excluded), newest first."""
        out: list[bool] = []
        with self._lock:
            cursor = self._conn.execute(
                "SELECT payload FROM events WHERE kind = 'evaluation' AND feature = ?"
                " AND ts >= ? ORDER BY seq DESC",
                (feature, since),
            )
            for (body,) in cursor:
                data = json.loads(body)
                if data["status"] != "escalated":
                    out.append(bool(data["fallback"]))
                    if len(out) >= limit:
                        break
        return out

    @_sqlite_errors
    def find_reusable(
        self, feature: str, state_hash: str, model_version: str, *, since: float
    ) -> EvaluationRecord | None:
        """Most recent provider-backed evaluation with the same semantic state."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, eval_id, payload FROM events WHERE kind = 'evaluation'"
                " AND feature = ? AND model_version = ? AND ts >= ? ORDER BY seq DESC",
                (feature, model_version, since),
            ).fetchall()
        for ts, eval_id, body in rows:
            data = json.loads(body)
            if data["state_hash"] == state_hash and data["status"] == "ok":
                return EvaluationRecord.from_row(ts, feature, model_version, eval_id, data)
        return None

    @_sqlite_errors
    def tokens_since(self, feature: str, since: float) -> int:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM events WHERE kind = 'evaluation' AND feature = ? AND ts >= ?",
                (feature, since),
            ).fetchall()
        return sum(int(json.loads(body)["tokens"]) for (body,) in rows)

    @_sqlite_errors
    def head(self) -> str:
        """Hash of the latest event. Store it elsewhere to detect tail truncation."""
        with self._lock:
            row = self._conn.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return str(row[0]) if row else GENESIS

    @_sqlite_errors
    def verify(self, *, expected_head: str | None = None) -> int:
        """Recompute the hash chain. Returns the number of events checked.

        A hash chain alone cannot detect deletion of the newest events; pass a
        previously stored :meth:`head` as ``expected_head`` to check that too.
        """
        prev = GENESIS
        count = 0
        found = False
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, kind, feature, model_version, eval_id, payload, prev_hash, hash"
                " FROM events ORDER BY seq"
            ).fetchall()
        for seq, ts, kind, feature, mv, eval_id, body, prev_hash, digest in rows:
            if prev_hash != prev:
                raise LedgerIntegrityError(f"event {seq}: prev_hash does not match event {seq - 1}")
            if _row_hash(prev, ts, kind, feature, mv, eval_id, body) != digest:
                raise LedgerIntegrityError(f"event {seq}: hash does not match its contents")
            prev = digest
            count += 1
            found = found or digest == expected_head
        if expected_head is not None and expected_head != GENESIS and not found:
            raise LedgerIntegrityError("expected head not found: the log was truncated or replaced")
        return count
