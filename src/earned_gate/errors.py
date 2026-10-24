"""Exceptions raised by earned-gate.

:meth:`earned_gate.Gate.evaluate` raises only *integration bugs*
(:class:`BoundaryError`, :class:`ConfigError`). Everything a provider can do
wrong (time out, fail, return nonsense, be unsure), a failing escalation rule
and a failing ledger resolve to a fallback with a status instead, so callers
never need a try/except around it to stay fail-safe.
"""

from __future__ import annotations


class EarnedGateError(Exception):
    """Base class for all earned-gate errors."""


class ConfigError(EarnedGateError):
    """The gate, a question or the rollout policy is misconfigured."""


class BoundaryError(EarnedGateError):
    """A request was rejected before it went anywhere near the network.

    Raised for a context that is not a JSON object, contains a value that looks
    like a secret, or is larger than the configured limit. The message names the
    offending path, never the offending value.
    """


class ProviderUnavailable(EarnedGateError):
    """A provider cannot run at all, for example because a credential is missing.

    Providers may raise this; the gate turns it into a fallback with status
    ``provider_unavailable`` rather than propagating it.
    """


class LedgerError(EarnedGateError):
    """The ledger could not be opened, read or written.

    Raised when an existing ledger was required but the path does not exist or
    is not an earned-gate ledger, when the file cannot be opened or created,
    and for every SQLite error (locked, read-only, full, closed, corrupt) from
    a public :class:`~earned_gate.Ledger` method.
    :meth:`earned_gate.Gate.evaluate` never raises it; it falls back with
    status ``ledger_error`` instead.
    """


class LedgerIntegrityError(LedgerError):
    """The hash chain of the event log does not verify."""
