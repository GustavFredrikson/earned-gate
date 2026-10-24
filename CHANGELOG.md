# Changelog

## 0.1.0 (unreleased)

First release.

* Typed questions (`BooleanQuestion`, `ChoiceQuestion`, `ScoreQuestion`) with
  per-question fallback and minimum confidence; several questions over one
  state go out as one provider request.
* Request boundary: contexts must be JSON-only, at most `max_context_bytes`
  (default 8,192) and free of credential-shaped key names and values.
  Violations raise `BoundaryError`.
* Gate flow: deterministic escalation, reuse by semantic hash (re-validated
  under the current questions and config), rolling 24-hour token budget,
  provider call with timeout, policy (allowlist, confidence, materiality,
  rollout gate), optional multi-signal floor.
* Failure semantics: `evaluate()` raises only `BoundaryError` and
  `ConfigError`; provider, rule and ledger failures resolve to the top route
  with a status (`timeout`, `provider_error`, `provider_unavailable`,
  `budget_exhausted`, `rule_error`, `ledger_error`, `partial`).
* Hash-chained SQLite ledger (WAL) of evaluations, labels and escalations,
  with `verify(expected_head=...)`. It stores hashes and validated answers,
  never the context.
* Fail-closed rollout gate per model version: minimum labelled and
  would-suppress counts, zero false negatives, weighted miss score, fallback
  rate and a recent-window degraded-provider check. Escalations auto-label
  earlier suppressions as misses.
* Measurement stage: with suppression off or an empty allowlist the gate
  always returns the top route and logs a shadow route as evidence.
* Rollout reports are cached briefly and revalidated before each use, so a
  label written from another connection or the CLI closes the gate at once.
* `earned-gate` CLI: `verify`, `rollout` and `label` on an existing ledger
  (`verify` exits 1 on a broken hash chain, 2 on a bad path or ledger).
* Examples: a cron-style wake gate and an OpenAI-compatible adapter sketch.
