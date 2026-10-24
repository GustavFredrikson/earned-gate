# Design notes

Source: the note
[*Not every model call needs to be an agent turn*](https://gustavfredrikson.se/notes/not-every-model-call-is-an-agent-turn/)
(2026-09-02), which describes a typed-question gate in front of a
training-coach poll. earned-gate is a provider-agnostic, open implementation
of that design. Where the note is
silent, the choices below are interpretations and are marked as such.

## Taken directly from the note

* Three question types (boolean / choice / score), no generated text.
* Boundary order: projection, secret-free check, size check (8,192 bytes),
  then the model, then deterministic policy.
* Malformed requests raise; provider failures fall back with a status.
* Every request carries a fallback, which is the pre-gate behaviour.
* Gate order: deterministic escalation, reuse by semantic hash ignoring
  timestamps (re-resolved under current config: stored answers are
  re-validated against the current questions, including `min_confidence`, and
  the policy runs with the current config), model, policy (suppression
  on, allowlist, confidence ≥ 0.90, rollout gate open), multi-signal floor.
* Routes ordered cheapest first; in the note's configuration only the two
  cheapest skip the agent. The library lets any non-top route be skippable.
* Hash-chained event log with route, probabilities, model version, tokens.
* Escalations auto-label earlier suppressions as false negatives.
* Rollout gate thresholds: 30 days, current model version, ≥ 30 labelled,
  ≥ 15 would-suppress, 0 FN, weighted miss ≤ 0.15 with FN ×3, fallback rate
  ≤ 0.2, provider not degraded. Labels scored against the shadow route.
* Several questions over one shared state in one request.

## Interpretations (not specified in the note)

* **Miss score**: `(3·FN + other mismatches) / labelled`, where a mismatch is
  any labelled evaluation whose shadow route differs from its label.
* **Degraded provider**: fallback rate over the last 20 asked evaluations of
  the feature (any model version) > 0.5.
* **Materiality question**: if it answers "yes" (or falls back), a cheap route
  is raised to the denied route. The note asks the question but does not say
  how it combines with the route answer.
* **Denied route** defaults to the first non-skippable route ("ask a fixed
  question"), matching "raised to at least ask". An explicit `denied_route`
  (and `signal_floor.raise_to`) must not be skippable, or a raised answer
  would still skip the agent.
* **Stage C** (an allowlist wider than `skippable`) is kept minimal in 0.1.0:
  with the default denied route it changes no decision, and the rollout gate
  does not score its extra authority separately.
* **Stage A** returns the top route (agent) for every evaluation, so the
  caller's behaviour is unchanged while evidence accumulates.
* **Empty allowlist** is stage A whatever `suppression_enabled` says, and the
  shadow route uses `skippable` as its allowlist. Otherwise every shadow route
  would be raised to the denied route, would-suppress would stay at 0 and the
  rollout gate could never open from measurement traffic.
* **Token budget** (`daily_token_budget`) is an addition; the note mentions
  token cost only as a consequence. It is a soft cap over a rolling 24 hours,
  checked before each call (concurrent calls can overshoot), and timeouts or
  provider errors record 0 tokens.
* **Partial** status: the provider answered (or a stored answer was reused)
  but at least one question fell back, including the case where every answer
  is out of schema. It counts toward the fallback rate, is never reused, and
  can still skip the agent when the route and materiality answers resolved.
* **Escalation reasons are not stored** in the ledger, only the fact that an
  evaluation escalated. A reason string is free text from the caller and may
  be derived from the context, which the ledger never stores. Auditing an
  escalation means looking at the caller's own logs.
* **Fallback rate denominator**: every non-escalation evaluation, including
  reused answers and `budget_exhausted` / `rule_error` rows where the
  provider was never called. An exhausted budget therefore also counts as a
  fallback and can trip the "provider degraded" check; that is conservative.
* Escalations are excluded from the fallback rate and from evidence counts.
* **Evidence is per wake**: a reused answer is a separate evaluation that can
  be labelled, so one model call reused across several wakes counts several
  times toward the minimums. Each wake is a real routing decision, so this is
  defensible, but a feature with heavy reuse reaches the minimums with fewer
  distinct model answers. Counting distinct `state_hash` values is an
  alternative left open.
* **Escalation auto-labelling** labels every shadow-skippable evaluation in
  the lookback window (default 6 h, any model version) as needing the top
  route, unless its latest label is already non-skippable (it already counts
  as a miss, so no duplicate is appended). It does override a skippable label
  an agent gave earlier: conservative, and in line with the note.
* **Failure semantics beyond the provider** (the note only covers provider
  failures): an escalation rule that raises escalates to the top route with
  status `rule_error`, is recorded as a fallback and does not auto-label; a
  ledger read/write error returns the top route with status `ledger_error` and
  is not recorded. Both are logged. Fail-safe beats visibility here because a
  caller that crashes on a locked DB loses the wake entirely.

## Open questions

* Async API (`async def evaluate`). Today: call from `asyncio.to_thread`.
* A timed-out provider thread keeps running until it returns; Python cannot
  kill threads. Providers should honour their `timeout` argument.
* The hash chain is not tamper-proof: it detects accidental or naive edits,
  but anyone who can write the file can recompute every hash, and tail
  truncation is invisible to the chain. Both need an externally stored
  `head()` passed to `verify(expected_head=...)`, or an HMAC.
* Should labels be restricted to known routes at the ledger level (today the
  `Gate.label` API checks, the raw `Ledger.label` does not)?
