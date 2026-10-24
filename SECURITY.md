# Security

earned-gate is a routing aid, not a security boundary. Do not use it to decide
permissions, approvals, idempotency, data freshness or whether to write.

* The boundary rejects contexts that look like they contain credentials, but
  it is pattern-based. Send explicit projections, never raw state.
* The ledger never stores the context itself. Per evaluation it stores an
  unsalted SHA-256 of the context with volatile keys removed, mixed with the
  question specs and model version (`state_hash`,
  used for reuse; it is not a privacy measure, see below), the feature name,
  model version, status, routes, token count, and each answer's value,
  confidence and validated probability fields (`p_yes` or `probs`). Anything
  else a provider returns (reasoning text, echoed context) is dropped.
  Labels store the route and its source. Escalation reasons and error
  messages are not stored. Treat the database
  file as internal data.
* The ledger's hash chain detects accidental or naive edits only. It is
  not tamper-proof: it has no key, and anyone who can write the file can
  recompute every hash. For tamper-evidence, store `Ledger.head()` outside the
  writer's reach and pass it to `verify(expected_head=...)`.
* Context hashes are unsalted SHA-256 over small projections, so values can
  be guessed by brute force; and escalation-rule error messages are logged,
  so logs can contain text derived from the context. When the ledger write
  fails after a deterministic escalation, the escalation reason itself is
  logged too.
* Report vulnerabilities privately via GitHub security advisories on the
  repository rather than in a public issue.
