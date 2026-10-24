# earned-gate

Not every model call needs to be an agent turn.

`earned-gate` puts a cheap, typed model judgment between your deterministic
code and an expensive agent: *"does this bounded change matter enough to wake
the agent at all?"*. The cheap model only gets to **skip** the agent after it
has earned that authority on your own labelled traffic, and every failure path
falls back to what you did before the gate existed.

It is a small, dependency-free Python library (stdlib + SQLite) that
implements the design in
[Not every model call needs to be an agent turn](https://gustavfredrikson.se/notes/not-every-model-call-is-an-agent-turn/).

```
world state
   ↓  explicit feature projection        (your code)
   ↓  boundary: JSON-only, secret-free, ≤ 8 KiB      → BoundaryError (a bug, raises)
   ↓  deterministic escalation                         → top route, no model call
   ↓  reuse (semantic hash, TTL, same model)           → zero tokens
   ↓  token budget                                     → fallback
   ↓  provider: one request, several typed questions   → failures become fallbacks
   ↓  policy: suppression on? allowlist? confidence? materiality? rollout gate open?
   ↓  multi-signal floor
   ↓  hash-chained SQLite ledger
cheap route  OR  full agent
```

## Three layers, not two

| Layer | Owns |
| --- | --- |
| Deterministic code | invariants, permissions, safety, freshness, writes |
| **earned-gate + a typed model** | bounded semantic classification and routing |
| Agent | open-ended synthesis, planning, tool use |

The gate must never become a security, authorization, idempotency, approval,
freshness or mutation boundary, and it cannot write anything except its own
ledger. If a route leads to a state change, put that change behind
preview/apply/undo; for MCP tools, that is
[mcp-changeset](https://github.com/GustavFredrikson/mcp-changeset).

## Quick start

```python
from earned_gate import BooleanQuestion, ChoiceQuestion, Gate, GateConfig, Ledger

ROUTES = ("ignore", "persist", "ask", "defer", "agent")  # cheapest first; last = old behaviour

config = GateConfig(
    feature="training-watch",
    routes=ROUTES,
    skippable=frozenset({"ignore", "persist"}),  # routes that end a wake without the agent
    allowlist=frozenset({"ignore", "persist"}),  # what the model may pick once trusted
    route_question="route",
    materiality_question="material",
    suppression_enabled=False,  # stage A: measure only
)
questions = {
    "material": BooleanQuestion("Could this change alter the next decision?", fallback=True),
    "route": ChoiceQuestion(
        "Which route matches the required response?", options=ROUTES, fallback="agent"
    ),
}

with Ledger("gate.db") as ledger, Gate(config, my_provider, ledger) as gate:
    decision = gate.evaluate({"trigger": "hrv_shift", "hrv_delta_pct": -3}, questions)
    if decision.skip_agent:
        ...  # cheap path
    else:
        run_agent()  # also on every failure
    gate.label(decision.eval_id, actual_route)  # what was really needed, later
```

`my_provider` is anything with a `model_version` and an
`ask(request, *, timeout) -> ProviderResponse`. Questions are typed and
bounded on purpose:

| Type | Provider returns | Value | Confidence |
| --- | --- | --- | --- |
| `BooleanQuestion` | `{"p_yes": 0.02}` | `False` | distance from 50/50: `0.96` |
| `ChoiceQuestion` | `{"probs": {"a": 0.1, "b": 0.9}}` | `"b"` | `0.9` |
| `ScoreQuestion` | `{"probs": [p1..p5]}` | `0, .25, .5, .75, 1` | top level's probability |

There is no free-text answer type. Several questions over one state go out as
**one** request; each keeps its own fallback, minimum confidence and status.

## Failure semantics

`Gate.evaluate()` raises exactly two exceptions, both integration bugs you
should fix, and both before anything is written or sent:

* `BoundaryError`: the context is not a JSON object of JSON types, has a key
  name or value that looks like a credential, is larger than
  `max_context_bytes` (or holds more than `max_context_bytes` values), is
  cyclic or nested deeper than 64 levels, or cannot be encoded as JSON (e.g.
  an integer too long for Python 3.11+'s `str()`). Any `Mapping`, not only
  `dict`, is accepted at any depth. Also raised when
  `signals` is not a mapping of string names to finite `int`/`float` values
  (`str`, `None`, `bool`, NaN, ±inf and integers too large for a float are
  rejected, because a NaN would silently disable the signal floor);
* `ConfigError`: `questions` is not a mapping of string ids to
  `BooleanQuestion` / `ChoiceQuestion` / `ScoreQuestion` (e.g. `None`, a
  list, a plain string as a question), a question's spec is not
  JSON-serialisable (e.g. non-JSON choice options), the questions do not match
  the config (route question, options, fallback, materiality question),
  `escalate` is neither a string nor `None`, the `Gate` was already closed,
  or `signals` were passed without
  a `signal_floor` or contain names the floor does not know.

`GateConfig(...)` itself raises `ConfigError` for an invalid configuration,
including wrong types (a string where a collection of route names is
expected, non-number signal-floor weights) and non-finite numbers. Lists and
sets are accepted for the route collections. A `denied_route` or
`signal_floor.raise_to` that is skippable, a negative or zero
`escalation_lookback_s`, and a negative `rollout_cache_s` are also a
`ConfigError`, because each would let a raised route still skip the agent or
silently turn a check off. One caveat: if you pass your own objects (a
`Mapping` subclass as `context`, `questions` or `signals`, an escalation rule,
a provider) whose dunder methods raise, earned-gate tries to fail safe, but it
cannot promise to wrap every error from your own code. Everything else resolves to the top route with
`skip_agent=False` and a status saying why:

* provider: `timeout`, `provider_error`, `provider_unavailable` (e.g. missing
  key, or the provider's `model_version` raised), `budget_exhausted`, and per
  question `out_of_schema` / `low_confidence`;
* `rule_error`: one of your escalation rules raised. The evaluation is
  recorded as a fallback (so it counts toward the fallback rate and can close
  the gate) but does not auto-label earlier suppressions;
* `ledger_error`: reading or writing the SQLite ledger failed (locked, full,
  closed, corrupt). Nothing is recorded, so `decision.recorded` is false and
  that `eval_id` cannot be labelled. Tokens the provider already spent still
  count against `daily_token_budget` for the next 24 hours, in memory in
  the same `Gate`.

When the provider answers but some questions fall back (`out_of_schema`,
`low_confidence`), the status is `partial`, also when every answer is out of
schema. A `partial` decision can still skip the agent: a fallen-back route
question resolves to the top route and a fallen-back materiality question
raises the route, but a fallback on any other question does not block a
cheap route. `partial` counts as a fallback in the rollout numbers and is
never reused.

`daily_token_budget` is a soft cap over a rolling 24 hours, not a calendar
day. It is checked before each provider call, so concurrent evaluations can
overshoot it, and timeouts and provider errors record 0 tokens even if the
provider billed for them.

`rule_error`, `ledger_error` and an unreadable `model_version` are also logged
as warnings on the `earned_gate` logger. Exceptions that are not `Exception`
subclasses (`KeyboardInterrupt`, `SystemExit`) are never swallowed. Outside
`evaluate()`, explicit ledger operations (`Gate.label`, `Ledger(...)`, every
public `Ledger` method, the CLI) raise `ConfigError` or `LedgerError`; SQLite
errors (locked, read-only, full, closed, corrupt) are always wrapped in
`LedgerError`, and the CLI turns any `EarnedGateError` into exit code 2
(1 when `verify` finds a broken hash chain).

A correct fallback can hide a broken integration, so **fallback rate is a
health metric**: it is part of the rollout gate, and a recent-window "provider
degraded" check closes the gate even before the 30-day numbers move.

## Authority is earned in stages

| Stage | Config | What the model may do |
| --- | --- | --- |
| A, measurement | `suppression_enabled=False` | nothing; every evaluation and its *shadow route* is logged |
| B, guarded suppression | allowlist ⊆ skippable | end a wake via the cheapest routes, while the gate holds |
| C, wider allowlist | allowlist includes a non-skippable route such as `ask` | limited, see below |

Stage C is narrow in 0.1.0. A wider allowlist only changes a decision when `denied_route`
is set above the route you add (e.g. `denied_route="defer"` with `ask` on the
allowlist); with the default `denied_route` (the cheapest non-skippable route)
the decisions are the same as in stage B and only `stage` reads "C". A
non-skippable route never sets `skip_agent`, so the agent still runs and has to
act on `route` itself. The rollout gate scores choosing `ask` where `agent` was
needed as an ordinary mismatch, not a false negative, so stage-C authority is
not earned separately.

An empty `allowlist` means "no authority yet": the gate stays in stage A and
never skips, even with `suppression_enabled=True`. Its shadow route is then
computed as if the allowlist were `skippable` (confidence, materiality and the
signal floor still apply), so stage-A traffic produces would-suppress evidence
and the rollout gate can open before you choose an allowlist.

The rollout gate fails closed. By default, within 30 days and for the
**current model version only**, it needs ≥ 30 labelled evaluations, ≥ 15 of
them would-suppress cases, **zero false negatives**, a weighted miss score ≤
0.15 (false negatives weigh 3×), a fallback rate ≤ 0.2 and a non-degraded
provider. Labels are scored against the shadow route, so stage-A traffic is
calibration data. Evidence is counted per wake, not per model call: a reused
answer is its own evaluation and can be labelled, so one call reused across
several wakes counts several times. A deterministic escalation automatically
labels earlier (would-be) suppressions in its lookback window
(`escalation_lookback_s`, default 6 h) as misses. With the default
`max_false_negatives=0`, one escalation can therefore keep the gate closed
until those misses age out of the 30-day window.
Nothing loosens a threshold automatically, and a new model version starts from
zero. Every numeric setting in `RolloutPolicy` and `GateConfig` must be a
finite number (counts must be `int`); a NaN or infinity would silently disable
the check it guards, so it is a `ConfigError`.

Scale: with suppression on, the gate reads its evidence (the window's
evaluations for the current model version and their labels, plus the newest
`degraded_window` rows) and reuses the report only while nothing that could
close the gate has been written. Before serving a cached report it checks
SQLite's `PRAGMA data_version` plus an in-process write counter: any commit
from another connection or process (such as `earned-gate label`), and any
label, fallback or escalation through the same `Ledger`, forces a fresh
assessment at once. `rollout_cache_s` (default 60 s) only bounds how long
evidence ageing out of the 30-day window can go unnoticed.
`Gate.rollout_status()` is never cached; `rollout_cache_s=0` recomputes on
every evaluation.

```console
$ earned-gate rollout gate.db --feature training-watch --model-version my-model-v1 \
    --routes ignore,persist,ask,defer,agent --skippable ignore,persist
{ "open": false, "reasons": ["labelled evaluations 12 < 30", ...], ... }   # exit 1
$ earned-gate verify gate.db
{"ok": true, "events": 4211}
```

`rollout` checks `--routes`/`--skippable` the same way `GateConfig` does
(exit 2 on a bad set) and always applies the **default** thresholds; only
`--window-days` is configurable, so a gate built with a custom
`RolloutPolicy` can disagree with it. Every CLI command needs an existing
ledger: a missing path, or a file that is
not an earned-gate ledger, exits 2 and creates nothing. In code,
`Ledger(path, create=False)` gives the same guarantee.

## Examples

* [`examples/training_watch.py`](examples/training_watch.py): a cron-style
  wake gate; exits 0 only when the caller may skip the agent, so a crash also
  means "invoke the agent".
* [`examples/openai_compatible.py`](examples/openai_compatible.py): adapter
  sketch for any OpenAI-compatible endpoint. A general LLM's self-reported
  probabilities are not calibrated, which is what stage A is for.

## What it is not

* Not a guardrail against prompt injection or a content filter. The boundary's
  secret check is a tripwire for common key/token shapes, not DLP. Key names
  are matched per segment (split on `_`, `-`, camelCase) and, like values,
  scanned for credential shapes (`ghp_…`, `sk-…`). Rejected: `token` and
  `pass` anywhere (`token_value`, `github_token_str`, `user_pass_hash`),
  qualified plurals (`github_tokens`), `secret`, `password`, `passcode`,
  `api_key`, `accessToken`, `client_secret`, `pwd`; run-together names that
  end in a long stem (`webhooksecret`, `githubtoken`) or contain `password`,
  `passwd`, `apikey`, `privatekey`, `accesskey`, `secretkey`, `authtoken` or
  `accesstoken` anywhere (`dbpasswordhash`); and `auth` /
  `oauth` / `cookie` anywhere (`auth_key`, `cookie_value`, `oauth2`) unless
  followed by a descriptive word such as `method`, `type`, `provider`,
  `status`, `required`, `accepted` or `consent`. The only allowances for
  `token`/`pass` are quantity-shaped keys: a key ending in `count`, `limit`,
  `rate`, `total`, `used`, `remaining` or `budget` (`token_count`,
  `pass_rate`), a bare `tokens`, and LLM usage counts like `max_tokens` /
  `input_tokens`, and only when the value is a number. Plurals count as
  their singular (`api_keys`, `session_ids`). Unrelated words such as
  `passenger_count`, `author`, `tokenizer`, `secretary` and
  `compass_heading` pass. Stems such as `password` and `apikey` are matched
  with separators removed, so `compass_word` and `api_keyword` are rejected
  too. Use `allow_keys` for exact names that are not credentials in your
  domain. String values that look like a URL with a password in it
  (`postgres://user:pass@host`) are rejected.
* Not a change detector for nested volatile fields. Volatile keys (`ts`,
  `time`, `request_id`, ...) are ignored at any depth
  when hashing for reuse, so two contexts that differ only in a nested `time`
  reuse the same answer within `reuse_ttl_s`. Rename meaningful fields or set
  `volatile_keys`.
* Not an approval mechanism for writes (use preview/apply/undo).
* Not an evaluation framework: it records evidence on *your* traffic and
  applies fixed thresholds; it makes no claim that one model beats another.

## Development

```console
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy && uv run pyright
uv run pytest --cov
```

MIT licensed. See [docs/design.md](docs/design.md) for decisions and open
questions.
