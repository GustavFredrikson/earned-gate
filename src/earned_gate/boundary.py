"""The request boundary: projection, secret-free check and size check.

Callers are expected to send an explicit *projection* of world state (only the
fields the question needs), not raw state. This module enforces what it can
about that projection before anything leaves the process:

* the context is a JSON object built from JSON types only;
* no key name (matched per segment) or string value looks like a credential;
* it is acyclic and nested at most :data:`MAX_DEPTH` levels deep;
* the canonical encoding fits in ``max_bytes``.

A violation raises :class:`~earned_gate.errors.BoundaryError`, because it is an
integration bug the caller should see, not a provider failure to paper over.
The secret check is a tripwire, not a DLP system: it catches the common shapes
of keys and tokens, and the README says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any, cast

from .errors import BoundaryError

DEFAULT_MAX_BYTES = 8192

#: Maximum nesting of objects/arrays in a context (the root object is level 1).
MAX_DEPTH = 64

# Key names are matched per *segment*, not as substrings, so ``passenger_count``,
# ``author`` or ``tokenizer`` pass while ``db_pass``, ``authToken`` or ``api_key``
# do not. A key is split on any non-alphanumeric character and on camelCase
# boundaries (``XAuthToken`` -> ``x``, ``auth``, ``token``), then lower-cased.

#: Segments that mark a credential wherever they appear in a key.
SECRET_KEY_SEGMENTS = frozenset(
    {
        "password",
        "passwd",
        "passphrase",
        "passcode",
        "passkey",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "bearer",
        "authorization",
        # common run-together spellings
        "apikey",
        "accesskey",
        "privatekey",
        "sessionid",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "secretkey",
        "authkey",
        "dbpass",
        "dbpwd",
        "pwd",
    }
)

#: Long, unambiguous stems that mark a credential when a single segment *ends*
#: with them, which catches run-together names (``dbpassword``,
#: ``webhooksecret``, ``apisecret``).
SECRET_SEGMENT_SUFFIXES = (
    "password",
    "passwd",
    "passphrase",
    "passcode",
    "passkey",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "apikey",
    "authkey",
    "accesskey",
    "secretkey",
    "authtoken",
    "accesstoken",
    "refreshtoken",
    "privatekey",
    "bearer",
)

#: Stems that mark a credential when a segment *starts* with them
#: (``passwordhash``). ``secret`` is not here, so ``secretary`` passes.
SECRET_SEGMENT_PREFIXES = ("password", "passwd", "passphrase", "passcode", "apikey", "privatekey")

#: ``token`` and ``pass`` mark a credential anywhere in a key (``token_value``,
#: ``github_token_str``, ``user_pass_hash``) unless the key ends in one of
#: :data:`QUANTITY_SUFFIXES` (``token_count``, ``pass_rate``).
SECRET_KEY_WORDS = frozenset({"token", "pass"})

#: Last segments that make a ``token``/``pass``/``tokens`` key a quantity, not a
#: credential. This is the only allowance for those words; extend it with
#: ``allow_keys`` rather than here.
QUANTITY_SUFFIXES = frozenset(
    {"count", "counts", "limit", "limits", "rate", "rates", "total", "used", "remaining", "budget"}
)

#: ``tokens`` (plural) is a credential with any qualifier (``github_tokens``,
#: ``slack_tokens``, ``tokens_value``) except: the bare key ``tokens``, a key
#: ending in a quantity suffix (``tokens_used``), or a key ending in
#: ``<prefix>_tokens`` with one of these LLM usage/limit prefixes
#: (``max_tokens``, ``input_tokens``, ``completion_tokens``).
TOKEN_COUNT_PREFIXES = frozenset(
    {"max", "min", "num", "total", "input", "output", "prompt", "completion", "cached", "reasoning"}
)

#: ``auth``/``cookie`` words mark a credential anywhere in a key (``auth_key``,
#: ``cookie_value``, ``x_auth``, ``oauth2``) unless the next non-numeric
#: segment is one of :data:`AUTH_SAFE_NEXT` (``auth_method``,
#: ``oauth2_provider``, ``cookies_accepted``).
AUTH_WORDS = frozenset({"auth", "oauth", "cookie", "cookies"})
AUTH_SAFE_NEXT = frozenset(
    {
        "method",
        "methods",
        "type",
        "types",
        "mode",
        "provider",
        "providers",
        "status",
        "required",
        "enabled",
        "disabled",
        "accepted",
        "consent",
        "scheme",
        "flow",
        "policy",
        "domain",
        "url",
        "uri",
        "endpoint",
    }
)

#: Adjacent segment pairs that mark a credential anywhere in a key.
SECRET_KEY_PAIRS = frozenset(
    {
        ("api", "key"),
        ("access", "key"),
        ("private", "key"),
        ("signing", "key"),
        ("encryption", "key"),
        ("session", "id"),
        ("session", "token"),
        ("auth", "header"),
    }
)

_SEGMENT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def key_segments(key: str) -> list[str]:
    """Split a key name into lower-case segments (``"X-AuthToken"`` -> x, auth, token)."""
    return [m.group(0).lower() for m in _SEGMENT_RE.finditer(key)]


def _is_token_word(seg: str) -> bool:
    return seg == "token" or (seg.endswith("token") and len(seg) > len("token"))


def _is_qualified_tokens(seg: str, segs: list[str], i: int) -> bool:
    if seg == "tokens":
        if len(segs) == 1:
            return False  # a bare "tokens" is almost always a count
        last_usage = i == len(segs) - 1 and i > 0 and segs[i - 1] in TOKEN_COUNT_PREFIXES
        return not last_usage
    # run-together (precondition: seg ends with "tokens"): githubtokens vs maxtokens
    return seg[: -len("tokens")] not in TOKEN_COUNT_PREFIXES


def _auth_word_is_credential(segs: list[str], i: int) -> bool:
    following = [s for s in segs[i + 1 :] if not s.isdigit()]
    return not following or following[0] not in AUTH_SAFE_NEXT


#: Matched anywhere in the key with separators removed (``dbpasswordhash``).
#: ``secret`` is left out on purpose: it would flag ``secretary``.
SECRET_SUBSTRINGS = (
    "password",
    "passwd",
    "apikey",
    "privatekey",
    "accesskey",
    "secretkey",
    "authtoken",
    "accesstoken",
)


def _singular(seg: str) -> str:
    return seg[:-1] if len(seg) > 2 and seg.endswith("s") and not seg.endswith("ss") else seg


def looks_like_secret_key(key: str, *, allow_quantities: bool = True) -> bool:
    """True if a key name looks like it holds a credential (segment-aware).

    With ``allow_quantities=False`` the quantity allowances (a quantity
    suffix, a bare ``tokens``, LLM usage counts such as ``max_tokens``) are
    ignored; :func:`validate_context` uses that for keys whose value is not a
    number. See :data:`SECRET_KEY_SEGMENTS`, :data:`SECRET_SEGMENT_SUFFIXES`,
    :data:`SECRET_SEGMENT_PREFIXES`, :data:`SECRET_SUBSTRINGS`, :data:`SECRET_KEY_WORDS`,
    :data:`QUANTITY_SUFFIXES`, :data:`TOKEN_COUNT_PREFIXES`,
    :data:`AUTH_WORDS` and :data:`SECRET_KEY_PAIRS`.
    """
    segs = key_segments(key)
    if not segs:
        return False
    joined = "".join(segs)
    if any(stem in joined for stem in SECRET_SUBSTRINGS):
        return True
    for s in segs:
        if (
            s in SECRET_KEY_SEGMENTS
            or s.endswith(SECRET_SEGMENT_SUFFIXES)
            or s.startswith(SECRET_SEGMENT_PREFIXES)
        ):
            return True
    quantity = allow_quantities and len(segs) > 1 and segs[-1] in QUANTITY_SUFFIXES
    if not quantity:
        for i, s in enumerate(segs):
            if s in SECRET_KEY_WORDS or _is_token_word(s):
                return True
            if s.endswith("tokens") and (not allow_quantities or _is_qualified_tokens(s, segs, i)):
                return True
    if any(s in AUTH_WORDS and _auth_word_is_credential(segs, i) for i, s in enumerate(segs)):
        return True
    singular = [_singular(s) for s in segs]  # api_keys, session_ids
    return any(pair in SECRET_KEY_PAIRS for pair in pairwise(singular))


#: Value shapes that look like credentials.
SECRET_VALUE_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_\w{40,}")),
    ("AWS access key id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("bearer header", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}")),
    ("URL with credentials", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")),
)


def _plain(value: object) -> object:
    # json only encodes dict; any other Mapping (MappingProxyType, ChainMap, a
    # custom Mapping) is encoded as the equivalent dict.
    if isinstance(value, Mapping):
        mapping: Mapping[Any, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return dict(mapping)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used for size checks and hashing."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_plain
    )


def _walk(value: Any, path: str, max_nodes: int) -> Iterable[tuple[str, Any]]:
    """Yield ``(path, value)`` for every node, iteratively.

    Raises :class:`BoundaryError` for a cycle, for containers nested deeper
    than :data:`MAX_DEPTH` (instead of hitting Python's recursion limit), and
    after ``max_nodes`` values: shared references can make a small object
    graph expand to millions of nodes before the size check runs.
    """
    seen = 0
    # Stack of (path, value, depth, ids of the containers above this node).
    stack: list[tuple[str, Any, int, frozenset[int]]] = [(path, value, 1, frozenset())]
    while stack:
        p, v, depth, above = stack.pop()
        seen += 1
        if seen > max_nodes:
            raise BoundaryError(f"context has more than {max_nodes} values")
        yield p, v
        if not isinstance(v, (Mapping, list)):
            continue
        node = id(cast(object, v))
        if node in above:
            raise BoundaryError(f"{p}: context is cyclic")
        if depth > MAX_DEPTH:
            raise BoundaryError(f"{p}: context is nested deeper than {MAX_DEPTH} levels")
        here = above | {node}
        children: list[tuple[str, Any, int, frozenset[int]]] = []
        if isinstance(v, Mapping):
            mapping: Mapping[Any, Any] = v  # pyright: ignore[reportUnknownVariableType]
            children = [(f"{p}.{k}", c, depth + 1, here) for k, c in mapping.items()]
        else:
            seq: list[Any] = v  # pyright: ignore[reportUnknownVariableType]
            children = [(f"{p}[{i}]", c, depth + 1, here) for i, c in enumerate(seq)]
        stack.extend(reversed(children))


def validate_context(
    context: object,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_keys: frozenset[str] = frozenset(),
) -> str:
    """Validate a projected context and return its canonical JSON encoding.

    ``allow_keys`` exempts exact key names from the key-name check (for example
    ``"session_id"`` when it is not a credential in your domain); values under
    them are still scanned. Key names are matched per segment, see
    :func:`looks_like_secret_key`.
    """
    if not isinstance(context, Mapping):
        raise BoundaryError("context must be a JSON object (a projection), not raw state")
    for path, value in _walk(context, "$", max_nodes=max_bytes):
        if isinstance(value, Mapping):
            mapping: Mapping[Any, Any] = value  # pyright: ignore[reportUnknownVariableType]
            for k in mapping:
                if not isinstance(k, str):
                    raise BoundaryError(f"{path}: object keys must be strings")
                for label, rx in SECRET_VALUE_RES:
                    if rx.search(k):  # never echo the key itself
                        raise BoundaryError(f"{path}: a key looks like a {label}")
                if k in allow_keys:
                    continue
                if looks_like_secret_key(k):
                    raise BoundaryError(f"{path}.{k}: key name looks like a credential")
                child = mapping[k]
                is_number = isinstance(child, (int, float)) and not isinstance(child, bool)
                if not is_number and looks_like_secret_key(k, allow_quantities=False):
                    raise BoundaryError(
                        f"{path}.{k}: a quantity-shaped token/pass key must hold a number"
                    )
        elif isinstance(value, str):
            for label, rx in SECRET_VALUE_RES:
                if rx.search(value):
                    raise BoundaryError(f"{path}: value looks like a {label}")
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise BoundaryError(f"{path}: NaN/Infinity is not JSON")
        elif value is None or isinstance(value, (bool, int, list)):
            pass
        else:
            raise BoundaryError(f"{path}: {type(value).__name__} is not a JSON type")
    try:
        encoded = canonical_json(context)
        size = len(encoded.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        # e.g. an int too long for str() on Python 3.11+, or a lone surrogate
        # (UnicodeEncodeError) from surrogateescape-decoded input
        raise BoundaryError(f"context cannot be encoded as JSON: {type(exc).__name__}") from exc
    if size > max_bytes:
        raise BoundaryError(f"context is {size} bytes, limit is {max_bytes}")
    return encoded


def validate_signals(signals: object) -> dict[str, float]:
    """Validate the ``signals`` passed to :meth:`~earned_gate.Gate.evaluate`.

    ``None`` means no signals. Otherwise it must be a mapping of names to
    finite ``int``/``float`` values (not ``bool``). A NaN would make the
    multi-signal floor's ``score >= threshold`` silently false, so anything
    else raises :class:`BoundaryError`.
    """
    if signals is None:
        return {}
    if not isinstance(signals, Mapping):
        raise BoundaryError("signals must be a mapping of names to numbers")
    mapping: Mapping[Any, Any] = signals  # pyright: ignore[reportUnknownVariableType]
    out: dict[str, float] = {}
    for name, value in mapping.items():
        if not isinstance(name, str):
            raise BoundaryError("signals: names must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BoundaryError(
                f"signals.{name}: expected a finite number, got {type(value).__name__}"
            )
        try:
            number = float(value)
        except OverflowError:
            raise BoundaryError(f"signals.{name}: number is too large") from None
        except (TypeError, ValueError):  # an int/float subclass whose __float__ raises
            raise BoundaryError(f"signals.{name}: not convertible to a float") from None
        if not math.isfinite(number):
            raise BoundaryError(f"signals.{name}: NaN/Infinity is not allowed")
        out[name] = number
    return out


def _strip(value: Any, volatile: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        mapping: Mapping[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return {k: _strip(v, volatile) for k, v in mapping.items() if k not in volatile}
    if isinstance(value, list):
        seq: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
        return [_strip(v, volatile) for v in seq]
    return value


def semantic_hash(context: Mapping[str, Any], volatile_keys: frozenset[str], *extra: Any) -> str:
    """Hash of the context with volatile keys (timestamps, ids) removed at any depth.

    At *any* depth: ``{"race": {"time": "2:59"}}`` and ``{"race": {"time": "4:30"}}``
    hash the same, so the second reuses the first answer within ``reuse_ttl_s``.
    Do not put meaningful data under a volatile key name.

    ``extra`` is mixed in so that, for example, changing a question's prompt or
    the model version invalidates reuse.
    """
    payload = [_strip(context, volatile_keys), *extra]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
