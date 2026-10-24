from __future__ import annotations

import pytest

from earned_gate import BoundaryError, semantic_hash, validate_context

VOLATILE = frozenset({"ts", "observed_at"})


def test_valid_projection_returns_canonical_json() -> None:
    out = validate_context({"b": 1, "a": [True, None, 1.5, "x"]})
    assert out == '{"a":[true,null,1.5,"x"],"b":1}'


@pytest.mark.parametrize("ctx", [[1, 2], "state", None, 3])
def test_non_object_context_is_rejected(ctx: object) -> None:
    with pytest.raises(BoundaryError, match="JSON object"):
        validate_context(ctx)


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "api_key",
        "API-KEY",
        "refresh_token",
        "Authorization",
        "cookie",
        "secret",
        "token",
        "refresh_tokens",
        "access_token",
        "accessToken",
        "apiKey",
        "APIKey",
        "apikey",
        "X-Auth-Token",
        "auth",
        "db_pass",
        "dbPassword",
        "passwd",
        "passphrase",
        "client_secret",
        "clientSecret",
        "aws_secret_access_key",
        "private_key",
        "privateKey",
        "session_id",
        "sessionId",
        "credentials",
        "bearer",
        "set-cookie",
        "github_token",
        "auth_header",
    ],
)
def test_secret_looking_keys_are_rejected(key: str) -> None:
    with pytest.raises(BoundaryError, match="credential"):
        validate_context({"nested": {key: "x"}})


@pytest.mark.parametrize(
    "key",
    [
        "passenger_count",
        "passengerCount",
        "author",
        "authors",
        "compass_heading",
        "tokenizer",
        "token_count",
        "tokens_used",
        "max_tokens",
        "tokens",
        "primary_key",
        "sort_key",
        "pass_rate",
        "bypass",
        "keyboard_layout",
        "monkey",
        "session_count",
        "secretary",
        "authority",
        "oauth_provider",
        "cookies_accepted",
    ],
)
def test_innocent_keys_containing_secret_words_are_accepted(key: str) -> None:
    validate_context({"nested": {key: 1}})


def test_allow_keys_exempts_exact_names() -> None:
    validate_context({"auth_token_count": 12}, allow_keys=frozenset({"auth_token_count"}))
    with pytest.raises(BoundaryError):
        validate_context({"session_id": "s"}, allow_keys=frozenset({"auth_token_count"}))


@pytest.mark.parametrize(
    "value",
    [
        "sk-" + "a" * 32,
        "ghp_" + "A" * 36,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIE",
        "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4fwpM",
        "Authorization was Bearer abcdefghijklmnopqrstuvwxyz012345",
        "xoxb-1234567890-abcdef",
        "AIza" + "B" * 35,
    ],
)
def test_secret_looking_values_are_rejected_without_echoing_them(value: str) -> None:
    with pytest.raises(BoundaryError) as info:
        validate_context({"notes": ["fine", value]})
    assert "$.notes[1]" in str(info.value)
    assert value not in str(info.value)


@pytest.mark.parametrize("value", [{1, 2}, object(), float("nan"), float("inf"), b"x", (1, 2)])
def test_non_json_values_are_rejected(value: object) -> None:
    with pytest.raises(BoundaryError):
        validate_context({"v": value})


def test_non_string_keys_are_rejected() -> None:
    with pytest.raises(BoundaryError, match="keys must be strings"):
        validate_context({"v": {1: "x"}})


def test_size_limit() -> None:
    validate_context({"x": "a" * 100}, max_bytes=200)
    with pytest.raises(BoundaryError, match="limit is 50"):
        validate_context({"x": "a" * 100}, max_bytes=50)


def test_size_is_measured_in_utf8_bytes() -> None:
    with pytest.raises(BoundaryError):
        validate_context({"x": "å" * 30}, max_bytes=50)


def test_semantic_hash_ignores_volatile_keys_at_any_depth() -> None:
    a = {"ts": 1, "event": {"kind": "hrv", "observed_at": "2026-10-24"}, "list": [{"ts": 9}]}
    b = {"ts": 2, "event": {"kind": "hrv", "observed_at": "2026-10-25"}, "list": [{"ts": 3}]}
    assert semantic_hash(a, VOLATILE) == semantic_hash(b, VOLATILE)
    c = {"ts": 2, "event": {"kind": "sleep"}, "list": [{"ts": 3}]}
    assert semantic_hash(a, VOLATILE) != semantic_hash(c, VOLATILE)


def test_semantic_hash_mixes_in_extra() -> None:
    assert semantic_hash({"a": 1}, VOLATILE, "m1") != semantic_hash({"a": 1}, VOLATILE, "m2")


def test_key_segments_and_punctuation_only_keys() -> None:
    from earned_gate.boundary import key_segments, looks_like_secret_key

    assert key_segments("X-AuthToken_v2") == ["x", "auth", "token", "v", "2"]
    assert key_segments("APIKey") == ["api", "key"]
    assert not looks_like_secret_key("__")
    validate_context({"__": 1, "": 2})


# Compound names that a plain word list would let through.
@pytest.mark.parametrize(
    "key",
    [
        "access_token_value",
        "refresh_token_value",
        "auth_token_raw",
        "github_token_str",
        "token_value",
        "github_tokens",
        "slack_tokens",
        "passcode",
        "tokenValue",
        "api_tokens",
        "user_pass_hash",
        "pass_value",
        "first_pass",
        "secret_count",
        "passkey",
        "tokens_value",
    ],
)
def test_credential_compounds_are_rejected(key: str) -> None:
    with pytest.raises(BoundaryError, match="credential"):
        validate_context({key: "x"})


@pytest.mark.parametrize(
    "key",
    [
        "passenger_count",
        "author",
        "tokenizer",
        "max_tokens",
        "token_count",
        "tokens",
        "pass_rate",
        "compass_heading",
        "tokens_used",
        "token_limit",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "pass_count",
        "maxTokens",
    ],
)
def test_quantity_shaped_token_and_pass_keys_are_accepted(key: str) -> None:
    validate_context({key: 1})


def test_cyclic_context_is_a_boundary_error() -> None:
    ctx: dict[str, object] = {"a": 1}
    ctx["self"] = ctx
    with pytest.raises(BoundaryError, match="cyclic"):
        validate_context(ctx)
    lst: list[object] = []
    lst.append(lst)
    with pytest.raises(BoundaryError, match="cyclic"):
        validate_context({"l": lst})


def test_too_deep_context_is_a_boundary_error() -> None:
    deep: object = 1
    for _ in range(5000):
        deep = {"d": deep}
    with pytest.raises(BoundaryError, match="nested deeper than 64"):
        validate_context({"x": deep}, max_bytes=10**9)


def test_depth_limit_boundary() -> None:
    ok: object = 1
    for _ in range(63):
        ok = [ok]
    validate_context({"x": ok})  # 64 containers including the root
    with pytest.raises(BoundaryError, match="nested deeper"):
        validate_context({"x": [ok]})


def test_shared_but_acyclic_values_are_fine() -> None:
    shared = {"k": 1}
    validate_context({"a": shared, "b": shared, "c": [shared, shared]})


# Run-together names, and auth/cookie in any position.
@pytest.mark.parametrize(
    "key",
    [
        "dbpassword",
        "adminpassword",
        "userpassword",
        "DBPASSWORD",
        "passwordhash",
        "githubtoken",
        "apitoken",
        "MYTOKEN",
        "apisecret",
        "webhooksecret",
        "webhooksecrets",
        "secretkey",
        "dbpass",
        "db_pwd",
        "auth_key",
        "authKey",
        "auth_value",
        "auth_data",
        "auth_string",
        "cookie_value",
        "cookieValue",
        "session_cookie_value",
        "cookie_jar",
        "oauth",
        "oauth2",
        "oauth2_token",
        "githubtokens",
    ],
)
def test_run_together_and_mid_key_credentials_are_rejected(key: str) -> None:
    with pytest.raises(BoundaryError, match="credential"):
        validate_context({key: "x"})


@pytest.mark.parametrize(
    "key",
    [
        "auth_method",
        "auth_type",
        "auth_required",
        "oauth_provider",
        "oauth2_provider",
        "cookies_accepted",
        "cookie_consent",
        "secretary",
        "bypass",
        "compass",
        "githubtoken_count",
        "maxtokens",
    ],
)
def test_safe_auth_cookie_and_run_together_keys_are_accepted(key: str) -> None:
    validate_context({key: 1})


# Key names are scanned for credential value shapes too.
@pytest.mark.parametrize(
    "key",
    ["ghp_" + "A" * 36, "sk-" + "a" * 32, "AKIA" + "ABCDEFGHIJKLMNOP", "xoxb-1234567890-abcdef"],
)
def test_keys_shaped_like_credentials_are_rejected_without_echoing_them(key: str) -> None:
    with pytest.raises(BoundaryError) as info:
        validate_context({"usage_by_key": {key: 3}})
    assert key not in str(info.value)
    assert "$.usage_by_key" in str(info.value)


# Plurals, more run-together names and pwd anywhere.
@pytest.mark.parametrize(
    "key",
    [
        "passwords",
        "user_passwords",
        "api_keys",
        "access_keys",
        "private_keys",
        "session_ids",
        "authkey",
        "MYAPIKEY",
        "cookie_header",
        "auth_code",
        "oauth_verifier",
        "oauth_code",
        "pwd_hash",
        "dbpasswordhash",
        "mypasswordfield",
        "userauthtokenvalue",
        "x_apikey_value",
    ],
)
def test_plural_and_run_together_credentials_are_rejected(key: str) -> None:
    with pytest.raises(BoundaryError, match="credential"):
        validate_context({key: "x"})


# Quantity-shaped token/pass keys are only allowed with a numeric value.
@pytest.mark.parametrize(
    "ctx",
    [
        {"token_count": "ghx-opaque-value"},
        {"tokens": ["opaque-token-value"]},
        {"max_tokens": "abc"},
        {"pass_rate": None},
        {"tokens_used": True},
    ],
)
def test_quantity_shaped_keys_must_hold_numbers(ctx: dict[str, object]) -> None:
    with pytest.raises(BoundaryError, match="must hold a number"):
        validate_context(ctx)


def test_quantity_shaped_keys_with_numbers_pass() -> None:
    validate_context({"token_count": 12, "tokens": 3.5, "max_tokens": 100, "pass_rate": 0.9})
    validate_context({"token_count": "n/a"}, allow_keys=frozenset({"token_count"}))
