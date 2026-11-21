"""Regression tests for credential redaction across engines and channels (AUD-01).

All credentials in this module are fictitious and every HTTP call is served offline
through ``httpx.MockTransport``.
"""

import json
import logging
import traceback
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, quote_plus

import httpx
import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core._redaction import (
    extract_url_secrets,
    redact_secrets,
    sanitize_error_body,
)
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines.anthropic import AnthropicEngine
from chatflow_agent.core.engines.gemini import GeminiEngine
from chatflow_agent.core.engines.openai_compatible import OpenAIEngine
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role

OPENAI_SECRET = "sk-FAKE-OPENAI-SECRET+123/456=xyz"
ANTHROPIC_SECRET = "sk-ant-FAKE-ANTHROPIC-SECRET+789/012=abc"
WA_SECRET = "EAAG-FAKE-WA-TOKEN+345/678=def"
GEMINI_SECRET = "AIza-FAKE-GEMINI-KEY+999/888=ghi"
PROXY_SECRET = "proxy-FAKE-PASS-4242"

HISTORY = [Message(role=Role.USER, content="hello")]


def _secret_variants(secret: str) -> list[str]:
    q = quote(secret, safe="")
    qp = quote_plus(secret, safe="")
    q_lower = "".join(
        c.lower() if (i >= 1 and q[i - 1] == "%") or (i >= 2 and q[i - 2] == "%") else c
        for i, c in enumerate(q)
    )
    qp_lower = "".join(
        c.lower() if (i >= 1 and qp[i - 1] == "%") or (i >= 2 and qp[i - 2] == "%") else c
        for i, c in enumerate(qp)
    )
    json_ascii = json.dumps(secret)[1:-1]
    json_unicode = json.dumps(secret, ensure_ascii=False)[1:-1]
    return [
        secret,
        q,
        q_lower,
        qp,
        qp_lower,
        json_ascii,
        json_unicode,
        json_ascii.replace("/", r"\/"),
        json_unicode.replace("/", r"\/"),
    ]


def _assert_not_in_text(text: str, secret: str) -> None:
    for variant in _secret_variants(secret):
        assert variant not in text, f"Leaked variant {variant!r} in: {text!r}"


def _assert_exception_clean(exc: BaseException, secret: str) -> None:
    _assert_not_in_text(str(exc), secret)
    _assert_not_in_text(repr(exc), secret)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _assert_not_in_text(tb, secret)
    assert exc.__cause__ is None, f"Unexpected __cause__: {exc.__cause__!r}"
    assert exc.__context__ is None, f"Unexpected __context__: {exc.__context__!r}"


# ---------------------------------------------------------------------------
# Unit tests for core._redaction helpers
# ---------------------------------------------------------------------------


def test_redact_secrets_covers_plain_url_and_json_escaped_forms() -> None:
    tricky = 'sk-FAKE-"quote"\\slash+1/2=3'
    payload = " ".join(
        [
            tricky,
            quote(tricky, safe=""),
            quote_plus(tricky, safe=""),
            json.dumps({"err": tricky}),
        ]
    )
    redacted = redact_secrets(payload, [tricky])
    assert "FAKE" not in redacted
    assert "***" in redacted


def test_sanitize_error_body_redacts_before_truncation() -> None:
    # Place the secret straddling the 1000-char boundary.
    body = ("A" * 985) + OPENAI_SECRET + ("B" * 50)
    sanitized = sanitize_error_body(body, [OPENAI_SECRET], limit=1000)
    assert "FAKE-OPENAI" not in sanitized
    assert OPENAI_SECRET[:10] not in sanitized
    assert "***" in sanitized
    assert sanitized.endswith("… (truncated)")


def test_extract_url_secrets_captures_userinfo_and_query_values() -> None:
    url = f"https://admin:{PROXY_SECRET}@gateway.internal/v1?api_key={OPENAI_SECRET}&mode=fast"
    secrets = extract_url_secrets(url)
    assert PROXY_SECRET in secrets
    assert OPENAI_SECRET in secrets
    redacted = redact_secrets(f"Failed connecting to {url}", secrets)
    assert PROXY_SECRET not in redacted
    assert OPENAI_SECRET not in redacted
    assert "gateway.internal/v1" in redacted


def test_sec002_redact_secrets_covers_json_escaped_slashes_and_lowercase_percent() -> None:
    secret = OPENAI_SECRET
    slash_escaped = json.dumps({"error": secret}).replace("/", r"\/")
    lower_percent = quote(secret, safe="").replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")
    mixed_percent = quote_plus(secret, safe="").replace("%2F", "%2f")

    for payload in (slash_escaped, f"key={lower_percent}", f"key={mixed_percent}"):
        redacted = redact_secrets(payload, [secret])
        _assert_not_in_text(redacted, secret)
        assert "FAKE-OPENAI" not in redacted
        assert "***" in redacted


def test_sec002_extract_url_secrets_captures_username_only_token() -> None:
    token = "tok-FAKE-USER-ONLY-9988"
    url = f"https://{token}@gateway.internal/v1"
    secrets = extract_url_secrets(url)
    assert token in secrets
    # Must redact both when embedded in the URL and when echoed standalone.
    assert token not in redact_secrets(f"Upstream {url} rejected token {token}", secrets)


# ---------------------------------------------------------------------------
# OpenAIEngine redaction and exception-chaining tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body_factory"),
    [
        (401, lambda s: f"Unauthorized: invalid api key {s}"),
        (401, lambda s: f"Proxy error for key={quote(s, safe='')}"),
        (401, lambda s: json.dumps({"error": {"message": f"Invalid key: {s}"}})),
        (500, lambda s: ("X" * 985) + s + " tail"),
    ],
)
async def test_openai_http_error_body_is_redacted_before_truncation(
    status: int, body_factory: Callable[[str], str]
) -> None:
    body = body_factory(OPENAI_SECRET)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, text=body))
    ) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, http_client=client)
        engine.max_retries = 0
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, OPENAI_SECRET)
    assert OPENAI_SECRET[:12] not in str(exc_info.value)
    assert str(status) in str(exc_info.value)


@pytest.mark.parametrize(
    ("exc_factory", "expected_name"),
    [
        (lambda req: httpx.ConnectError("connection refused", request=req), "ConnectError"),
        (lambda req: httpx.ConnectTimeout("connect timed out", request=req), "ConnectTimeout"),
        (lambda req: httpx.ReadTimeout("read timed out", request=req), "ReadTimeout"),
        (lambda req: httpx.RemoteProtocolError("protocol error", request=req), "RemoteProtocolError"),
    ],
)
async def test_openai_request_errors_do_not_chain_request_headers(
    exc_factory: Callable[[httpx.Request], Exception],
    expected_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc_factory(req)

    caplog.set_level(logging.WARNING, logger="chatflow_agent.engine.openai")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, http_client=client)
        engine.max_retries = 1
        engine.base_delay = 0.0
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, OPENAI_SECRET)
    assert expected_name in str(exc_info.value)
    _assert_not_in_text(caplog.text, OPENAI_SECRET)


async def test_openai_base_url_credentials_redacted_in_logs_and_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    url_with_creds = f"https://user:{PROXY_SECRET}@gateway.internal/v1"

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"Connect failed to {req.url}", request=req)

    caplog.set_level(logging.WARNING, logger="chatflow_agent.engine.openai")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, base_url=url_with_creds, http_client=client)
        engine.max_retries = 1
        engine.base_delay = 0.0
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, PROXY_SECRET)
    _assert_exception_clean(exc_info.value, OPENAI_SECRET)
    _assert_not_in_text(caplog.text, PROXY_SECRET)


async def test_openai_non_json_200_does_not_leak_json_decode_error_doc() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text=f"<html>bad key={OPENAI_SECRET}</html>")
        )
    ) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, http_client=client)
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, OPENAI_SECRET)


@pytest.mark.parametrize("bad_key", [f"{OPENAI_SECRET}-ñ", f"{OPENAI_SECRET}\r\n", f" {OPENAI_SECRET}"])
async def test_openai_rejects_invalid_header_api_key_without_echoing(bad_key: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []}))
    ) as client:
        engine = OpenAIEngine(api_key=bad_key, http_client=client)
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, OPENAI_SECRET)


# ---------------------------------------------------------------------------
# AnthropicEngine redaction and exception-chaining tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body_factory"),
    [
        (401, lambda s: f"authentication_error: bad x-api-key {s}"),
        (401, lambda s: f"key={quote(s, safe='')}"),
        (401, lambda s: json.dumps({"type": "error", "error": {"message": f"Invalid key {s}"}})),
        (500, lambda s: ("Y" * 985) + s + " tail"),
    ],
)
async def test_anthropic_http_error_body_is_redacted_before_truncation(
    status: int, body_factory: Callable[[str], str]
) -> None:
    body = body_factory(ANTHROPIC_SECRET)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, text=body))
    ) as client:
        engine = AnthropicEngine(api_key=ANTHROPIC_SECRET, http_client=client)
        engine.max_retries = 0
        agent = Agent(name="a", instructions="test", model="claude-3-5-sonnet-latest", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, ANTHROPIC_SECRET)
    assert ANTHROPIC_SECRET[:12] not in str(exc_info.value)
    assert str(status) in str(exc_info.value)


@pytest.mark.parametrize(
    ("exc_factory", "expected_name"),
    [
        (lambda req: httpx.ConnectError("connection refused", request=req), "ConnectError"),
        (lambda req: httpx.ReadTimeout("read timed out", request=req), "ReadTimeout"),
    ],
)
async def test_anthropic_request_errors_do_not_chain_request_headers(
    exc_factory: Callable[[httpx.Request], Exception],
    expected_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc_factory(req)

    caplog.set_level(logging.WARNING, logger="chatflow_agent.engine.anthropic")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = AnthropicEngine(api_key=ANTHROPIC_SECRET, http_client=client)
        engine.max_retries = 1
        engine.base_delay = 0.0
        agent = Agent(name="a", instructions="test", model="claude-3-5-sonnet-latest", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, ANTHROPIC_SECRET)
    assert expected_name in str(exc_info.value)
    _assert_not_in_text(caplog.text, ANTHROPIC_SECRET)


async def test_anthropic_non_json_200_does_not_leak_json_decode_error_doc() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text=f"bad gateway {ANTHROPIC_SECRET}")
        )
    ) as client:
        engine = AnthropicEngine(api_key=ANTHROPIC_SECRET, http_client=client)
        agent = Agent(name="a", instructions="test", model="claude-3-5-sonnet-latest", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, ANTHROPIC_SECRET)


@pytest.mark.parametrize("bad_key", [f"{ANTHROPIC_SECRET}-ñ", f"{ANTHROPIC_SECRET}\r\n"])
async def test_anthropic_rejects_invalid_header_api_key_without_echoing(bad_key: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"content": []}))
    ) as client:
        engine = AnthropicEngine(api_key=bad_key, http_client=client)
        agent = Agent(name="a", instructions="test", model="claude-3-5-sonnet-latest", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, ANTHROPIC_SECRET)


# ---------------------------------------------------------------------------
# GeminiEngine residual edge cases (JSONDecodeError.doc + JSON-escaped secret)
# ---------------------------------------------------------------------------


async def test_gemini_non_json_200_does_not_leak_json_decode_error_doc() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text=f"<html>bad {GEMINI_SECRET}</html>")
        )
    ) as client:
        engine = GeminiEngine(api_key=GEMINI_SECRET, http_client=client)
        agent = Agent(name="a", instructions="test", model="gemini-2.5-flash", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, GEMINI_SECRET)


async def test_gemini_redacts_json_escaped_secret_with_quotes_and_backslash() -> None:
    tricky_key = 'AIza-FAKE-"quote"\\slash'
    body = json.dumps({"error": {"message": f"Invalid key {tricky_key}"}})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text=body))
    ) as client:
        engine = GeminiEngine(api_key=tricky_key, http_client=client)
        agent = Agent(name="a", instructions="test", model="gemini-2.5-flash", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    assert "FAKE" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


# ---------------------------------------------------------------------------
# WhatsAppChannel outbound logging redaction tests
# ---------------------------------------------------------------------------


def _fake_httpx(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    def _make_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    class _FakeHttpx:
        ConnectError = httpx.ConnectError
        ConnectTimeout = httpx.ConnectTimeout
        ReadTimeout = httpx.ReadTimeout
        WriteTimeout = httpx.WriteTimeout
        AsyncClient = staticmethod(_make_client)

    return _FakeHttpx


@pytest.mark.parametrize(
    "body_text",
    [
        json.dumps({"error": {"message": f"Invalid OAuth token {WA_SECRET}", "code": 190}}),
        f"<html>Upstream auth failed for Bearer {WA_SECRET}</html>",
        ("Z" * 985) + WA_SECRET + " tail",
    ],
)
async def test_whatsapp_outbound_http_error_logs_are_redacted_and_bounded(
    body_text: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="chatflow_agent.whatsapp")
    channel = WhatsAppChannel(
        verify_token="verify-FAKE-token",
        access_token=WA_SECRET,
        phone_number_id="1234567890",
        app_secret="app-FAKE-secret",
    )
    channel._httpx = _fake_httpx(lambda req: httpx.Response(401, text=body_text))

    ok = await channel._send_single_outbound_whatsapp("5491112345678", "hola")

    assert ok is False
    _assert_not_in_text(caplog.text, WA_SECRET)
    assert WA_SECRET[:12] not in caplog.text
    assert "5491***678" in caplog.text
    assert "5491112345678" not in caplog.text


async def test_whatsapp_outbound_connect_error_logs_are_redacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"Proxy https://u:{PROXY_SECRET}@p.internal rejected Bearer {WA_SECRET}",
            request=req,
        )

    caplog.set_level(logging.WARNING, logger="chatflow_agent.whatsapp")
    channel = WhatsAppChannel(
        verify_token="verify-FAKE-token",
        access_token=WA_SECRET,
        phone_number_id="1234567890",
        app_secret="app-FAKE-secret",
        max_outbound_retries=1,
        outbound_base_delay=0.0,
    )
    channel._httpx = _fake_httpx(handler)

    ok = await channel._send_single_outbound_whatsapp("5491112345678", "hola")

    assert ok is False
    _assert_not_in_text(caplog.text, WA_SECRET)
    _assert_not_in_text(caplog.text, PROXY_SECRET)


@pytest.mark.parametrize("bad_token", [f"{WA_SECRET}-ñ", f"{WA_SECRET}\r\n"])
async def test_whatsapp_outbound_invalid_access_token_is_not_logged(
    bad_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="chatflow_agent.whatsapp")
    channel = WhatsAppChannel(
        verify_token="verify-FAKE-token",
        access_token=bad_token,
        phone_number_id="1234567890",
        app_secret="app-FAKE-secret",
    )
    channel._httpx = _fake_httpx(lambda req: httpx.Response(200, json={"messages": []}))

    ok = await channel._send_single_outbound_whatsapp("5491112345678", "hola")

    assert ok is False
    _assert_not_in_text(caplog.text, WA_SECRET)


# ---------------------------------------------------------------------------
# End-to-end SEC-002 regression tests across engines and WhatsAppChannel
# ---------------------------------------------------------------------------


async def test_sec002_openai_redacts_json_escaped_slash_lowercase_percent_and_user_only_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    url_token = "tok-FAKE-OPENAI-USER-ONLY"
    base_url = f"https://{url_token}@gateway.internal/v1"
    lower_pct = quote(OPENAI_SECRET, safe="").replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")
    slash_json = json.dumps({"error": f"bad {OPENAI_SECRET} and {url_token}"}).replace("/", r"\/")
    body = f"{slash_json} qs={lower_pct}"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text=body))
    ) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, base_url=base_url, http_client=client)
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, OPENAI_SECRET)
    _assert_exception_clean(exc_info.value, url_token)

    # Also verify retry logs and ConnectError when username-only token is echoed standalone.
    def fail_connect(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"Auth failed for user {url_token} at {req.url}", request=req)

    caplog.set_level(logging.WARNING, logger="chatflow_agent.engine.openai")
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_connect)) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, base_url=base_url, http_client=client)
        engine.max_retries = 1
        engine.base_delay = 0.0
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info2:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info2.value, url_token)
    _assert_exception_clean(exc_info2.value, OPENAI_SECRET)
    _assert_not_in_text(caplog.text, url_token)


async def test_sec002_anthropic_redacts_json_escaped_slash_lowercase_percent_and_user_only_url() -> None:
    url_token = "tok-FAKE-ANTHROPIC-USER-ONLY"
    base_url = f"https://{url_token}@gateway.internal/v1"
    lower_pct = quote(ANTHROPIC_SECRET, safe="").replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")
    body = json.dumps({"error": f"bad {ANTHROPIC_SECRET} {url_token}"}).replace("/", r"\/") + f" {lower_pct}"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text=body))
    ) as client:
        engine = AnthropicEngine(api_key=ANTHROPIC_SECRET, base_url=base_url, http_client=client)
        agent = Agent(name="a", instructions="test", model="claude-3-5-sonnet-latest", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, ANTHROPIC_SECRET)
    _assert_exception_clean(exc_info.value, url_token)


async def test_sec002_gemini_and_whatsapp_redact_json_escaped_slash_and_lowercase_percent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gem_lower = quote(GEMINI_SECRET, safe="").replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")
    gem_body = json.dumps({"error": GEMINI_SECRET}).replace("/", r"\/") + f" key={gem_lower}"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text=gem_body))
    ) as client:
        engine = GeminiEngine(api_key=GEMINI_SECRET, http_client=client)
        agent = Agent(name="a", instructions="test", model="gemini-2.5-flash", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    _assert_exception_clean(exc_info.value, GEMINI_SECRET)

    wa_lower = quote(WA_SECRET, safe="").replace("%2B", "%2b").replace("%2F", "%2f").replace("%3D", "%3d")
    wa_body = json.dumps({"error": {"message": f"bad {WA_SECRET} {wa_lower}"}}).replace("/", r"\/")
    caplog.set_level(logging.WARNING, logger="chatflow_agent.whatsapp")
    channel = WhatsAppChannel(
        verify_token="verify-FAKE-token",
        access_token=WA_SECRET,
        phone_number_id="1234567890",
        app_secret="app-FAKE-secret",
    )
    channel._httpx = _fake_httpx(lambda req: httpx.Response(401, text=wa_body))
    ok = await channel._send_single_outbound_whatsapp("5491112345678", "hola")
    assert ok is False
    _assert_not_in_text(caplog.text, WA_SECRET)


# ---------------------------------------------------------------------------
# SEC-003 regression tests: preserve diagnostics while protecting sensitive params
# ---------------------------------------------------------------------------


def test_sec003_generic_query_params_do_not_corrupt_diagnostics() -> None:
    url = (
        "https://azure.example.com/openai/deployments/gpt-4o/chat/completions"
        "?v=1&mode=fast&api-version=2024-02-01"
    )
    secrets = extract_url_secrets(url)
    assert "1" not in secrets
    assert "fast" not in secrets
    assert "2024-02-01" not in secrets

    diag = "HTTP 401 Unauthorized: fast path rejected api-version 2024-02-01 on attempt 1"
    assert redact_secrets(diag, secrets) == diag


def test_sec003_sensitive_short_and_encoded_query_tokens_are_still_redacted() -> None:
    short_token = "t0k"
    encoded_token = "a+b/c"
    encoded_qs = quote(encoded_token, safe="").lower()
    url = (
        f"https://gateway.internal/v1?v=1&mode=fast&api-version=2024-02-01"
        f"&token={short_token}&auth={encoded_qs}&key=xy&p=sk-FAKE-PREFIX"
    )
    secrets = extract_url_secrets(url)

    err_text = (
        f"HTTP 401 on {url}: fast mode failed for {short_token}, {encoded_token}, "
        f"{encoded_qs}, sk-FAKE-PREFIX on 2024-02-01"
    )
    redacted = redact_secrets(err_text, secrets)

    # All sensitive tokens (including short, encoded, k=v, and prefix-matched) are redacted.
    assert short_token not in redacted
    assert encoded_token not in redacted
    assert encoded_qs not in redacted
    assert "key=xy" not in redacted
    assert "sk-FAKE-PREFIX" not in redacted
    # Non-sensitive diagnostic details remain readable.
    assert "HTTP 401" in redacted
    assert "fast mode failed" in redacted
    assert "2024-02-01" in redacted


async def test_sec003_openai_preserves_http_diagnostics_with_generic_base_url_query() -> None:
    base_url = "https://azure.example.com/v1?v=1&mode=fast&api-version=2024-02-01"
    response_body = (
        f"HTTP 401: fast model error on api-version 2024-02-01 (key {OPENAI_SECRET})"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text=response_body))
    ) as client:
        engine = OpenAIEngine(api_key=OPENAI_SECRET, base_url=base_url, http_client=client)
        agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
        with pytest.raises(ProviderError) as exc_info:
            await engine.generate_turn_async(agent, HISTORY)

    msg = str(exc_info.value)
    _assert_exception_clean(exc_info.value, OPENAI_SECRET)
    assert "401" in msg
    assert "fast model error" in msg
    assert "2024-02-01" in msg


# ---------------------------------------------------------------------------
# Final Codex observations: strict ISO dates, partial percent-encoding, \u002f
# ---------------------------------------------------------------------------


def test_iso_date_exemption_allows_known_api_versions_and_redacts_secret_suffixes() -> None:
    benign_url = (
        "https://azure.example.com/v1"
        "?api-version=2024-02-01&rev=2024-02-01-preview&stage=2024-02-01-rc1"
    )
    benign_secrets = extract_url_secrets(benign_url)
    assert "2024-02-01" not in benign_secrets
    assert "2024-02-01-preview" not in benign_secrets
    assert "2024-02-01-rc1" not in benign_secrets

    secret_val = "2024-01-01-FAKESECRETVALUE123456"
    secret_url = f"https://gateway.internal/v1?p={secret_val}"
    extracted = extract_url_secrets(secret_url)
    assert secret_val in extracted
    redacted = redact_secrets(f"Failed for {secret_val} on {secret_url}", extracted)
    assert secret_val not in redacted
    assert "FAKESECRETVALUE" not in redacted


def test_partial_percent_encoding_preserving_slashes_is_redacted() -> None:
    secret = OPENAI_SECRET  # contains +, /, =
    partial_q = quote(secret)  # keeps '/' literal, encodes '+' -> %2B, '=' -> %3D
    partial_q_lower = partial_q.replace("%2B", "%2b").replace("%3D", "%3d")
    partial_qp = quote_plus(secret, safe="/")
    partial_qp_lower = partial_qp.replace("%2B", "%2b").replace("%3D", "%3d")

    for variant in (partial_q, partial_q_lower, partial_qp, partial_qp_lower):
        assert "/" in variant
        redacted = redact_secrets(f"upstream rejected {variant}", [secret])
        assert variant not in redacted
        assert "FAKE-OPENAI" not in redacted
        assert "***" in redacted


async def test_json_unicode_escaped_slash_is_redacted_in_errors() -> None:
    lower_u = json.dumps({"error": OPENAI_SECRET}).replace("/", r"\u002f")
    upper_u = json.dumps({"error": OPENAI_SECRET}).replace("/", r"\u002F")

    for body in (lower_u, upper_u):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req, b=body: httpx.Response(401, text=b))
        ) as client:
            engine = OpenAIEngine(api_key=OPENAI_SECRET, http_client=client)
            agent = Agent(name="a", instructions="test", model="gpt-4o-mini", engine=engine)
            with pytest.raises(ProviderError) as exc_info:
                await engine.generate_turn_async(agent, HISTORY)

        _assert_exception_clean(exc_info.value, OPENAI_SECRET)
        assert "FAKE-OPENAI" not in str(exc_info.value)
        assert r"\u002f" not in str(exc_info.value).lower()
