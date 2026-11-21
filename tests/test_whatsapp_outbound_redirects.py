"""Regression tests for #210: reject non-2xx outbound WhatsApp responses (3xx redirects)."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel


def _channel(**kwargs: Any) -> WhatsAppChannel:
    defaults: dict[str, Any] = dict(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        fallback_message="FALLBACK",
        max_outbound_retries=1,
        outbound_base_delay=0.0,
        outbound_max_delay=0.0,
    )
    defaults.update(kwargs)
    return WhatsAppChannel(**defaults)


@pytest.mark.parametrize("status_code", [200, 201])
async def test_outbound_accepts_2xx_and_passes_follow_redirects_false(status_code: int) -> None:
    """2xx responses succeed and client.post explicitly passes follow_redirects=False."""
    channel = _channel()
    captured_kwargs: list[dict[str, Any]] = []

    async def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured_kwargs.append(kwargs)
        return httpx.Response(status_code=status_code, json={"messages": [{"id": "wamid.ok"}]})

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=fake_post)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        ok = await channel._send_single_outbound_whatsapp("5491112345678", "hello")

    assert ok is True
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("follow_redirects") is False


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
async def test_outbound_rejects_3xx_redirects_without_retry(
    status_code: int, caplog: pytest.LogCaptureFixture
) -> None:
    """3xx responses must return False, log an error with masked phone, and never retry."""
    channel = _channel(max_outbound_retries=3)
    resp = httpx.Response(
        status_code=status_code,
        headers={"Location": "https://other.example/redirect"},
        text="<html>Moved</html>",
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)

    with (
        caplog.at_level("WARNING", logger="chatflow_agent.whatsapp"),
        patch("httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__.return_value = mock_client
        ok = await channel._send_single_outbound_whatsapp("5491112345678", "hello")

    assert ok is False
    assert mock_client.post.call_count == 1
    assert mock_client.post.call_args.kwargs.get("follow_redirects") is False
    error_messages = [r.message for r in caplog.records if r.levelname == "ERROR"]
    assert any(str(status_code) in msg and "5491***678" in msg for msg in error_messages)
    assert "5491112345678" not in caplog.text


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
async def test_outbound_stops_remaining_chunks_on_3xx_redirect(status_code: int) -> None:
    """A 3xx redirect on chunk 1 must stop _send_outbound_whatsapp from sending chunk 2."""
    channel = _channel(max_outbound_retries=0)
    long_reply = ("Para 1: " + "A" * 2500) + "\n\n" + ("Para 2: " + "B" * 2500)

    resp = httpx.Response(
        status_code=status_code,
        headers={"Location": "https://other.example/redirect"},
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_outbound_whatsapp("5491112345678", long_reply)

    assert mock_client.post.call_count == 1


@pytest.mark.parametrize("status_code", [400, 401, 500, 502])
async def test_outbound_preserves_4xx_and_5xx_definitive_failure_behavior(
    status_code: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-429 4xx/5xx responses continue to fail definitively without retrying."""
    channel = _channel(max_outbound_retries=3)
    resp = httpx.Response(
        status_code=status_code,
        json={"error": {"message": "upstream failure", "code": 999}},
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)

    with (
        caplog.at_level("ERROR", logger="chatflow_agent.whatsapp"),
        patch("httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__.return_value = mock_client
        ok = await channel._send_single_outbound_whatsapp("5491112345678", "hello")

    assert ok is False
    assert mock_client.post.call_count == 1
    assert any(str(status_code) in r.message and "5491***678" in r.message for r in caplog.records)


async def test_outbound_preserves_429_retry_then_200_success() -> None:
    """429 rate-limit retry behavior remains intact."""
    channel = _channel(max_outbound_retries=2)
    responses = [
        httpx.Response(status_code=429, headers={"Retry-After": "0"}),
        httpx.Response(status_code=200, json={"messages": [{"id": "wamid.ok"}]}),
    ]
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=responses)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        ok = await channel._send_single_outbound_whatsapp("5491112345678", "hello")

    assert ok is True
    assert mock_client.post.call_count == 2


async def test_outbound_real_httpx_client_overrides_follow_redirects_true(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real httpx.AsyncClient(follow_redirects=True) must still not follow outbound 3xx redirects."""
    fake_token = "EAAG-FAKE-WA-TOKEN-REDIRECT-TEST"
    redirect_target = "https://redirect.example/capture?secret=loc_fake_123"
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if "redirect.example" in str(request.url):
            return httpx.Response(200, json={"messages": [{"id": "should-never-happen"}]})
        return httpx.Response(
            302,
            headers={"Location": redirect_target},
            text="<html>302 Found</html>",
        )

    def _make_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        )

    class _RealHttpxWithMockTransport:
        ConnectError = httpx.ConnectError
        ConnectTimeout = httpx.ConnectTimeout
        ReadTimeout = httpx.ReadTimeout
        WriteTimeout = httpx.WriteTimeout
        AsyncClient = staticmethod(_make_client)

    channel = _channel(access_token=fake_token, max_outbound_retries=2)
    channel._httpx = _RealHttpxWithMockTransport

    with caplog.at_level("WARNING", logger="chatflow_agent.whatsapp"):
        ok = await channel._send_single_outbound_whatsapp("5491112345678", "hello")

    assert ok is False
    assert len(seen_urls) == 1
    assert "graph.facebook.com" in seen_urls[0]
    assert "redirect.example" not in seen_urls[0]
    assert redirect_target not in caplog.text
    assert "loc_fake_123" not in caplog.text
    assert fake_token not in caplog.text
    assert "5491112345678" not in caplog.text
    assert any("302" in r.message and "5491***678" in r.message for r in caplog.records)
