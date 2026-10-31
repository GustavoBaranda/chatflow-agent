"""Acceptance for #169: stop after failed chunk, empty reply fallback, credential warning."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel


def _channel(**kwargs) -> WhatsAppChannel:
    defaults = dict(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        fallback_message="FALLBACK",
        max_outbound_retries=0,
    )
    defaults.update(kwargs)
    return WhatsAppChannel(**defaults)


@pytest.mark.asyncio
async def test_outbound_stops_after_first_failed_chunk() -> None:
    """Later chunks must not be posted once an earlier chunk fails (#169)."""
    channel = _channel()
    long_reply = ("Para 1: " + "A" * 2500) + "\n\n" + ("Para 2: " + "B" * 2500)

    resp_fail = httpx.Response(
        status_code=500,
        text='{"error":{"message":"server error","code":1}}',
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp_fail)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_outbound_whatsapp("5491100000000", long_reply)

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_outbound_24h_hook_fires_at_most_once_per_reply() -> None:
    """A multi-chunk reply with 131047 must invoke the hook once, not per chunk."""
    hook_calls: list[str] = []

    def hook(phone: str) -> None:
        hook_calls.append(phone)

    channel = _channel(on_24h_window_expired=hook)
    long_reply = ("Para 1: " + "A" * 2500) + "\n\n" + ("Para 2: " + "B" * 2500)

    resp_24h = httpx.Response(
        status_code=400,
        json={"error": {"code": 131047, "message": "window expired"}},
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp_24h)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_outbound_whatsapp("5491199999999", long_reply)

    assert mock_client.post.call_count == 1
    assert hook_calls == ["5491199999999"]


@pytest.mark.asyncio
async def test_empty_reply_sends_fallback_not_empty_body() -> None:
    """Empty/whitespace replies must be replaced with fallback before posting."""
    channel = _channel()
    posted: list[dict] = []

    async def capture_post(url, json=None, headers=None, timeout=None):
        posted.append(json)
        return httpx.Response(status_code=200, json={"messages": [{"id": "ok"}]})

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=capture_post)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_outbound_whatsapp("5491100000000", "   ")

    assert len(posted) == 1
    assert posted[0]["text"]["body"] == "FALLBACK"


def test_missing_outbound_credentials_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Construction without access_token/phone_number_id must warn once."""
    with caplog.at_level("WARNING", logger="chatflow_agent.whatsapp"):
        WhatsAppChannel(verify_token="tok", verify_signature=False)

    assert any("without outbound credentials" in r.message for r in caplog.records)
