"""Tests for WhatsApp outbound resilience: backoff retry, 24h window detection, and LLM timeout."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner


class SlowEngine(GeminiEngine):
    """Engine that delays longer than timeout to trigger fallback."""

    def __init__(self, delay_seconds: float = 0.5) -> None:
        super().__init__()
        self.delay = delay_seconds

    async def generate_turn_async(self, agent, history):
        await asyncio.sleep(self.delay)
        return EngineTurnResult(text="Should not reach here")


@pytest.mark.asyncio
async def test_outbound_retry_on_429_rate_limit() -> None:
    """Outbound sends must retry on HTTP 429 using exponential backoff."""
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=3,
        outbound_base_delay=0.01,  # Fast for unit tests
    )

    resp_429 = httpx.Response(status_code=429, text="Rate limit exceeded")
    resp_200 = httpx.Response(status_code=200, json={"messages": [{"id": "wamid_sent"}]})

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[resp_429, resp_429, resp_200])

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_single_outbound_whatsapp("5491100000000", "Hello!")

    assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_outbound_retry_on_network_connect_error() -> None:
    """Outbound sends must retry on transient connection failures."""
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=2,
        outbound_base_delay=0.01,
    )

    resp_200 = httpx.Response(status_code=200, json={"messages": [{"id": "wamid_sent"}]})

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=[httpx.ConnectError("Connection reset"), resp_200]
    )

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_single_outbound_whatsapp("5491100000000", "Hello!")

    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_outbound_no_retry_on_500_server_error() -> None:
    """Outbound sends must NOT retry on 5xx errors to prevent duplicate message dispatch."""
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=3,
        outbound_base_delay=0.01,
    )

    resp_500 = httpx.Response(status_code=500, text="Internal Server Error")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp_500)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_single_outbound_whatsapp("5491100000000", "Hello!")

    assert mock_client.post.call_count == 1  # No retries for 5xx


@pytest.mark.asyncio
async def test_24h_window_expired_error_code_131047() -> None:
    """Meta error 131047 must invoke on_24h_window_expired hook without retrying."""
    hook_called_with = []

    async def hook(phone: str) -> None:
        hook_called_with.append(phone)

    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        on_24h_window_expired=hook,
    )

    resp_meta_24h = httpx.Response(
        status_code=400,
        json={"error": {"code": 131047, "message": "Re-engagement message"}},
    )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp_meta_24h)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_single_outbound_whatsapp("5491199999999", "Ping")

    assert mock_client.post.call_count == 1
    assert hook_called_with == ["5491199999999"]


@pytest.mark.asyncio
async def test_llm_dispatch_timeout_triggers_fallback() -> None:
    """Slow LLM dispatch exceeding llm_timeout_seconds must fall back to fallback_message."""
    engine = SlowEngine(delay_seconds=0.3)
    agent = Agent(name="SlowBot", instructions="Test")
    runner = Runner(starting_agent=agent, engine=engine)

    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_tok",
        phone_number_id="12345",
        verify_signature=False,
        llm_timeout_seconds=0.05,  # Times out quickly
        fallback_message="Custom fallback message: timeout occurred",
    )
    channel.attach(runner)

    delivered_replies = []

    async def capture_outbound(to_phone: str, text: str) -> None:
        delivered_replies.append((to_phone, text))

    with patch.object(channel, "_send_outbound_whatsapp", side_effect=capture_outbound):
        await channel._process_message_background(
            wamid="wamid.timeout_test",
            sender_phone="5491177777777",
            user_text="Slow message",
        )

    assert len(delivered_replies) == 1
    assert delivered_replies[0][1] == "Custom fallback message: timeout occurred"
