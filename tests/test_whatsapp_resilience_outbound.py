"""Tests for WhatsApp outbound resilience: backoff retry, 24h window detection, and LLM timeout."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.memory import SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError


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
        outbound_base_delay=0.01,
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
async def test_outbound_retry_respects_retry_after_header() -> None:
    """Outbound retry must parse and respect Retry-After response header."""
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=2,
        outbound_base_delay=0.01,
    )

    resp_429 = httpx.Response(
        status_code=429,
        text="Rate limit",
        headers={"Retry-After": "0.02"},
    )
    resp_200 = httpx.Response(status_code=200, json={"messages": [{"id": "ok"}]})

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[resp_429, resp_200])

    with patch("httpx.AsyncClient") as mock_cls, patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_single_outbound_whatsapp("5491100000000", "Hello!")

        assert mock_client.post.call_count == 2
        assert mock_sleep.call_count == 1
        # First retry sleep delay should be >= 0.02 (parsed Retry-After)
        slept = mock_sleep.call_args[0][0]
        assert slept >= 0.02, f"Expected sleep >= 0.02s from Retry-After, got {slept}"


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
async def test_outbound_no_retry_on_read_timeout() -> None:
    """ReadTimeout must NOT retry because Meta may have already received and delivered the message."""
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=3,
        outbound_base_delay=0.01,
    )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Read timed out waiting for Meta response"))

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        await channel._send_single_outbound_whatsapp("5491100000000", "Hello!")

    # Must NOT retry: call_count is strictly 1
    assert mock_client.post.call_count == 1


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
async def test_24h_window_hook_exception_does_not_crash() -> None:
    """If on_24h_window_expired raises an exception, it must be caught safely."""
    def failing_hook(phone: str) -> None:
        raise RuntimeError("Hook crashed unexpectedly!")

    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        on_24h_window_expired=failing_hook,
    )

    resp_meta_24h = httpx.Response(
        status_code=400,
        json={"error": {"code": 131047, "message": "Re-engagement message"}},
    )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp_meta_24h)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        # Must execute cleanly without unhandled exception
        await channel._send_single_outbound_whatsapp("5491199999999", "Ping")

    assert mock_client.post.call_count == 1


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
        llm_timeout_seconds=0.05,
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


class FailingTurnEngine(GeminiEngine):
    """Engine that succeeds on turn 1 and fails on turn 2."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def generate_turn_async(self, agent, history):
        self.call_count += 1
        if self.call_count == 1:
            return EngineTurnResult(text="Turn 1 OK")
        raise ProviderError("Upstream LLM 500 error mid-turn")


@pytest.mark.asyncio
async def test_llm_failure_mid_turn_rolls_back_history_and_sends_fallback(tmp_path: Path) -> None:
    """Full path: LLM engine fails mid-turn -> history rolls back -> fallback sent to user (no silence)."""
    db_file = tmp_path / "test_mid_turn_fail.db"
    store = SQLiteSessionStore(db_path=db_file)
    engine = FailingTurnEngine()
    agent = Agent(name="ResilientBot", instructions="Test")
    runner = Runner(starting_agent=agent, session_store=store, engine=engine)

    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_tok",
        phone_number_id="12345",
        verify_signature=False,
        fallback_message="Disculpa, ocurrió un error inesperado. Intenta nuevamente.",
    )
    channel.attach(runner)

    delivered: list[tuple[str, str]] = []

    async def fake_send(to_phone: str, text: str) -> None:
        delivered.append((to_phone, text))

    with patch.object(channel, "_send_outbound_whatsapp", side_effect=fake_send):
        # 1. Turn 1: succeeds
        await channel._process_message_background(
            wamid="wamid.turn1",
            sender_phone="5491100000001",
            user_text="Turn 1 question",
        )
        assert len(delivered) == 1
        assert delivered[0] == ("5491100000001", "Turn 1 OK")

        session = store.get_or_create("5491100000001", default_agent=agent)
        assert len(session.history) == 2  # USER + MODEL

        # 2. Turn 2: LLM engine raises ProviderError mid-turn
        await channel._process_message_background(
            wamid="wamid.turn2",
            sender_phone="5491100000001",
            user_text="Turn 2 question (fails)",
        )

        # Confirm fallback message WAS sent to user (no silence!)
        assert len(delivered) == 2
        assert delivered[1] == (
            "5491100000001",
            "Disculpa, ocurrió un error inesperado. Intenta nuevamente.",
        )

        # Confirm history was rolled back: Turn 2 was removed, only Turn 1 remains
        session_after = store.get_or_create("5491100000001", default_agent=agent)
        assert len(session_after.history) == 2
        assert session_after.history[-1].content == "Turn 1 OK"

        # Also reload from disk SQLite directly to confirm database integrity
        fresh_store = SQLiteSessionStore(db_path=db_file)
        reloaded = fresh_store.get_or_create("5491100000001", default_agent=agent)
        assert len(reloaded.history) == 2
        assert reloaded.history[0].content == "Turn 1 question"
        assert reloaded.history[1].content == "Turn 1 OK"
