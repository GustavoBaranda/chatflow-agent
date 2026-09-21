"""Tests for session concurrency locking and channel resilience shields."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from chatflow_agent import Agent, Runner
from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message


class DelayedEchoEngine(BaseEngine):
    """Engine that simulates async processing delay to verify concurrency locking."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.processed_log: list[str] = []

    async def generate_turn_async(
        self,
        agent: Agent,
        history: list[Message],
    ) -> EngineTurnResult:
        last_msg = history[-1].content or ""
        # Simulate thinking time
        await asyncio.sleep(self.delay)
        self.processed_log.append(last_msg)
        return EngineTurnResult(text=f"Echo: {last_msg}")


class FailingEngine(BaseEngine):
    """Engine that simulates an unexpected LLM provider crash (e.g. quota exceeded)."""

    async def generate_turn_async(
        self,
        agent: Agent,
        history: list[Message],
    ) -> EngineTurnResult:
        raise ProviderError("Meta/Google API Quota Exceeded (HTTP 429)")


@pytest.mark.asyncio
async def test_session_concurrency_locking() -> None:
    """Verify that 3 rapid simultaneous messages from the same user are serialized strictly in FIFO order."""
    agent = Agent(name="Worker", instructions="Echo messages.")
    engine = DelayedEchoEngine(delay=0.03)
    runner = Runner(starting_agent=agent, engine=engine)

    # Launch 3 concurrent requests on the same session_id
    responses = await asyncio.gather(
        runner.run_async("user_concurrent", "Message 1"),
        runner.run_async("user_concurrent", "Message 2"),
        runner.run_async("user_concurrent", "Message 3"),
    )

    # Assert all completed
    assert [r.content for r in responses] == [
        "Echo: Message 1",
        "Echo: Message 2",
        "Echo: Message 3",
    ]

    # Verify session history order is strictly preserved
    session = runner.get_session("user_concurrent")
    assert session is not None
    user_history = [m.content for m in session.history if m.role.value == "user"]
    assert user_history == ["Message 1", "Message 2", "Message 3"]


def test_whatsapp_anti_500_error_shield() -> None:
    """Verify that when AI provider crashes, WhatsApp webhook returns HTTP 200 and dispatches fallback."""
    failing_agent = Agent(name="CrashBot", instructions="Crash.")
    runner = Runner(starting_agent=failing_agent, engine=FailingEngine())

    channel = WhatsAppChannel(
        verify_token="test_token",
        access_token="mock_meta_token",
        phone_number_id="123456",
        fallback_message="Servicio temporalmente no disponible.",
        verify_signature=False,
    )
    channel.attach(runner)

    client = TestClient(channel.app)

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "5491100001111",
                                    "type": "text",
                                    "text": {"body": "Hola bot"},
                                }
                            ]
                        }
                    }
                ]
            }
        ],
    }

    # Must return HTTP 200 (NOT 500!) to prevent Meta retry loops
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["active_agent"] == "fallback"
    assert data["reply"] == "Servicio temporalmente no disponible."


def test_whatsapp_unsupported_media_handling() -> None:
    """Verify that audio/image/document messages are intercepted and given a polite notice."""
    agent = Agent(name="TextOnlyBot", instructions="Only text.")
    runner = Runner(starting_agent=agent)

    channel = WhatsAppChannel(
        verify_token="test_token",
        access_token="mock_meta_token",
        phone_number_id="123456",
        unsupported_media_message="Solo se admiten mensajes de texto por ahora.",
        verify_signature=False,
    )
    channel.attach(runner)

    client = TestClient(channel.app)

    # Payload simulating an audio/voice note message
    audio_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "5491199998888",
                                    "type": "audio",
                                    "audio": {"id": "audio_123"},
                                }
                            ]
                        }
                    }
                ]
            }
        ],
    }

    response = client.post("/webhook", json=audio_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unsupported_media_handled"
    assert data["reply"] == "Solo se admiten mensajes de texto por ahora."


@pytest.mark.asyncio
async def test_telegram_fallback_resilience() -> None:
    """Verify that Telegram channel responds with custom fallback when AI provider errors."""
    failing_agent = Agent(name="TelegramCrash", instructions="Crash.")
    runner = Runner(starting_agent=failing_agent, engine=FailingEngine())

    channel = TelegramChannel(
        token="mock_token",
        fallback_message="Error de red en Telegram, reintenta pronto.",
    )
    channel.attach(runner)

    # Mock telegram update & context
    mock_update = MagicMock()
    mock_update.effective_chat.id = 998877
    mock_update.message.text = "Hello Telegram"
    mock_update.effective_user.username = "test_user"
    mock_update.message.reply_text = AsyncMock()

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock()

    await channel._handle_message(mock_update, mock_context)

    # Assert fallback message was sent to chat
    mock_update.message.reply_text.assert_called_once_with(
        "Error de red en Telegram, reintenta pronto."
    )
