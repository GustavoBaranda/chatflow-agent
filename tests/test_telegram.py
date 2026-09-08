"""Tests for TelegramChannel adapter and event handlers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner


class MockEngine(GeminiEngine):
    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply

    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text=self.reply)


@pytest.fixture
def telegram_channel() -> TelegramChannel:
    agent = Agent(name="Telegram Bot", instructions="Telegram prompt.")
    runner = Runner(
        starting_agent=agent,
        engine=MockEngine("Telegram answer from ChatFlow!"),
    )
    # Using dummy token for testing handlers
    channel = TelegramChannel(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
    channel.attach(runner)
    return channel


@pytest.mark.asyncio
async def test_telegram_start_handler(telegram_channel: TelegramChannel) -> None:
    update = MagicMock()
    update.message = AsyncMock()
    context = MagicMock()

    await telegram_channel._handle_start(update, context)
    update.message.reply_text.assert_called_once_with(telegram_channel.start_message)


@pytest.mark.asyncio
async def test_telegram_message_dispatch(telegram_channel: TelegramChannel) -> None:
    update = MagicMock()
    update.effective_chat = MagicMock(id=987654)
    update.effective_user = MagicMock(username="tester")
    update.message = AsyncMock(text="Where are you located?")
    context = MagicMock()
    context.bot = AsyncMock()

    await telegram_channel._handle_message(update, context)

    # 1. Typing action sent
    context.bot.send_chat_action.assert_called_once()

    # 2. Reply sent with runner output
    update.message.reply_text.assert_called_once_with(
        "Telegram answer from ChatFlow!"
    )


@pytest.mark.asyncio
async def test_telegram_reset_handler(telegram_channel: TelegramChannel) -> None:
    update = MagicMock()
    update.effective_chat = MagicMock(id=987654)
    update.message = AsyncMock()
    context = MagicMock()

    # Populate some session history
    session = telegram_channel.runner.session_store.get_or_create(
        session_id="987654", default_agent=telegram_channel.runner.starting_agent
    )
    assert session is not None

    await telegram_channel._handle_reset(update, context)
    assert len(session.history) == 0
    update.message.reply_text.assert_called_once_with(
        "Conversation memory has been reset."
    )
