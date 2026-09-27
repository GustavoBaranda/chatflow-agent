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


@pytest.mark.asyncio
async def test_telegram_reset_allowed_in_private_chat(
    telegram_channel: TelegramChannel,
) -> None:
    """In private 1-on-1 chats, /reset must be allowed without admin restrictions."""
    update = MagicMock()
    update.effective_chat = MagicMock(id=111, type="private")
    update.message = AsyncMock()
    context = MagicMock()
    context.bot = AsyncMock()

    session = telegram_channel.runner.session_store.get_or_create(
        session_id="111", default_agent=telegram_channel.runner.starting_agent
    )
    session.history.append(MagicMock())
    assert len(session.history) > 0

    await telegram_channel._handle_reset(update, context)

    context.bot.get_chat_member.assert_not_called()
    assert len(session.history) == 0
    update.message.reply_text.assert_called_once_with(
        "Conversation memory has been reset."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["administrator", "creator"])
async def test_telegram_reset_allowed_for_group_admin(
    telegram_channel: TelegramChannel, status: str
) -> None:
    """In group/supergroup chats, /reset must succeed if caller is admin or creator."""
    chat_id = -100123456
    user_id = 42
    update = MagicMock()
    update.effective_chat = MagicMock(id=chat_id, type="supergroup")
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()

    context = MagicMock()
    member = MagicMock(status=status)
    context.bot = AsyncMock()
    context.bot.get_chat_member = AsyncMock(return_value=member)

    session = telegram_channel.runner.session_store.get_or_create(
        session_id=str(chat_id), default_agent=telegram_channel.runner.starting_agent
    )
    session.history.append(MagicMock())
    assert len(session.history) > 0

    await telegram_channel._handle_reset(update, context)

    context.bot.get_chat_member.assert_called_once_with(
        chat_id=chat_id, user_id=user_id
    )
    assert len(session.history) == 0
    update.message.reply_text.assert_called_once_with(
        "Conversation memory has been reset."
    )


@pytest.mark.asyncio
async def test_telegram_reset_rejected_for_regular_group_member(
    telegram_channel: TelegramChannel,
) -> None:
    """In group/supergroup chats, /reset must be rejected if caller is a regular member."""
    chat_id = -100987654
    user_id = 99
    update = MagicMock()
    update.effective_chat = MagicMock(id=chat_id, type="group")
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()

    context = MagicMock()
    member = MagicMock(status="member")
    context.bot = AsyncMock()
    context.bot.get_chat_member = AsyncMock(return_value=member)

    session = telegram_channel.runner.session_store.get_or_create(
        session_id=str(chat_id), default_agent=telegram_channel.runner.starting_agent
    )
    session.history.append(MagicMock())
    assert len(session.history) > 0

    await telegram_channel._handle_reset(update, context)

    context.bot.get_chat_member.assert_called_once_with(
        chat_id=chat_id, user_id=user_id
    )
    # Memory must NOT be reset
    assert len(session.history) > 0
    update.message.reply_text.assert_called_once_with(
        "Only group admins can reset the conversation."
    )


@pytest.mark.asyncio
async def test_telegram_reset_fail_closed_on_api_error(
    telegram_channel: TelegramChannel,
) -> None:
    """If get_chat_member raises an error (network, API, etc.), fail-closed: do not reset."""
    chat_id = -100555555
    user_id = 88
    update = MagicMock()
    update.effective_chat = MagicMock(id=chat_id, type="supergroup")
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()

    context = MagicMock()
    context.bot = AsyncMock()
    context.bot.get_chat_member = AsyncMock(
        side_effect=RuntimeError("Telegram API timeout / network failure")
    )

    session = telegram_channel.runner.session_store.get_or_create(
        session_id=str(chat_id), default_agent=telegram_channel.runner.starting_agent
    )
    session.history.append(MagicMock())
    assert len(session.history) > 0

    await telegram_channel._handle_reset(update, context)

    # Memory must NOT be reset (fail-closed)
    assert len(session.history) > 0
    update.message.reply_text.assert_called_once_with(
        "Only group admins can reset the conversation."
    )

