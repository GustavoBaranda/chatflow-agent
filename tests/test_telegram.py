"""Tests for TelegramChannel adapter and event handlers."""

import sys
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import DependencyError
from chatflow_agent.types import AgentResponse


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
async def test_telegram_message_dispatch_splits_long_messages() -> None:
    """Outbound responses exceeding 4096 characters must be split and sent in multiple chunks (SEC-08)."""
    long_response = ("Part 1: " + "A" * 2500) + "\n\n" + ("Part 2: " + "B" * 2500)
    agent = Agent(name="LongBot", instructions="Test")
    runner = Runner(starting_agent=agent, engine=MockEngine(long_response))
    channel = TelegramChannel(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
    channel.attach(runner)

    update = MagicMock()
    update.effective_chat = MagicMock(id=987654)
    update.effective_user = MagicMock(username="tester")
    update.message = AsyncMock(text="Tell me a long story")
    context = MagicMock()
    context.bot = AsyncMock()

    await channel._handle_message(update, context)

    # Must be called twice because 5000+ characters were split into 2 chunks under 4096 chars
    assert update.message.reply_text.call_count == 2
    calls = update.message.reply_text.call_args_list
    assert calls[0].args[0].startswith("Part 1:")
    assert len(calls[0].args[0]) <= 4096
    assert calls[1].args[0].startswith("Part 2:")
    assert len(calls[1].args[0]) <= 4096


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


@pytest.mark.asyncio
async def test_telegram_dispatch_exception_replies_with_fallback(
    telegram_channel: TelegramChannel,
) -> None:
    """When dispatch_async raises, the user must receive fallback_message."""
    telegram_channel.dispatch_async = AsyncMock(
        side_effect=RuntimeError("LLM provider failure / timeout")
    )

    update = MagicMock()
    update.effective_chat = MagicMock(id=424242)
    update.effective_user = MagicMock(username="fallback_user")
    update.message = AsyncMock(text="trigger failure")
    context = MagicMock()
    context.bot = AsyncMock()

    await telegram_channel._handle_message(update, context)

    update.message.reply_text.assert_called_once_with(telegram_channel.fallback_message)


@pytest.mark.asyncio
async def test_telegram_typing_action_failure_does_not_interrupt_processing(
    telegram_channel: TelegramChannel,
) -> None:
    """send_chat_action failures must be swallowed; message handling continues."""
    update = MagicMock()
    update.effective_chat = MagicMock(id=515151)
    update.effective_user = MagicMock(username="typing_user")
    update.message = AsyncMock(text="hello despite typing failure")
    context = MagicMock()
    context.bot = AsyncMock()
    context.bot.send_chat_action = AsyncMock(
        side_effect=RuntimeError("Telegram rate limit / network error")
    )

    await telegram_channel._handle_message(update, context)

    context.bot.send_chat_action.assert_called_once()
    update.message.reply_text.assert_called_once_with("Telegram answer from ChatFlow!")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "build_update",
    [
        pytest.param(
            lambda: MagicMock(
                effective_chat=None,
                message=AsyncMock(text="ignored"),
                effective_user=MagicMock(username="x"),
            ),
            id="missing_effective_chat",
        ),
        pytest.param(
            lambda: MagicMock(
                effective_chat=MagicMock(id=1),
                message=None,
                effective_user=MagicMock(username="x"),
            ),
            id="missing_message",
        ),
        pytest.param(
            lambda: MagicMock(
                effective_chat=MagicMock(id=1),
                message=AsyncMock(text=None),
                effective_user=MagicMock(username="x"),
            ),
            id="message_text_none_media",
        ),
    ],
)
async def test_telegram_non_text_updates_return_early_without_dispatch(
    telegram_channel: TelegramChannel,
    build_update: Callable[[], MagicMock],
) -> None:
    """Updates without chat/message/text must return early and skip dispatch_async."""
    telegram_channel.dispatch_async = AsyncMock(
        return_value=AgentResponse(
            content="should not be used",
            active_agent_name="Telegram Bot",
        )
    )
    update = build_update()
    context = MagicMock()
    context.bot = AsyncMock()

    await telegram_channel._handle_message(update, context)

    telegram_channel.dispatch_async.assert_not_called()
    context.bot.send_chat_action.assert_not_called()
    if update.message is not None:
        update.message.reply_text.assert_not_called()


def test_telegram_raises_dependency_error_when_optional_package_missing() -> None:
    """Instantiating TelegramChannel without python-telegram-bot raises DependencyError."""
    blocked = {
        name: None
        for name in list(sys.modules)
        if name == "telegram" or name.startswith("telegram.")
    }
    blocked.update(
        {
            "telegram": None,
            "telegram.constants": None,
            "telegram.ext": None,
        }
    )

    with patch.dict(sys.modules, blocked):
        with pytest.raises(DependencyError) as exc_info:
            TelegramChannel(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")

    assert exc_info.value.feature == "TelegramChannel"
    assert exc_info.value.extra_package == "telegram"


@pytest.mark.asyncio
async def test_telegram_multi_chat_session_isolation(
    telegram_channel: TelegramChannel,
) -> None:
    """Distinct chat_ids must dispatch to separate runner session IDs without cross-talk."""
    captured_session_ids: list[str] = []
    original_dispatch = telegram_channel.dispatch_async

    async def spy_dispatch(
        session_id: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> AgentResponse:
        captured_session_ids.append(session_id)
        return await original_dispatch(session_id, message, metadata)

    telegram_channel.dispatch_async = spy_dispatch  # type: ignore[method-assign]

    context = MagicMock()
    context.bot = AsyncMock()

    for chat_id, text in ((111001, "message from chat A"), (222002, "message from chat B")):
        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = MagicMock(username=f"user_{chat_id}")
        update.message = AsyncMock(text=text)
        await telegram_channel._handle_message(update, context)

    assert captured_session_ids == ["111001", "222002"]

    assert telegram_channel.runner is not None
    session_a = telegram_channel.runner.get_session("111001")
    session_b = telegram_channel.runner.get_session("222002")
    assert session_a is not None
    assert session_b is not None
    assert session_a is not session_b
    assert session_a.session_id == "111001"
    assert session_b.session_id == "222002"
    user_contents_a = [m.content for m in session_a.history if m.content]
    user_contents_b = [m.content for m in session_b.history if m.content]
    assert "message from chat A" in user_contents_a
    assert "message from chat B" in user_contents_b
    assert "message from chat A" not in user_contents_b
    assert "message from chat B" not in user_contents_a

