"""Tests for TelegramChannel adapter and event handlers."""

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ConfigurationError, DependencyError, ReentrantTurnError
from chatflow_agent.types import AgentResponse, Message, Role


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
    update.message.reply_text.assert_called_once_with("Telegram answer from ChatFlow!")


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
async def test_telegram_empty_reply_uses_fallback_message() -> None:
    """Empty/whitespace engine replies must send fallback_message, not an empty body (#169)."""
    agent = Agent(name="EmptyBot", instructions="Test")
    runner = Runner(starting_agent=agent, engine=MockEngine("   "))
    channel = TelegramChannel(
        token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        fallback_message="FALLBACK_TG",
    )
    channel.attach(runner)

    update = MagicMock()
    update.effective_chat = MagicMock(id=987654)
    update.effective_user = MagicMock(username="tester")
    update.message = AsyncMock(text="hi")
    context = MagicMock()
    context.bot = AsyncMock()

    await channel._handle_message(update, context)

    update.message.reply_text.assert_called_once_with("FALLBACK_TG")


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
    update.message.reply_text.assert_called_once_with("Conversation memory has been reset.")


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
    update.message.reply_text.assert_called_once_with("Conversation memory has been reset.")


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

    context.bot.get_chat_member.assert_called_once_with(chat_id=chat_id, user_id=user_id)
    assert len(session.history) == 0
    update.message.reply_text.assert_called_once_with("Conversation memory has been reset.")


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

    context.bot.get_chat_member.assert_called_once_with(chat_id=chat_id, user_id=user_id)
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


class GatedEngine(BaseEngine):
    """Deterministic engine controlled by per-message asyncio.Events."""

    def __init__(self) -> None:
        self.entered: dict[str, asyncio.Event] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.fail_messages: set[str] = set()
        self.call_order: list[tuple[str, list[str | None]]] = []

    def entered_event(self, text: str) -> asyncio.Event:
        return self.entered.setdefault(text, asyncio.Event())

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        user_text = next((m.content for m in reversed(history) if m.role == Role.USER), "") or ""
        self.call_order.append((user_text, [m.content for m in history]))
        self.entered_event(user_text).set()
        gate = self.gates.get(user_text)
        if gate is not None:
            await gate.wait()
        if user_text in self.fail_messages:
            raise RuntimeError(f"Simulated provider failure for {user_text}")
        return EngineTurnResult(text=f"reply:{user_text}")


def _build_text_update(chat_id: int, text: str, username: str = "user") -> tuple[Any, Any]:
    update = MagicMock()
    update.effective_chat = MagicMock(id=chat_id, type="private")
    update.effective_user = MagicMock(id=chat_id + 1000, username=username)
    update.message = AsyncMock(text=text)
    context = MagicMock()
    context.bot = AsyncMock()
    return update, context


def _build_reset_update(
    chat_id: int,
    *,
    chat_type: str = "private",
    user_id: int = 10,
    member_status: str | Exception = "administrator",
) -> tuple[Any, Any]:
    update = MagicMock()
    update.effective_chat = MagicMock(id=chat_id, type=chat_type)
    update.effective_user = MagicMock(id=user_id, username=f"user_{user_id}")
    update.message = AsyncMock()
    context = MagicMock()
    context.bot = AsyncMock()
    if isinstance(member_status, Exception):
        context.bot.get_chat_member = AsyncMock(side_effect=member_status)
    else:
        context.bot.get_chat_member = AsyncMock(return_value=MagicMock(status=member_status))
    return update, context


def test_telegram_default_constructor_backward_compatibility() -> None:
    """Default constructor remains sequential (max_concurrent_updates == 1) and compatible (#170)."""
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    channel = TelegramChannel(token, "Custom start", "Custom fallback", ["message"])
    assert channel.token == token
    assert channel.start_message == "Custom start"
    assert channel.fallback_message == "Custom fallback"
    assert channel.allowed_updates == ["message"]
    assert channel.concurrent_updates is False
    assert channel.app.update_processor.max_concurrent_updates == 1


def test_telegram_concurrent_updates_configuration() -> None:
    """concurrent_updates is keyword-only and configures PTB's update processor (#170)."""
    from telegram.ext import SimpleUpdateProcessor

    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"

    channel_true = TelegramChannel(token=token, concurrent_updates=True)
    assert channel_true.concurrent_updates is True
    assert channel_true.app.update_processor.max_concurrent_updates == 256

    channel_int = TelegramChannel(token=token, concurrent_updates=8)
    assert channel_int.concurrent_updates == 8
    assert channel_int.app.update_processor.max_concurrent_updates == 8

    custom_processor = SimpleUpdateProcessor(4)
    channel_custom = TelegramChannel(token=token, concurrent_updates=custom_processor)
    assert channel_custom.concurrent_updates is custom_processor
    assert channel_custom.app.update_processor is custom_processor

    # Positional 5th argument must be rejected because concurrent_updates is keyword-only
    with pytest.raises(TypeError):
        TelegramChannel(token, "start", "fallback", ["message"], True)  # type: ignore[misc]


@pytest.mark.parametrize("invalid_value", [0, -1, -100, "true", None, 1.5, object()])
def test_telegram_invalid_concurrent_updates_raises_configuration_error(
    invalid_value: Any,
) -> None:
    """Invalid concurrent_updates values fail fast with ConfigurationError (a ValueError) (#170)."""
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    with pytest.raises(ConfigurationError):
        TelegramChannel(token=token, concurrent_updates=invalid_value)


@pytest.mark.asyncio
async def test_telegram_slow_turn_in_chat_a_does_not_block_chat_b() -> None:
    """When updates run concurrently, a slow turn in chat A does not block chat B (#170)."""
    engine = GatedEngine()
    engine.gates["slow-A"] = asyncio.Event()
    runner = Runner(starting_agent=Agent(name="Bot", instructions="x"), engine=engine)
    channel = TelegramChannel(
        token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        concurrent_updates=True,
    )
    channel.attach(runner)

    update_a, context_a = _build_text_update(1001, "slow-A")
    update_b, context_b = _build_text_update(2002, "fast-B")

    task_a = asyncio.create_task(channel._handle_message(update_a, context_a))
    await asyncio.wait_for(engine.entered_event("slow-A").wait(), timeout=2.0)
    assert not task_a.done()

    # Chat B must complete while Chat A is still blocked on its gate
    await asyncio.wait_for(channel._handle_message(update_b, context_b), timeout=2.0)
    update_b.message.reply_text.assert_called_once_with("reply:fast-B")
    assert not task_a.done()

    engine.gates["slow-A"].set()
    await asyncio.wait_for(task_a, timeout=2.0)
    update_a.message.reply_text.assert_called_once_with("reply:slow-A")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sqlite", [False, True])
async def test_telegram_same_chat_turns_remain_serialized(
    use_sqlite: bool, tmp_path: Path
) -> None:
    """Two concurrent updates in the same chat remain serialized by Runner's session lock (#170)."""
    engine = GatedEngine()
    engine.gates["first"] = asyncio.Event()
    store = SQLiteSessionStore(tmp_path / "tg_serial.db") if use_sqlite else SessionStore()
    try:
        runner = Runner(
            starting_agent=Agent(name="Bot", instructions="x"),
            session_store=store,
            engine=engine,
        )
        channel = TelegramChannel(
            token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            concurrent_updates=True,
        )
        channel.attach(runner)

        update_1, context_1 = _build_text_update(777, "first")
        update_2, context_2 = _build_text_update(777, "second")

        task_1 = asyncio.create_task(channel._handle_message(update_1, context_1))
        await asyncio.wait_for(engine.entered_event("first").wait(), timeout=2.0)

        task_2 = asyncio.create_task(channel._handle_message(update_2, context_2))
        await asyncio.sleep(0.05)

        # Second turn must not enter the engine until the first turn finishes
        assert not engine.entered_event("second").is_set()
        assert not task_2.done()

        engine.gates["first"].set()
        await asyncio.wait_for(asyncio.gather(task_1, task_2), timeout=2.0)

        assert [call[0] for call in engine.call_order] == ["first", "second"]
        assert engine.call_order[1][1] == ["first", "reply:first", "second"]
    finally:
        if isinstance(store, SQLiteSessionStore):
            store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sqlite", [False, True])
async def test_telegram_reset_waits_for_in_flight_turn_in_same_session(
    use_sqlite: bool, tmp_path: Path
) -> None:
    """/reset waits for an in-flight turn of the same session and does not block other chats (#170)."""
    engine = GatedEngine()
    engine.gates["in-flight"] = asyncio.Event()
    store = SQLiteSessionStore(tmp_path / "tg_reset_wait.db") if use_sqlite else SessionStore()
    try:
        runner = Runner(
            starting_agent=Agent(name="Bot", instructions="x"),
            session_store=store,
            engine=engine,
        )
        channel = TelegramChannel(
            token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            concurrent_updates=True,
        )
        channel.attach(runner)

        await runner.run_async("555", "seed")

        update_msg, context_msg = _build_text_update(555, "in-flight")
        turn_task = asyncio.create_task(channel._handle_message(update_msg, context_msg))
        await asyncio.wait_for(engine.entered_event("in-flight").wait(), timeout=2.0)

        update_reset, context_reset = _build_reset_update(555)
        reset_task = asyncio.create_task(channel._handle_reset(update_reset, context_reset))
        await asyncio.sleep(0.05)

        # /reset must wait while the turn holds the session lock
        assert not reset_task.done()
        update_reset.message.reply_text.assert_not_called()

        # Meanwhile, another chat can still run and reset freely
        update_other, context_other = _build_text_update(999, "other-chat")
        await asyncio.wait_for(channel._handle_message(update_other, context_other), timeout=2.0)
        update_other.message.reply_text.assert_called_once_with("reply:other-chat")

        # Release the in-flight turn; turn completes first, then reset clears history
        engine.gates["in-flight"].set()
        await asyncio.wait_for(turn_task, timeout=2.0)
        await asyncio.wait_for(reset_task, timeout=2.0)

        update_msg.message.reply_text.assert_called_once_with("reply:in-flight")
        update_reset.message.reply_text.assert_called_once_with(
            "Conversation memory has been reset."
        )
        session = runner.get_session("555")
        assert session is not None
        assert session.history == []
    finally:
        if isinstance(store, SQLiteSessionStore):
            store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sqlite", [False, True])
async def test_telegram_reset_cannot_corrupt_failed_turn_rollback(
    use_sqlite: bool, tmp_path: Path
) -> None:
    """A /reset queued during a failing turn waits for rollback and then clears cleanly;
    a failing turn queued after /reset rolls back to the post-reset empty state (#170)."""
    engine = GatedEngine()
    engine.gates["boom"] = asyncio.Event()
    engine.fail_messages.add("boom")
    engine.fail_messages.add("boom-after-reset")
    store = SQLiteSessionStore(tmp_path / "tg_rollback.db") if use_sqlite else SessionStore()
    try:
        runner = Runner(
            starting_agent=Agent(name="Bot", instructions="x"),
            session_store=store,
            engine=engine,
        )
        channel = TelegramChannel(
            token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            concurrent_updates=True,
        )
        channel.attach(runner)

        await runner.run_async("888", "initial")

        # Case A: /reset arrives while a turn is in flight and that turn subsequently fails
        update_msg, context_msg = _build_text_update(888, "boom")
        turn_task = asyncio.create_task(channel._handle_message(update_msg, context_msg))
        await asyncio.wait_for(engine.entered_event("boom").wait(), timeout=2.0)

        update_reset, context_reset = _build_reset_update(888)
        reset_task = asyncio.create_task(channel._handle_reset(update_reset, context_reset))
        await asyncio.sleep(0.05)
        assert not reset_task.done()

        # Before releasing the failing turn, history still has initial + boom user message
        session_during = runner.get_session("888")
        assert session_during is not None
        assert [m.content for m in session_during.history] == ["initial", "reply:initial", "boom"]

        engine.gates["boom"].set()
        await asyncio.wait_for(turn_task, timeout=2.0)
        await asyncio.wait_for(reset_task, timeout=2.0)

        update_msg.message.reply_text.assert_called_once_with(channel.fallback_message)
        update_reset.message.reply_text.assert_called_once_with(
            "Conversation memory has been reset."
        )
        session_after = runner.get_session("888")
        assert session_after is not None
        assert session_after.history == []

        # Case B: A subsequent failing turn after /reset must roll back to empty history
        # (never resurrecting pre-reset messages)
        await runner.run_async("888", "before-second-reset")
        assert await runner.reset_session_async("888") is True
        with pytest.raises(RuntimeError, match="Simulated provider failure"):
            await runner.run_async("888", "boom-after-reset")
        session_final = runner.get_session("888")
        assert session_final is not None
        assert session_final.history == []
    finally:
        if isinstance(store, SQLiteSessionStore):
            store.close()


@pytest.mark.asyncio
async def test_telegram_group_reset_authorization_stays_fail_closed_under_concurrency() -> None:
    """Unauthorized or failed group /reset checks return before waiting on or touching the session (#170)."""
    engine = GatedEngine()
    engine.gates["group-turn"] = asyncio.Event()
    runner = Runner(starting_agent=Agent(name="Bot", instructions="x"), engine=engine)
    channel = TelegramChannel(
        token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        concurrent_updates=True,
    )
    channel.attach(runner)

    chat_id = -100777888
    await runner.run_async(str(chat_id), "preserved")

    update_msg, context_msg = _build_text_update(chat_id, "group-turn")
    turn_task = asyncio.create_task(channel._handle_message(update_msg, context_msg))
    await asyncio.wait_for(engine.entered_event("group-turn").wait(), timeout=2.0)

    # 1. Non-admin member attempt completes immediately without waiting for the in-flight turn
    update_member, ctx_member = _build_reset_update(
        chat_id, chat_type="supergroup", user_id=50, member_status="member"
    )
    await asyncio.wait_for(channel._handle_reset(update_member, ctx_member), timeout=2.0)
    update_member.message.reply_text.assert_called_once_with(
        "Only group admins can reset the conversation."
    )

    # 2. API error during get_chat_member also fails closed immediately
    update_err, ctx_err = _build_reset_update(
        chat_id,
        chat_type="group",
        user_id=51,
        member_status=RuntimeError("API down"),
    )
    await asyncio.wait_for(channel._handle_reset(update_err, ctx_err), timeout=2.0)
    update_err.message.reply_text.assert_called_once_with(
        "Only group admins can reset the conversation."
    )

    engine.gates["group-turn"].set()
    await asyncio.wait_for(turn_task, timeout=2.0)

    session = runner.get_session(str(chat_id))
    assert session is not None
    assert [m.content for m in session.history] == [
        "preserved",
        "reply:preserved",
        "group-turn",
        "reply:group-turn",
    ]


@pytest.mark.asyncio
async def test_runner_reset_session_reentrancy_and_missing_session() -> None:
    """Runner.reset_session_async refuses same-session reentrant calls and returns False when absent (#170)."""
    agent = Agent(name="ToolBot", instructions="x")
    runner_holder: list[Runner] = []
    caught_reentrant: list[ReentrantTurnError] = []

    @agent.tool
    async def attempt_self_reset() -> str:
        try:
            await runner_holder[0].reset_session_async("self-session")
        except ReentrantTurnError as err:
            caught_reentrant.append(err)
            raise
        return "unreachable"

    class ToolCallingEngine(BaseEngine):
        def __init__(self) -> None:
            self.calls = 0

        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            self.calls += 1
            if self.calls == 1:
                from chatflow_agent.types import ToolCall

                return EngineTurnResult(
                    tool_calls=[ToolCall(id="call_1", name="attempt_self_reset", args={})]
                )
            tool_msg = history[-1]
            assert tool_msg.tool_results[0].is_error is True
            return EngineTurnResult(text="handled reentrant error")

    runner = Runner(starting_agent=agent, engine=ToolCallingEngine())
    runner_holder.append(runner)

    # Non-existent session returns False without creating a session
    assert await runner.reset_session_async("missing-session") is False
    assert runner.get_session("missing-session") is None

    # Direct reentrant check via _turn_lineage inside tool execution
    response = await runner.run_async("self-session", "try reset")
    assert response.content == "handled reentrant error"
    assert len(caught_reentrant) == 1
    session = runner.get_session("self-session")
    assert session is not None
    assert len(session.history) > 0

    # Sync wrapper rejects calls from a running event loop
    with pytest.raises(RuntimeError, match="Runner.reset_session\\(\\) cannot be called"):
        runner.reset_session("self-session")
