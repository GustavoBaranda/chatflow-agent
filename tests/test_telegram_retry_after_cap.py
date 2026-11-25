"""Regression tests for #211: bound TelegramChannel RetryAfter flood-control wait time."""

import asyncio
import datetime
import math
import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import RetryAfter
from telegram.warnings import PTBDeprecationWarning

from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.exceptions import ConfigurationError
from chatflow_agent.types import AgentResponse


def _make_retry_after(value: int | float | datetime.timedelta) -> RetryAfter:
    """Construct ``RetryAfter`` while suppressing only PTB's numeric-to-timedelta deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PTBDeprecationWarning)
        return RetryAfter(value)  # type: ignore[arg-type]


def _make_channel(**kwargs: Any) -> TelegramChannel:
    with patch("telegram.ext.Application.builder") as mock_builder:
        mock_app = MagicMock()
        mock_builder.return_value.token.return_value.concurrent_updates.return_value.build.return_value = (
            mock_app
        )
        return TelegramChannel(token="123456:FAKE_TELEGRAM_TOKEN", **kwargs)


def _make_update(text: str = "hello", chat_id: int = 42) -> tuple[MagicMock, MagicMock]:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user.username = "tester"
    update.message.text = text
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    return update, context


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 0.0, 1, 5.5, 60.0, 120])
def test_max_retry_after_seconds_accepts_valid_non_negative_finite_numbers(
    value: float,
) -> None:
    channel = _make_channel(max_retry_after_seconds=value)
    assert channel.max_retry_after_seconds == float(value)


def test_max_retry_after_seconds_default_is_60_and_keyword_only() -> None:
    channel = _make_channel()
    assert channel.max_retry_after_seconds == 60.0
    # Must be keyword-only: passing a 5th positional argument raises TypeError.
    with pytest.raises(TypeError):
        TelegramChannel("tok", "start", "fallback", None, 30.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_value",
    [True, False, -1, -0.01, math.nan, math.inf, -math.inf, "60", None],
)
def test_max_retry_after_seconds_rejects_invalid_values(bad_value: Any) -> None:
    with pytest.raises(ConfigurationError, match="max_retry_after_seconds"):
        _make_channel(max_retry_after_seconds=bad_value)


@pytest.mark.parametrize(
    "bad_value_factory",
    [
        pytest.param(lambda: 10**1000, id="10**1000"),
        pytest.param(lambda: 10**10000, id="10**10000"),
        pytest.param(lambda: -(10**10000), id="-(10**10000)"),
    ],
)
def test_max_retry_after_seconds_rejects_extreme_integers_with_configuration_error(
    bad_value_factory: Callable[[], int],
) -> None:
    """Extreme integers overflowing float must raise ConfigurationError, never ValueError from str conversion."""
    with pytest.raises(ConfigurationError, match="max_retry_after_seconds"):
        _make_channel(max_retry_after_seconds=bad_value_factory())


# ---------------------------------------------------------------------------
# RetryAfter bounding behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retry_after_arg",
    [3600, 60.1, datetime.timedelta(hours=1)],
)
async def test_retry_after_exceeding_cap_aborts_immediately_without_sleeping(
    retry_after_arg: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Delays above max_retry_after_seconds must not sleep, must log safely, and return False."""
    channel = _make_channel(max_retry_after_seconds=60.0)
    update, _ = _make_update()
    update.message.reply_text = AsyncMock(side_effect=_make_retry_after(retry_after_arg))

    sleep_mock = AsyncMock()
    with (
        caplog.at_level("WARNING", logger="chatflow_agent.telegram"),
        patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock),
    ):
        ok = await channel._send_chunk(update, "secret reply text")

    assert ok is False
    assert sleep_mock.await_count == 0
    assert update.message.reply_text.await_count == 1
    error_msgs = [r.message for r in caplog.records if r.levelname == "ERROR"]
    assert any("max_retry_after_seconds" in m and "RetryAfter" in m for m in error_msgs)
    assert "secret reply text" not in caplog.text
    assert "FAKE_TELEGRAM_TOKEN" not in caplog.text


@pytest.mark.parametrize(
    "retry_after_arg,expected_sleep",
    [
        (5, 5.0),
        (60.0, 60.0),
        (datetime.timedelta(seconds=12), 12.0),
        (0, 0.0),
    ],
)
async def test_retry_after_within_cap_retries_normally_and_succeeds(
    retry_after_arg: Any,
    expected_sleep: float,
) -> None:
    """Delays <= max_retry_after_seconds continue to sleep and retry normally."""
    channel = _make_channel(max_retry_after_seconds=60.0)
    update, _ = _make_update()
    update.message.reply_text = AsyncMock(
        side_effect=[_make_retry_after(retry_after_arg), None]
    )

    sleep_mock = AsyncMock()
    with patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock):
        ok = await channel._send_chunk(update, "hello")

    assert ok is True
    assert update.message.reply_text.await_count == 2
    sleep_mock.assert_awaited_once_with(expected_sleep)


async def test_retry_after_zero_cap_allows_zero_delay_and_aborts_positive_delay() -> None:
    """max_retry_after_seconds=0.0 permits 0s RetryAfter but aborts any positive delay."""
    channel = _make_channel(max_retry_after_seconds=0.0)
    update, _ = _make_update()

    # 0s delay succeeds
    update.message.reply_text = AsyncMock(side_effect=[_make_retry_after(0), None])
    sleep_mock = AsyncMock()
    with patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock):
        assert await channel._send_chunk(update, "ok") is True
    sleep_mock.assert_awaited_once_with(0.0)

    # Positive delay aborts immediately
    update.message.reply_text = AsyncMock(side_effect=_make_retry_after(1))
    sleep_mock.reset_mock()
    with patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock):
        assert await channel._send_chunk(update, "blocked") is False
    assert sleep_mock.await_count == 0


async def test_consecutive_within_cap_retries_still_exhaust_at_max_outbound_retries() -> None:
    """Consecutive RetryAfter errors within cap exhaust after 3 retries (4 attempts total)."""
    channel = _make_channel(max_retry_after_seconds=30.0)
    update, _ = _make_update()
    update.message.reply_text = AsyncMock(side_effect=_make_retry_after(2))

    sleep_mock = AsyncMock()
    with patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock):
        ok = await channel._send_chunk(update, "hello")

    assert ok is False
    assert update.message.reply_text.await_count == 4
    assert sleep_mock.await_count == 3


async def test_partial_multi_chunk_delivery_aborts_on_excessive_retry_after_without_fallback() -> None:
    """Chunk 1 succeeds; chunk 2 hits excessive RetryAfter -> no sleep, no duplicate chunk, no fallback."""
    channel = _make_channel(max_retry_after_seconds=30.0, fallback_message="FALLBACK")
    long_reply = ("Chunk1: " + "A" * 2500) + "\n\n" + ("Chunk2: " + "B" * 2500)
    channel.dispatch_async = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResponse(content=long_reply, active_agent_name="Bot")
    )

    update, context = _make_update()
    sent: list[str] = []

    async def fake_reply(text: str) -> None:
        if len(sent) == 1:
            raise _make_retry_after(3600)
        sent.append(text)

    update.message.reply_text = AsyncMock(side_effect=fake_reply)
    sleep_mock = AsyncMock()
    with patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock):
        await channel._handle_message(update, context)

    assert len(sent) == 1
    assert sent[0].startswith("Chunk1:")
    assert "FALLBACK" not in sent
    assert sleep_mock.await_count == 0
    assert update.message.reply_text.await_count == 2


async def test_cancellation_during_bounded_retry_after_sleep_propagates() -> None:
    """asyncio.CancelledError during an in-bounds RetryAfter sleep propagates cleanly."""
    channel = _make_channel(max_retry_after_seconds=60.0)
    update, _ = _make_update()
    update.message.reply_text = AsyncMock(side_effect=_make_retry_after(10))

    started = asyncio.Event()

    async def slow_sleep(_delay: float) -> None:
        started.set()
        await asyncio.Future()

    with patch("chatflow_agent.channels.telegram.asyncio.sleep", slow_sleep):
        task = asyncio.create_task(channel._send_chunk(update, "hello"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_default_channel_aborts_excessive_retry_after_3600_without_sleeping() -> None:
    """Default TelegramChannel(token=...) aborts RetryAfter(3600) with 0 sleeps and returns False."""
    channel = _make_channel()
    update, _ = _make_update()
    update.message.reply_text = AsyncMock(side_effect=_make_retry_after(3600))

    sleep_mock = AsyncMock()
    with patch("chatflow_agent.channels.telegram.asyncio.sleep", sleep_mock):
        ok = await channel._send_chunk(update, "hello")

    assert ok is False
    assert sleep_mock.await_count == 0
    assert update.message.reply_text.await_count == 1
