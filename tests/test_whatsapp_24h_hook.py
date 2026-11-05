"""on_24h_window_expired runs for every callable shape, and awaitable results are awaited (#175).

Detection used to rely on ``asyncio.iscoroutinefunction``, which misses objects with an
``async __call__`` and plain functions that return a coroutine: their coroutine was created
but never awaited, so the hook silently never ran. The helper is also deprecated in
Python 3.14.
"""

import asyncio
import functools
import gc
import logging
import warnings
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel

PHONE = "5491199999999"


def _channel(hook: Any) -> WhatsAppChannel:
    return WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=0,
        on_24h_window_expired=hook,
    )


async def _send_with_window_expired(channel: WhatsAppChannel) -> tuple[bool, AsyncMock]:
    """Send one chunk against a mocked Meta API that answers error 131047."""
    resp = httpx.Response(
        status_code=400,
        json={"error": {"code": 131047, "message": "Re-engagement message"}},
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        ok = await channel._send_single_outbound_whatsapp(PHONE, "Ping")
    return ok, mock_client


@pytest.fixture
def no_unawaited_coroutines() -> Generator[None, None, None]:
    """Fail if a coroutine created during the test is garbage collected without being awaited."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
        gc.collect()
    unawaited = [w for w in caught if "was never awaited" in str(w.message)]
    assert not unawaited, [str(w.message) for w in unawaited]


def _hook_shapes(calls: list[str]) -> dict[str, Callable[..., Any]]:
    async def async_hook(phone: str) -> None:
        await asyncio.sleep(0)
        calls.append(phone)

    def sync_hook(phone: str) -> None:
        calls.append(phone)

    class AsyncCallable:
        async def __call__(self, phone: str) -> None:
            await asyncio.sleep(0)
            calls.append(phone)

    def returns_coroutine(phone: str) -> Any:
        return async_hook(phone)

    async def async_hook_with_tag(tag: str, phone: str) -> None:
        await asyncio.sleep(0)
        calls.append(f"{tag}:{phone}")

    class CustomAwaitable:
        """An awaitable that is not a coroutine."""

        def __init__(self, phone: str) -> None:
            self.phone = phone

        def __await__(self) -> Generator[Any, None, None]:
            yield from asyncio.sleep(0).__await__()
            calls.append(self.phone)

    return {
        "sync_function": sync_hook,
        "async_function": async_hook,
        "async_callable_instance": AsyncCallable(),
        "function_returning_coroutine": returns_coroutine,
        "partial_of_async_function": functools.partial(async_hook),
        "partial_with_bound_arg": functools.partial(async_hook_with_tag, "tag"),
        "partial_of_async_callable_instance": functools.partial(AsyncCallable()),
        "function_returning_custom_awaitable": CustomAwaitable,
    }


_SHAPES = list(_hook_shapes([]))


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", _SHAPES)
async def test_hook_runs_once_for_every_callable_shape(
    shape: str, no_unawaited_coroutines: None
) -> None:
    calls: list[str] = []
    hook = _hook_shapes(calls)[shape]

    ok, mock_client = await _send_with_window_expired(_channel(hook))

    assert ok is False  # 131047 abandons the chunk
    assert mock_client.post.call_count == 1  # and is never retried
    expected = f"tag:{PHONE}" if shape == "partial_with_bound_arg" else PHONE
    assert calls == [expected]


@pytest.mark.asyncio
async def test_sync_hook_runs_inline_on_the_event_loop_thread() -> None:
    """Synchronous hooks keep running inline, as before: they are not moved to a thread."""
    import threading

    seen: list[int] = []
    channel = _channel(lambda phone: seen.append(threading.get_ident()))

    await _send_with_window_expired(channel)

    assert seen == [threading.get_ident()]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["async_function", "async_callable_instance"])
async def test_async_hook_exception_is_logged_and_does_not_propagate(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    async def failing(phone: str) -> None:
        raise RuntimeError("hook boom")

    class FailingCallable:
        async def __call__(self, phone: str) -> None:
            raise RuntimeError("hook boom")

    hook = failing if kind == "async_function" else FailingCallable()

    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        ok, mock_client = await _send_with_window_expired(_channel(hook))

    assert ok is False
    assert mock_client.post.call_count == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("on_24h_window_expired" in m and "hook boom" in m for m in errors)
    assert all(PHONE not in m for m in errors)  # the phone stays masked


@pytest.mark.asyncio
async def test_async_hook_cancellation_propagates() -> None:
    """CancelledError is not swallowed by the hook guard (it is not an Exception)."""

    class CancellingCallable:
        async def __call__(self, phone: str) -> None:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _send_with_window_expired(_channel(CancellingCallable()))
