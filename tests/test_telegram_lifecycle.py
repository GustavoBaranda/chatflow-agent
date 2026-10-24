"""Lifecycle tests for the Gemini HTTP client scope that ``TelegramChannel.run()`` opens.

These drive the real python-telegram-bot ``Application.run_polling`` (its own event loop, hook
order and shutdown sequence) against a fake Bot API: both PTB request objects are replaced by
``FakeRequest``, which answers ``getMe``, ``getUpdates``, ``sendMessage`` and the rest in
memory. Gemini clients are the recording clients of ``test_gemini_lifecycle``. No external
network is used.
"""

import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import warnings
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("telegram")

from telegram.error import InvalidToken, TelegramError  # noqa: E402
from telegram.ext import Application  # noqa: E402
from telegram.request import BaseRequest, RequestData  # noqa: E402

from chatflow_agent.channels.telegram import TelegramChannel  # noqa: E402
from chatflow_agent.core.engines import GeminiEngine  # noqa: E402
from chatflow_agent.core.engines import gemini as gemini_module  # noqa: E402
from chatflow_agent.core.memory import SessionStore  # noqa: E402
from chatflow_agent.core.runner import Runner  # noqa: E402
from tests.test_gemini_lifecycle import (  # noqa: E402
    API_KEY,
    Clients,
    Gate,
    LoopbackServer,
    bot,
    history_of,
    make_runner,
    resource_warnings,
    text_payload,
)

TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
CHAT_ID = 42


def message_update(update_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 0,
            "chat": {"id": CHAT_ID, "type": "private"},
            "from": {"id": 7, "is_bot": False, "first_name": "Tester", "username": "tester"},
            "text": text,
        },
    }


class FakeBotApi:
    """In-memory Telegram Bot API shared by both of the application's request objects.

    ``getUpdates`` delivers one message update per text, once. Polling is stopped with
    ``stop_running()`` after ``stop_after_replies`` messages were sent (default: one per text).
    A method listed in ``failing`` answers with that HTTP error status instead, and
    ``shutdown_error`` makes the request objects' ``shutdown()`` (so PTB's ``app.shutdown()``)
    raise.
    """

    def __init__(self, texts: list[str], stop_after_replies: int | None = None) -> None:
        self.pending = [message_update(i + 1, text) for i, text in enumerate(texts)]
        self.stop_after_replies = len(texts) if stop_after_replies is None else stop_after_replies
        self.app: Any = None
        self.methods: list[str] = []
        self.sent: list[str] = []
        self.events: list[str] = []
        self.failing: dict[str, int] = {}
        self.shutdown_error: Exception | None = None

    def channel(self, runner: Runner) -> TelegramChannel:
        """A TelegramChannel whose application talks to this fake API instead of Telegram."""
        channel = TelegramChannel(token=TOKEN)
        channel.app = (
            Application.builder()
            .token(TOKEN)
            .request(FakeRequest(self))
            .get_updates_request(FakeRequest(self))
            .build()
        )
        channel._setup_handlers()
        channel.attach(runner)
        self.app = channel.app
        return channel

    async def handle(self, method: str, data: RequestData | None) -> tuple[int, bytes]:
        self.methods.append(method)
        params = data.parameters if data is not None else {}
        result: Any
        if method in self.failing:
            status = self.failing[method]
            return status, json.dumps({"ok": False, "error_code": status, "description": "Fake failure"}).encode()
        if method == "getMe":
            result = {"id": 1, "is_bot": True, "first_name": "Bot", "username": "test_bot"}
        elif method == "getUpdates":
            result, self.pending = self.pending, []
            if not result:
                await asyncio.sleep(0.01)
        elif method == "sendMessage":
            text = str(params["text"])
            self.sent.append(text)
            self.events.append(f"sent {text}")
            result = {"message_id": 100 + len(self.sent), "date": 0, "chat": {"id": CHAT_ID, "type": "private"}, "text": text}
            if len(self.sent) == self.stop_after_replies:
                self.request_stop()
        else:  # deleteWebhook, sendChatAction, ...
            result = True
        return 200, json.dumps({"ok": True, "result": result}).encode()

    def request_stop(self) -> None:
        """Stop polling as a handler calling ``app.stop_running()`` would, one iteration later.

        A handler that never really waits can finish inside PTB's
        ``run_until_complete(app.start())``; a ``loop.stop()`` made there only ends that call,
        and the ``run_forever()`` that follows would never stop. One iteration later the loop
        is in ``run_forever()``.
        """

        def stop() -> None:
            if self.app.running:
                self.app.stop_running()

        asyncio.get_running_loop().call_soon(stop)


class FakeRequest(BaseRequest):
    def __init__(self, api: FakeBotApi) -> None:
        self.api = api

    @property
    def read_timeout(self) -> float | None:
        return None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self.api.shutdown_error is not None:
            raise self.api.shutdown_error

    async def do_request(self, url: str, method: str, request_data: RequestData | None = None, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        return await self.api.handle(url.rsplit("/", 1)[-1], request_data)


@pytest.fixture
def fresh_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A current, not running loop for ``run_polling`` to use (PTB closes it when done)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    asyncio.set_event_loop(None)
    if not loop.is_closed():
        loop.close()


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> Clients:
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    return Clients(monkeypatch)


def run_polling(channel: TelegramChannel) -> None:
    # No PTB signal handlers in the test process; the SIGINT case runs in a subprocess.
    channel.run(stop_signals=None, bootstrap_retries=0)


Hook = Callable[[Any], Awaitable[None]]


def recording_hook(events: list[str], name: str, then: Callable[[Any], Any] | None = None) -> Hook:
    async def hook(app: Any) -> None:
        events.append(name)
        if then is not None:
            result = then(app)
            if asyncio.iscoroutine(result):
                await result

    return hook


# --- Startup and normal shutdown ----------------------------------------------------------


def test_handlers_share_one_pooled_client_that_closes_after_the_user_post_shutdown(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    runner = make_runner()
    api = FakeBotApi(["one", "two", "three"])
    channel = api.channel(runner)
    scope_open_in_user_post_init: list[bool] = []
    pooled_closed_in_user_post_shutdown: list[bool] = []

    def in_post_init(app: Any) -> None:
        scope_open_in_user_post_init.append(runner._client_pool_for_running_loop() is not None)

    def in_post_shutdown(app: Any) -> None:
        pooled_closed_in_user_post_shutdown.extend(c.is_closed for c in clients.pooled)
        clients.pooled[0].on_close = lambda: api.events.append("pooled client closed")

    user_post_init = recording_hook(api.events, "user post_init", in_post_init)
    user_post_shutdown = recording_hook(api.events, "user post_shutdown", in_post_shutdown)
    channel.app.post_init = user_post_init
    channel.app.post_shutdown = user_post_shutdown

    run_polling(channel)

    assert api.sent == ["ok", "ok", "ok"]
    assert api.events == [
        "user post_init",
        "sent ok",
        "sent ok",
        "sent ok",
        "user post_shutdown",
        "pooled client closed",
    ]
    assert scope_open_in_user_post_init == [True]  # the user's post_init already runs in the scope
    assert pooled_closed_in_user_post_shutdown == [False]  # ... and so does their post_shutdown
    assert clients.attempt == []
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 3
    assert pooled.created_on is fresh_loop
    assert pooled.close_calls == 1
    assert pooled.closed_while_loop_running is True
    assert fresh_loop.is_closed()  # PTB closed its loop after post_shutdown
    assert runner._client_pools == {}
    assert channel.app.post_init is user_post_init
    assert channel.app.post_shutdown is user_post_shutdown


def test_without_user_hooks_the_attributes_are_restored_to_none(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    runner = make_runner()
    api = FakeBotApi(["hi"])
    channel = api.channel(runner)

    run_polling(channel)

    assert api.sent == ["ok"]
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert channel.app.post_init is None
    assert channel.app.post_shutdown is None
    assert runner._client_pools == {}


def test_each_run_opens_and_closes_its_own_pool(clients: Clients) -> None:
    runner = make_runner()
    api = FakeBotApi([])
    channel = api.channel(runner)
    events: list[str] = []
    channel.app.post_init = recording_hook(events, "user post_init")
    channel.app.post_shutdown = recording_hook(events, "user post_shutdown")

    loops = []
    for text in ("first", "second"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loops.append(loop)
        api.pending = [message_update(len(loops), text)]
        api.stop_after_replies = len(loops)
        try:
            run_polling(channel)
        finally:
            asyncio.set_event_loop(None)

    assert api.sent == ["ok", "ok"]
    # The wrappers are not stacked: each user hook runs exactly once per run.
    assert events == ["user post_init", "user post_shutdown"] * 2
    assert [c.created_on for c in clients.pooled] == loops
    assert all(c.close_calls == 1 and c.closed_while_loop_running for c in clients.pooled)
    assert clients.attempt == []
    assert runner._client_pools == {}


# --- Startup failures ---------------------------------------------------------------------


def test_failed_bootstrap_opens_nothing_and_post_shutdown_is_harmless(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    runner = make_runner()
    api = FakeBotApi(["never delivered"])
    api.failing["getMe"] = 401
    channel = api.channel(runner)
    events: list[str] = []
    user_post_init = recording_hook(events, "user post_init")
    user_post_shutdown = recording_hook(events, "user post_shutdown")
    channel.app.post_init = user_post_init
    channel.app.post_shutdown = user_post_shutdown

    with pytest.raises(InvalidToken):
        run_polling(channel)

    # PTB skips post_init but still runs post_shutdown; the empty scope closes as a no-op.
    assert events == ["user post_shutdown"]
    assert "getUpdates" not in api.methods
    assert clients.created == []
    assert runner._client_pools == {}
    assert channel.app.post_init is user_post_init
    assert channel.app.post_shutdown is user_post_shutdown
    assert fresh_loop.is_closed()


def test_a_failing_user_post_init_still_closes_the_scope(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    runner = make_runner()
    api = FakeBotApi(["never delivered"])
    channel = api.channel(runner)
    events: list[str] = []

    async def warm_up_then_fail(app: Any) -> None:
        events.append("user post_init")
        await runner.run_async("warm-up", "hello")  # uses the pooled client
        raise ValueError("post_init failed")

    user_post_shutdown = recording_hook(events, "user post_shutdown")
    channel.app.post_init = warm_up_then_fail
    channel.app.post_shutdown = user_post_shutdown

    with pytest.raises(ValueError, match="post_init failed"):
        run_polling(channel)

    assert events == ["user post_init", "user post_shutdown"]
    assert api.sent == []
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 1
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}
    assert channel.app.post_init is warm_up_then_fail
    assert channel.app.post_shutdown is user_post_shutdown


def test_a_failing_user_post_shutdown_still_closes_the_scope(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    runner = make_runner()
    api = FakeBotApi(["hi"])
    channel = api.channel(runner)

    async def fail(app: Any) -> None:
        raise ValueError("post_shutdown failed")

    channel.app.post_shutdown = fail

    with pytest.raises(ValueError, match="post_shutdown failed"):
        run_polling(channel)

    assert api.sent == ["ok"]
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert channel.app.post_init is None
    assert channel.app.post_shutdown is fail


def test_attributes_are_restored_when_run_polling_raises_before_any_hook(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    runner = make_runner()
    channel = FakeBotApi([]).channel(runner)
    user_post_init = recording_hook([], "user post_init")
    channel.app.post_init = user_post_init

    with pytest.raises(TypeError):
        channel.run(stop_signals=None, no_such_argument=True)

    assert not fresh_loop.is_closed()  # PTB never ran it, so it is not closed either
    assert channel.app.post_init is user_post_init
    assert channel.app.post_shutdown is None
    assert runner._client_pools == {}


def test_a_loop_created_for_run_is_closed_and_unset_when_ptb_fails_before_running_it(
    clients: Clients, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off the main thread (and on 3.14+, anywhere) there is no current loop, so run() creates
    one, as PTB would. If PTB then fails before running it, run() closes that loop and unsets
    it again, so the thread is left as it was found."""
    created: list[asyncio.AbstractEventLoop] = []
    real_new_event_loop = asyncio.new_event_loop

    def recording_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = real_new_event_loop()
        created.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", recording_new_event_loop)
    runner = make_runner()
    channel = FakeBotApi([]).channel(runner)
    outcome: dict[str, Any] = {}

    def in_thread() -> None:
        try:
            channel.run(stop_signals=None, no_such_argument=True)
        except TypeError as err:
            outcome["error"] = err
        try:
            outcome["current loop"] = asyncio.get_event_loop()
        except RuntimeError:  # no current loop in this thread
            outcome["current loop"] = None

    thread = threading.Thread(target=in_thread)
    try:
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        assert isinstance(outcome["error"], TypeError)
        [loop] = created
        assert loop.is_closed()
        assert outcome["current loop"] is None
        assert channel.app.post_init is None and channel.app.post_shutdown is None
        assert runner._client_pools == {}
    finally:
        for loop in created:  # don't leak the loop if an assertion failed
            if not loop.is_closed():
                loop.close()


def test_a_startup_failure_after_post_init_closes_the_scope(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    """Starting the updater (deleteWebhook) fails after post_init opened the scope and used it."""
    runner = make_runner()
    api = FakeBotApi(["never delivered"])
    api.failing["deleteWebhook"] = 500
    channel = api.channel(runner)

    async def warm_up(app: Any) -> None:
        await runner.run_async("warm-up", "hello")

    channel.app.post_init = warm_up

    with pytest.raises(TelegramError):
        run_polling(channel)

    assert "getUpdates" not in api.methods
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 1
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}
    assert channel.app.post_init is warm_up


def test_stop_running_in_the_user_post_init_closes_the_scope(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    """PTB's documented early exit: app.stop_running() in post_init skips polling and shuts down."""
    runner = make_runner()
    api = FakeBotApi(["never delivered"])
    channel = api.channel(runner)

    async def warm_up_then_stop(app: Any) -> None:
        await runner.run_async("warm-up", "hello")
        app.stop_running()

    channel.app.post_init = warm_up_then_stop

    run_polling(channel)

    assert "getUpdates" not in api.methods
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}


# --- Shutdown with work in flight: no premature close ---------------------------------------


def test_shutdown_waits_for_the_handler_in_flight_before_closing_the_client(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    """Polling is stopped while a handler's Gemini request is in flight on the pooled client.
    PTB's stop() lets the handler finish and reply; only then does post_shutdown close it."""
    runner = make_runner()
    api = FakeBotApi(["slow"], stop_after_replies=0)
    channel = api.channel(runner)
    gate = Gate()

    async def stop_polling_while_in_flight(request: httpx.Request) -> httpx.Response:
        api.events.append("request in flight")
        api.request_stop()
        asyncio.get_running_loop().call_later(0.05, gate.release.set)
        return await gate(request)

    async def watch_close(app: Any) -> None:
        clients.pooled[0].on_close = lambda: api.events.append("pooled client closed")

    clients.reply = stop_polling_while_in_flight
    channel.app.post_shutdown = watch_close

    run_polling(channel)

    assert api.events == ["request in flight", "sent released", "pooled client closed"]
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert clients.attempt == []


def test_scope_exit_waits_for_a_spawned_turn_that_ptb_does_not_track(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    """A turn started with runner.spawn_turn is not one of PTB's tasks, so stop() does not
    wait for it. If its request is in flight at post_shutdown, the scope exit waits for that
    request (it is not cut off), then closes the client."""
    runner_box: list[Runner] = []
    spawned: list[asyncio.Task[Any]] = []
    gate = Gate()

    async def follow_up() -> str:
        """Start a follow-up turn."""
        spawned.append(runner_box[0].spawn_turn("background", "SLOW"))
        return "started"

    async def scripted(request: httpx.Request) -> httpx.Response:
        body = request.content
        if b"SLOW" in body:
            return await gate(request)
        if b"functionResponse" in body:
            return httpx.Response(200, json=text_payload("done"))
        call = {"functionCall": {"name": "follow_up", "args": {}}}
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [call]}}]})

    runner = make_runner(bot(tools=[follow_up]))
    runner_box.append(runner)
    api = FakeBotApi(["go"])
    channel = api.channel(runner)

    async def release_during_post_shutdown(app: Any) -> None:
        assert gate.entered.is_set() and not spawned[0].done()
        clients.pooled[0].on_close = lambda: api.events.append("pooled client closed")
        asyncio.get_running_loop().call_later(0.05, lambda: (api.events.append("released"), gate.release.set()))

    clients.reply = scripted
    channel.app.post_shutdown = release_during_post_shutdown

    run_polling(channel)

    assert api.events == ["sent done", "released", "pooled client closed"]
    assert spawned[0].done() and spawned[0].result().content == "released"
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 3
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert clients.attempt == []


def stop_signal() -> None:
    raise SystemExit  # what PTB's stop-signal handler does inside the loop on SIGINT/SIGTERM


def test_a_stop_signal_during_the_user_post_init_cancels_it_before_the_scope_closes(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    """A stop signal while the user's post_init waits on Gemini makes PTB abandon post_init
    and shut down. Left alone, post_init would run on during shutdown, open a per-attempt
    client for its next request, and be destroyed pending at loop.close() with that client
    open. It is cancelled first instead, so no client outlives the loop."""
    runner = make_runner()
    api = FakeBotApi(["never delivered"])
    channel = api.channel(runner)
    gate = Gate()

    async def signal_while_in_flight(request: httpx.Request) -> httpx.Response:
        api.events.append("request in flight")
        loop = asyncio.get_running_loop()
        loop.call_soon(stop_signal)
        loop.call_later(0.05, gate.release.set)  # without the cancel, post_init would go on
        return await gate(request)

    async def user_post_init(app: Any) -> None:
        api.events.append("user post_init")
        try:
            await runner.run_async("warm-up", "first")
            await runner.run_async("warm-up", "second")
        except asyncio.CancelledError:
            api.events.append("user post_init cancelled")
            raise

    async def user_post_shutdown(app: Any) -> None:
        api.events.append("user post_shutdown")
        clients.pooled[0].on_close = lambda: api.events.append("pooled client closed")

    clients.reply = signal_while_in_flight
    channel.app.post_init = user_post_init
    channel.app.post_shutdown = user_post_shutdown

    run_polling(channel)  # PTB handles the stop signal; run() returns normally

    assert api.events == [
        "user post_init",
        "request in flight",
        "user post_init cancelled",
        "user post_shutdown",
        "pooled client closed",
    ]
    assert "getUpdates" not in api.methods
    assert clients.attempt == []
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert history_of(runner, "warm-up") == []  # the cancelled turn was rolled back


# --- PTB shutdown failing or interrupted before post_shutdown --------------------------------
# PTB runs post_shutdown last in its `finally`; if an earlier step raises, it skips it and
# goes straight to closing the loop. run() then closes the scope itself, before the loop.


def test_a_failing_post_stop_skips_post_shutdown_but_the_connection_is_still_closed(
    loopback: LoopbackServer, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    runner = make_runner()
    api = FakeBotApi(["one", "two"])
    channel = api.channel(runner)
    events: list[str] = []

    async def fail(app: Any) -> None:
        raise ValueError("post_stop failed")

    user_post_shutdown = recording_hook(events, "user post_shutdown")
    channel.app.post_stop = fail
    channel.app.post_shutdown = user_post_shutdown

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="post_stop failed"):
            run_polling(channel)
        leaked = resource_warnings(record)

    assert events == []  # PTB skipped post_shutdown, and run() does not call it either
    assert api.sent == ["ok", "ok"]
    assert loopback.opened == 1
    assert loopback.wait_all_closed()  # the pooled connection was closed, not left to the runner
    assert leaked == []
    assert runner._client_pools == {}
    assert fresh_loop.is_closed()
    assert channel.app.post_shutdown is user_post_shutdown


def test_a_failing_ptb_shutdown_step_still_closes_the_pooled_client_before_the_loop(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    runner = make_runner()
    api = FakeBotApi(["hi"])
    api.shutdown_error = RuntimeError("request shutdown failed")  # makes app.shutdown() raise
    channel = api.channel(runner)
    events: list[str] = []
    channel.app.post_shutdown = recording_hook(events, "user post_shutdown")

    with pytest.raises(RuntimeError, match="request shutdown failed"):
        run_polling(channel)

    assert events == []
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert fresh_loop.is_closed()


def test_a_cleanup_error_does_not_replace_the_error_polling_failed_with(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop, caplog: pytest.LogCaptureFixture
) -> None:
    runner = make_runner()
    api = FakeBotApi(["hi"])
    channel = api.channel(runner)

    async def fail_and_break_the_close(app: Any) -> None:
        def broken_close() -> None:
            raise OSError("close failed")

        clients.pooled[0].on_close = broken_close
        raise ValueError("post_stop failed")

    channel.app.post_stop = fail_and_break_the_close

    with pytest.raises(ValueError, match="post_stop failed"):
        run_polling(channel)

    assert "close failed" in caplog.text
    assert clients.pooled[0].close_calls == 1
    assert runner._client_pools == {}
    assert fresh_loop.is_closed()


def test_a_stop_signal_during_the_user_post_shutdown_still_closes_the_scope(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    """A second stop signal while post_shutdown runs ends PTB's run_until_complete with
    post_shutdown still pending. run() cancels it; its `finally` closes the scope."""
    runner = make_runner()
    api = FakeBotApi(["hi"])
    channel = api.channel(runner)

    async def slow_post_shutdown(app: Any) -> None:
        api.events.append("user post_shutdown")
        clients.pooled[0].on_close = lambda: api.events.append("pooled client closed")
        asyncio.get_running_loop().call_soon(stop_signal)
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            api.events.append("user post_shutdown cancelled")
            raise

    channel.app.post_shutdown = slow_post_shutdown

    with pytest.raises(SystemExit):
        run_polling(channel)

    assert api.events == ["sent ok", "user post_shutdown", "user post_shutdown cancelled", "pooled client closed"]
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert fresh_loop.is_closed()
    assert channel.app.post_shutdown is slow_post_shutdown


def test_a_stop_signal_while_the_scope_waits_for_a_request_closes_it_and_keeps_close_loop_false(
    clients: Clients, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    """The scope exit in post_shutdown waits for a spawned turn's in-flight request when a
    stop signal arrives. run() cancels post_shutdown, so the exit stops waiting and closes the
    client at once. With close_loop=False the loop is left open, as PTB would leave it."""
    runner_box: list[Runner] = []
    spawned: list[asyncio.Task[Any]] = []
    gate = Gate()

    async def follow_up() -> str:
        """Start a follow-up turn."""
        spawned.append(runner_box[0].spawn_turn("background", "SLOW"))
        return "started"

    async def scripted(request: httpx.Request) -> httpx.Response:
        if b"SLOW" in request.content:
            return await gate(request)
        if b"functionResponse" in request.content:
            return httpx.Response(200, json=text_payload("done"))
        call = {"functionCall": {"name": "follow_up", "args": {}}}
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [call]}}]})

    runner = make_runner(bot(tools=[follow_up]))
    runner_box.append(runner)
    api = FakeBotApi(["go"])
    channel = api.channel(runner)

    async def signal_during_scope_exit(app: Any) -> None:
        assert gate.entered.is_set()
        pooled = clients.pooled[0]
        pooled.on_close = lambda: (api.events.append("pooled client closed"), gate.release.set())
        asyncio.get_running_loop().call_later(0.05, stop_signal)  # lands while the exit waits

    clients.reply = scripted
    channel.app.post_shutdown = signal_during_scope_exit

    with pytest.raises(SystemExit):
        channel.run(stop_signals=None, bootstrap_retries=0, close_loop=False)

    assert api.events == ["sent done", "pooled client closed"]
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert not fresh_loop.is_closed()  # close_loop=False is honoured
    fresh_loop.run_until_complete(asyncio.wait(spawned))  # the loop is still usable
    assert spawned[0].result().content == "released"  # its request was released by the close


def test_close_loop_false_leaves_the_loop_open_and_it_can_poll_again(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    runner = make_runner()
    api = FakeBotApi(["first"])
    channel = api.channel(runner)

    channel.run(stop_signals=None, bootstrap_retries=0, close_loop=False)
    assert not fresh_loop.is_closed()
    api.pending = [message_update(2, "second")]
    api.stop_after_replies = 2
    channel.run(stop_signals=None, bootstrap_retries=0, close_loop=False)

    assert not fresh_loop.is_closed()
    assert api.sent == ["ok", "ok"]
    assert [c.created_on for c in clients.pooled] == [fresh_loop, fresh_loop]
    assert all(c.close_calls == 1 and c.closed_while_loop_running for c in clients.pooled)
    assert runner._client_pools == {}


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
def test_run_called_inside_the_running_loop_fails_without_closing_it(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    """PTB can't run on a loop that is already running (here: from the user's post_init).
    That call must fail as PTB fails, and must not close the loop the outer run() is using."""
    runner = make_runner()
    api = FakeBotApi(["hi"])
    channel = api.channel(runner)
    errors: list[BaseException] = []

    async def run_again(app: Any) -> None:
        try:
            channel.run(stop_signals=None)
        except RuntimeError as err:
            errors.append(err)
        assert not asyncio.get_running_loop().is_closed()

    channel.app.post_init = run_again

    run_polling(channel)
    gc.collect()  # the PTB coroutines the inner call never awaited, under this test's filter

    assert [str(e) for e in errors] == ["This event loop is already running"]
    assert api.sent == ["ok"]  # the outer run went on normally
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert fresh_loop.is_closed()
    assert channel.app.post_init is run_again


# --- Ownership: an injected client is the caller's -------------------------------------------


def test_an_injected_client_is_used_by_handlers_and_never_closed(clients: Clients, fresh_loop: asyncio.AbstractEventLoop) -> None:
    sent: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=text_payload("injected"))

    injected = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    runner = Runner(bot(), session_store=SessionStore(), engine=GeminiEngine(api_key=API_KEY, http_client=injected))
    api = FakeBotApi(["a", "b"])

    run_polling(api.channel(runner))

    assert api.sent == ["injected", "injected"]
    assert len(sent) == 2
    assert not injected.is_closed
    assert clients.created == []  # neither a pooled nor a per-attempt client
    assert runner._client_pools == {}
    asyncio.run(injected.aclose())


# --- Real sockets -------------------------------------------------------------------------


@pytest.fixture
def loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[LoopbackServer]:
    server = LoopbackServer()
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.setattr(gemini_module, "_API_BASE_URL", server.base_url)
    yield server
    server.stop()


def test_loopback_polling_reuses_one_connection_and_closes_it_at_shutdown(
    loopback: LoopbackServer, fresh_loop: asyncio.AbstractEventLoop
) -> None:
    runner = make_runner()
    api = FakeBotApi(["one", "two", "three"])
    channel = api.channel(runner)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        run_polling(channel)
        leaked = resource_warnings(record)

    assert api.sent == ["ok", "ok", "ok"]
    assert loopback.requests == 3
    assert loopback.opened == 1
    assert loopback.wait_all_closed()
    assert leaked == []
    assert runner._client_pools == {}


SIGINT_DURING_A_HANDLER = """
import asyncio, json, signal
import httpx
from chatflow_agent.core.engines import gemini as gemini_module
from tests.test_gemini_lifecycle import make_runner, text_payload
from tests.test_telegram_lifecycle import FakeBotApi

api = FakeBotApi(["hi"], stop_after_replies=0)

async def reply(request):
    api.events.append("request in flight")
    signal.raise_signal(signal.SIGINT)  # Ctrl-C while the handler waits on Gemini
    await asyncio.sleep(0.05)
    return httpx.Response(200, json=text_payload("ok"))

class Pooled(httpx.AsyncClient):
    async def aclose(self):
        api.events.append("pooled client closed" if asyncio.get_running_loop().is_running() else "closed off-loop")
        await super().aclose()

gemini_module._new_pooled_client = lambda: Pooled(transport=httpx.MockTransport(reply))

async def post_shutdown(app):
    api.events.append("user post_shutdown")

runner = make_runner()
channel = api.channel(runner)
channel.app.post_shutdown = post_shutdown
asyncio.set_event_loop(asyncio.new_event_loop())
channel.run(bootstrap_retries=0)  # PTB's default stop signals: SIGINT, SIGTERM, SIGABRT
print(json.dumps({"events": api.events, "pools": len(runner._client_pools)}))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="PTB installs no stop-signal handlers on Windows")
def test_ctrl_c_during_a_handler_lets_it_finish_then_closes_the_client() -> None:
    """A real SIGINT, in a subprocess so PTB's signal handlers stay out of the test process.
    PTB turns it into a graceful stop: the in-flight handler finishes and replies, then the
    user's post_shutdown runs, then the pooled client closes while the loop still runs."""
    repo_root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_GEMINI_BASE_URL"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", SIGINT_DURING_A_HANDLER],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])

    assert outcome["events"] == ["request in flight", "sent ok", "user post_shutdown", "pooled client closed"]
    assert outcome["pools"] == 0
