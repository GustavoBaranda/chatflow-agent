"""Lifecycle tests for GeminiEngine HTTP clients and the Runner's ``async with runner:`` scope.

Engine-created clients are replaced by recording clients (``httpx.MockTransport``) to check
who creates, uses, and closes each client and on which event loop. A local 127.0.0.1 server
checks real TCP connection reuse and closing. No external network is used.
"""

import asyncio
import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Awaitable, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from chatflow_agent.core import runner as runner_module
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine, EngineTurnResult, GeminiEngine
from chatflow_agent.core.engines import gemini as gemini_module
from chatflow_agent.core.memory import SessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall

API_KEY = "lifecycle-test-key"


def text_payload(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def call_payload(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}}]}


Reply = Callable[[httpx.Request], Awaitable[httpx.Response]]


async def reply_text(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=text_payload("ok"))


class RecordingClient(httpx.AsyncClient):
    """Records the loop that created it, the loops it sent on, and how it was closed."""

    def __init__(self, kind: str, reply: Reply, **kwargs: Any) -> None:
        super().__init__(transport=httpx.MockTransport(reply), **kwargs)
        self.kind = kind
        self.created_on = asyncio.get_running_loop()
        self.sent_on: list[asyncio.AbstractEventLoop] = []
        self.closed_while_loop_running: bool | None = None
        self.close_calls = 0
        self.on_close: Callable[[], None] | None = None

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        loop = asyncio.get_running_loop()
        assert loop is self.created_on, "client used on a different event loop"
        self.sent_on.append(loop)
        return await super().send(request, **kwargs)

    async def aclose(self) -> None:
        self.closed_while_loop_running = asyncio.get_running_loop().is_running()
        self.close_calls += 1
        if self.on_close is not None:
            self.on_close()
        await super().aclose()

    async def __aexit__(self, *exc_info: Any) -> None:
        # httpx closes the transport here directly, without calling aclose().
        self.closed_while_loop_running = asyncio.get_running_loop().is_running()
        self.close_calls += 1
        await super().__aexit__(*exc_info)


class Clients:
    """Installs recording factories for pooled and per-attempt clients."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reply: Reply = reply_text
        self.created: list[RecordingClient] = []
        self._lock = threading.Lock()
        monkeypatch.setattr(gemini_module, "_new_pooled_client", lambda: self._make("pooled"))
        monkeypatch.setattr(gemini_module, "_new_attempt_client", lambda timeout: self._make("attempt", timeout=timeout))

    def _make(self, kind: str, **kwargs: Any) -> RecordingClient:
        client = RecordingClient(kind, lambda request: self.reply(request), **kwargs)
        with self._lock:
            self.created.append(client)
        return client

    def of_kind(self, kind: str) -> list[RecordingClient]:
        return [c for c in self.created if c.kind == kind]

    @property
    def pooled(self) -> list[RecordingClient]:
        return self.of_kind("pooled")

    @property
    def attempt(self) -> list[RecordingClient]:
        return self.of_kind("attempt")


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> Clients:
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    return Clients(monkeypatch)


def bot(**kwargs: Any) -> Agent:
    kwargs.setdefault("instructions", "Be helpful.")
    return Agent(name=kwargs.pop("name", "Bot"), model="gemini-2.5-flash", **kwargs)


def make_runner(agent: Agent | None = None, **engine_kwargs: Any) -> Runner:
    engine_kwargs.setdefault("base_delay", 0.001)
    return Runner(agent or bot(), session_store=SessionStore(), engine=GeminiEngine(api_key=API_KEY, **engine_kwargs))


class Gate:
    """A reply that blocks inside the HTTP attempt until released."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.entered.set()
        await self.release.wait()
        return httpx.Response(200, json=text_payload("released"))


# --- Reuse inside a scope -----------------------------------------------------------------


async def test_scope_reuses_one_pooled_client_across_turns_and_tool_iterations(clients: Clients) -> None:
    def lookup(order_id: str) -> str:
        """Look up an order."""
        return "shipped"

    replies = iter([call_payload("lookup", {"order_id": "A1"}), text_payload("Shipped."), text_payload("2"), text_payload("3")])

    async def scripted(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(replies))

    clients.reply = scripted
    runner = make_runner(bot(tools=[lookup]))
    loop = asyncio.get_running_loop()

    async with runner as entered:
        assert entered is runner
        assert clients.created == []  # entering opens nothing
        first = await runner.run_async("s1", "Where is A1?")
        await runner.run_async("s1", "again")
        await runner.run_async("s2", "other session")

    assert first.tool_calls_executed == ["lookup"]
    assert clients.attempt == []
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 4
    assert pooled.is_closed
    assert pooled.closed_while_loop_running is True
    assert pooled.created_on is loop


async def test_default_and_factory_built_engines_share_the_pooled_client(clients: Clients) -> None:
    specialist = Agent(name="Specialist", instructions="Specialist.", model="gemini-2.5-flash", provider="gemini", api_key="specialist-key")
    triage = bot(handoffs=[specialist])
    handoff_tool = triage.get_all_tools()[0].name
    replies = iter([call_payload(handoff_tool, {}), text_payload("Specialist here.")])
    seen_keys: list[str] = []

    async def scripted(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers["x-goog-api-key"])
        return httpx.Response(200, json=next(replies))

    clients.reply = scripted
    runner = make_runner(triage)
    async with runner:
        response = await runner.run_async("s1", "help")

    assert response.active_agent_name == "Specialist"
    assert seen_keys == [API_KEY, "specialist-key"]  # each engine still sends its own key
    assert clients.attempt == []
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 2
    assert pooled.is_closed


async def test_pooled_client_does_not_carry_cookies_between_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    # The pooled client is shared by engines with different API keys: a cookie set by a
    # gateway on one response must not be sent with another engine's request.
    seen: list[httpx.Request] = []

    async def set_cookie(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=text_payload("ok"), headers={"Set-Cookie": "gw_session=tenant-a; Path=/"})

    real_factory = gemini_module._new_pooled_client

    def pooled_with_mock_transport() -> httpx.AsyncClient:
        client = real_factory()
        client._transport = httpx.MockTransport(set_cookie)  # keep the real client's cookie settings
        return client

    monkeypatch.setattr(gemini_module, "_new_pooled_client", pooled_with_mock_transport)
    tenant_a = Runner(bot(), session_store=SessionStore(), engine=GeminiEngine(api_key="key-a"))
    other = Agent(name="Other", instructions="x", model="gemini-2.5-flash", provider="gemini", api_key="key-b")

    async with tenant_a:
        await tenant_a.run_async("s1", "first")
        lent = tenant_a._client_pool_for_running_loop()
        token = gemini_module._lent_pool.set(lent)
        try:
            await GeminiEngine(api_key="key-b").generate_turn_async(other, [Message(role=Role.USER, content="second")])
        finally:
            gemini_module._lent_pool.reset(token)

    assert len(seen) == 2
    assert seen[1].headers["x-goog-api-key"] == "key-b"
    assert "cookie" not in seen[1].headers


async def test_concurrent_first_requests_in_a_scope_create_one_pooled_client(clients: Clients) -> None:
    runner = make_runner()
    async with runner:
        responses = await asyncio.gather(*(runner.run_async(f"s{i}", "hi") for i in range(20)))

    assert [r.content for r in responses] == ["ok"] * 20
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 20
    assert clients.attempt == []


async def test_a_scope_without_gemini_requests_creates_no_client(clients: Clients) -> None:
    class StubEngine(GeminiEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            return EngineTurnResult(text="stub")

    runner = Runner(bot(), session_store=SessionStore(), engine=StubEngine())
    async with runner:
        assert (await runner.run_async("s1", "hi")).content == "stub"
    assert clients.created == []


# --- Outside a scope and injected clients ------------------------------------------------


async def test_outside_a_scope_each_attempt_uses_and_closes_its_own_client(clients: Clients) -> None:
    runner = make_runner()
    await runner.run_async("s1", "one")
    await runner.run_async("s1", "two")

    assert clients.pooled == []
    assert len(clients.attempt) == 2
    assert all(c.is_closed and c.closed_while_loop_running for c in clients.attempt)


async def test_injected_client_is_used_in_and_out_of_scopes_and_never_closed(clients: Clients) -> None:
    sent: list[httpx.Request] = []

    async def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=text_payload("injected"))

    injected = httpx.AsyncClient(transport=httpx.MockTransport(record))
    runner = Runner(bot(), session_store=SessionStore(), engine=GeminiEngine(api_key=API_KEY, http_client=injected))

    async with runner:
        await runner.run_async("s1", "in scope")
    await runner.run_async("s1", "out of scope")

    assert len(sent) == 2
    assert clients.created == []
    assert not injected.is_closed
    await injected.aclose()


# --- Scope exit -------------------------------------------------------------------------


async def test_exit_waits_for_in_flight_attempts_and_new_attempts_use_their_own_client(clients: Clients) -> None:
    gate = Gate()
    clients.reply = gate
    runner = make_runner()

    await runner.__aenter__()
    in_flight = asyncio.create_task(runner.run_async("s1", "slow"))
    await gate.entered.wait()
    [pooled] = clients.pooled

    exiting = asyncio.create_task(runner.__aexit__(None, None, None))
    await asyncio.sleep(0.01)
    assert not exiting.done()  # waiting for the in-flight attempt
    assert not pooled.is_closed

    clients.reply = reply_text
    after_exit = await runner.run_async("s2", "started after exit began")
    assert after_exit.content == "ok"
    assert len(clients.attempt) == 1  # not the pooled client, which is closing
    assert clients.attempt[0].is_closed

    gate.release.set()
    assert (await in_flight).content == "released"  # completed normally, not cut off
    await asyncio.wait_for(exiting, timeout=2)
    assert pooled.is_closed
    assert len(pooled.sent_on) == 1


async def test_exit_waits_for_every_in_flight_attempt_before_closing(clients: Clients) -> None:
    entered = 0
    both_entered = asyncio.Event()
    releases = [asyncio.Event(), asyncio.Event()]

    async def held(request: httpx.Request) -> httpx.Response:
        nonlocal entered
        index = entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await releases[index].wait()
        return httpx.Response(200, json=text_payload(f"reply {index}"))

    clients.reply = held
    runner = make_runner()
    await runner.__aenter__()
    first = asyncio.create_task(runner.run_async("s1", "one"))
    second = asyncio.create_task(runner.run_async("s2", "two"))
    await both_entered.wait()
    [pooled] = clients.pooled

    exiting = asyncio.create_task(runner.__aexit__(None, None, None))
    await asyncio.sleep(0.01)
    assert not exiting.done()

    releases[0].set()
    assert (await first).content == "reply 0"
    await asyncio.sleep(0.01)
    assert not exiting.done()  # the second attempt still uses the client
    assert not pooled.is_closed

    releases[1].set()
    assert (await second).content == "reply 1"
    await asyncio.wait_for(exiting, timeout=2)
    assert pooled.is_closed
    assert pooled.close_calls == 1
    assert len(pooled.sent_on) == 2


class RetryAcrossExit:
    """The first reply is a 503; the scope then exits while the engine backs off."""

    def __init__(self, later_status: int) -> None:
        self.later_status = later_status
        self.calls = 0
        self.first_reply_sent = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            self.first_reply_sent.set()
            return httpx.Response(503, json={"error": "unavailable"})
        if self.later_status != 200:
            return httpx.Response(self.later_status, json={"error": "unavailable"})
        return httpx.Response(200, json=text_payload("after exit"))


async def exit_during_backoff(clients: Clients, reply: RetryAcrossExit, **engine_kwargs: Any) -> asyncio.Task[Any]:
    clients.reply = reply
    runner = make_runner(base_delay=0.2, **engine_kwargs)
    await runner.__aenter__()
    turn = asyncio.create_task(runner.run_async("s1", "hi"))
    await reply.first_reply_sent.wait()
    await runner.__aexit__(None, None, None)  # returns while the engine is still backing off
    assert not turn.done()
    [pooled] = clients.pooled
    assert pooled.is_closed
    assert runner._client_pools == {}
    return turn


async def test_retry_after_scope_exit_uses_a_new_client_not_the_closed_pool(clients: Clients) -> None:
    reply = RetryAcrossExit(later_status=200)
    turn = await exit_during_backoff(clients, reply, max_retries=1)

    assert (await turn).content == "after exit"
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 1  # the retry did not touch the closed client
    [attempt] = clients.attempt
    assert len(attempt.sent_on) == 1
    assert all(c.is_closed for c in clients.created)


async def test_retry_failures_after_scope_exit_raise_provider_error_not_runtime_error(clients: Clients) -> None:
    reply = RetryAcrossExit(later_status=503)
    turn = await exit_during_backoff(clients, reply, max_retries=1)

    with pytest.raises(ProviderError, match=r"\(503\)"):
        await turn
    assert reply.calls == 2
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 1
    assert len(clients.attempt) == 1
    assert all(c.is_closed for c in clients.created)


async def test_exit_never_cancels_spawned_turns(clients: Clients) -> None:
    runner = make_runner()
    async with runner:
        spawned = runner.spawn_turn("s1", "follow-up")
    response = await asyncio.wait_for(spawned, timeout=2)

    assert not spawned.cancelled()
    assert response.content == "ok"
    assert clients.pooled == []  # the spawned turn's request started after the scope closed
    assert len(clients.attempt) == 1 and clients.attempt[0].is_closed


async def test_exit_cancelled_while_waiting_still_closes_the_pooled_client(clients: Clients) -> None:
    gate = Gate()
    clients.reply = gate
    runner = make_runner()

    await runner.__aenter__()
    in_flight = asyncio.create_task(runner.run_async("s1", "slow"))
    await gate.entered.wait()
    exiting = asyncio.create_task(runner.__aexit__(None, None, None))
    await asyncio.sleep(0.01)
    exiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exiting

    [pooled] = clients.pooled
    assert pooled.is_closed
    gate.release.set()
    await in_flight
    # The scope is gone: later requests use their own clients.
    await runner.run_async("s2", "after")
    assert len(clients.attempt) == 1


async def test_body_exception_propagates_and_the_pooled_client_is_closed(clients: Clients) -> None:
    runner = make_runner()
    with pytest.raises(ValueError, match="app failure"):
        async with runner:
            await runner.run_async("s1", "hi")
            raise ValueError("app failure")

    [pooled] = clients.pooled
    assert pooled.is_closed and pooled.closed_while_loop_running


async def test_provider_errors_and_cancelled_attempts_leave_the_pool_usable(clients: Clients) -> None:
    runner = make_runner(max_retries=0)

    async def bad_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    async with runner:
        clients.reply = bad_request
        with pytest.raises(ProviderError):
            await runner.run_async("s1", "fails")

        gate = Gate()
        clients.reply = gate
        cancelled = asyncio.create_task(runner.run_async("s2", "cancelled"))
        await gate.entered.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        clients.reply = reply_text
        assert (await runner.run_async("s3", "works")).content == "ok"

    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 3
    assert pooled.is_closed
    assert clients.attempt == []


async def test_a_turn_timeout_during_a_pooled_request_releases_the_client_and_rolls_back(clients: Clients) -> None:
    """turn_timeout cancels the turn mid-request: the pooled client is released, the turn rolled
    back, and the scope still closes the client once."""
    runner = make_runner()
    loop = asyncio.get_running_loop()
    in_use_after_timeout: list[int] = []

    async def scenario() -> None:
        async with runner:
            assert (await runner.run_async("s1", "first")).content == "ok"
            gate = Gate()
            clients.reply = gate
            with pytest.raises(asyncio.TimeoutError):
                await runner.run_async("s1", "slow", turn_timeout=0.5)
            assert gate.entered.is_set()
            in_use_after_timeout.append(runner._client_pools[loop][0]._in_use)
            assert history_of(runner, "s1") == ["first", "ok"]

            clients.reply = reply_text
            assert (await runner.run_async("s1", "after")).content == "ok"

    # Bounded: if the timed-out attempt kept the client, the scope exit would wait for it forever.
    await asyncio.wait_for(scenario(), timeout=5)
    assert in_use_after_timeout == [0]
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 3
    assert pooled.is_closed and pooled.close_calls == 1
    assert pooled.closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}


# --- Scope contract --------------------------------------------------------------------


async def test_nested_scopes_close_only_at_the_outermost_exit(clients: Clients) -> None:
    runner = make_runner()
    async with runner:
        async with runner:
            await runner.run_async("s1", "inner")
        [pooled] = clients.pooled
        assert not pooled.is_closed
        await runner.run_async("s1", "outer")
        assert clients.pooled == [pooled]
    assert pooled.is_closed
    assert len(pooled.sent_on) == 2


async def test_nested_entry_after_a_request_keeps_the_same_client_and_closes_it_once(clients: Clients) -> None:
    runner = make_runner()
    async with runner:
        await runner.run_async("s1", "before the inner scope")  # creates the pooled client
        [pooled] = clients.pooled
        async with runner:
            await runner.run_async("s1", "inner")
            assert clients.pooled == [pooled]
        assert not pooled.is_closed
        await runner.run_async("s1", "after the inner scope")
        assert pooled.close_calls == 0

    assert clients.pooled == [pooled]
    assert len(pooled.sent_on) == 3
    assert pooled.is_closed
    assert pooled.close_calls == 1
    assert clients.attempt == []


async def test_runner_is_reusable_after_a_scope_closes(clients: Clients) -> None:
    runner = make_runner()
    async with runner:
        await runner.run_async("s1", "first scope")
    await runner.run_async("s1", "between scopes")
    async with runner:
        await runner.run_async("s1", "second scope")

    first, second = clients.pooled
    assert first is not second
    assert first.is_closed and second.is_closed
    assert len(clients.attempt) == 1


async def test_exit_without_an_open_scope_raises_runtime_error() -> None:
    runner = make_runner()
    with pytest.raises(RuntimeError, match="without an open"):
        await runner.__aexit__(None, None, None)


def test_scopes_on_different_threads_get_separate_pools(clients: Clients) -> None:
    runner = make_runner()
    barrier = threading.Barrier(2, timeout=5)

    async def wait_for_other_thread(request: httpx.Request) -> httpx.Response:
        await asyncio.to_thread(barrier.wait)  # both threads are inside their scope at once
        return httpx.Response(200, json=text_payload("ok"))

    clients.reply = wait_for_other_thread
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        async def main() -> None:
            async with runner:
                await runner.run_async(f"thread-{i}", "hi")

        try:
            asyncio.run(main())
        except BaseException as err:  # surfaced below
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(clients.pooled) == 2
    assert clients.pooled[0].created_on is not clients.pooled[1].created_on
    assert all(c.is_closed and c.closed_while_loop_running for c in clients.pooled)
    assert runner._client_pools == {}


async def test_lent_pool_is_not_used_from_another_event_loop(clients: Clients) -> None:
    inner = GeminiEngine(api_key=API_KEY)

    class ThreadHoppingEngine(GeminiEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            # The lent pool's context variable is copied into the worker thread and into the
            # asyncio.run there, but that is a different event loop.
            return await asyncio.to_thread(lambda: asyncio.run(inner.generate_turn_async(agent, history)))

    runner = Runner(bot(), session_store=SessionStore(), engine=ThreadHoppingEngine())
    async with runner:
        assert (await runner.run_async("s1", "hi")).content == "ok"

    assert clients.pooled == []
    [attempt] = clients.attempt
    assert attempt.is_closed and attempt.closed_while_loop_running


# --- Runner.run(): one scope per call, on the call's own event loop ----------------------


def answers_a_tool_call(request: httpx.Request) -> bool:
    return any("functionResponse" in part for part in json.loads(request.content)["contents"][-1]["parts"])


def call_then_text(name: str) -> Reply:
    """Ask for tool ``name``, then answer "done" once the request carries its result."""

    async def reply(request: httpx.Request) -> httpx.Response:
        if answers_a_tool_call(request):
            return httpx.Response(200, json=text_payload("done"))
        return httpx.Response(200, json=call_payload(name, {}))

    return reply


def ping() -> str:
    """Ping."""
    return "pong"


def history_of(runner: Runner, session_id: str) -> list[str]:
    session = runner.get_session(session_id)
    assert session is not None
    return [m.content or "" for m in session.history]


def test_each_run_call_reuses_one_pooled_client_and_closes_it_on_its_own_loop(clients: Clients) -> None:
    clients.reply = call_then_text("ping")
    runner = make_runner(bot(tools=[ping]))

    responses = [runner.run("s1", f"turn {i}") for i in range(3)]

    assert [r.content for r in responses] == ["done"] * 3
    assert [r.tool_calls_executed for r in responses] == [["ping"]] * 3
    assert clients.attempt == []
    assert len(clients.pooled) == 3
    for pooled in clients.pooled:
        assert len(pooled.sent_on) == 2  # the tool call and the final answer share one client
        assert pooled.close_calls == 1
        assert pooled.closed_while_loop_running is True
    assert len({c.created_on for c in clients.pooled}) == 3  # no client is seen by two loops
    assert runner._client_pools == {}


def test_run_returns_what_run_async_returns(clients: Clients) -> None:
    clients.reply = call_then_text("ping")

    via_run = make_runner(bot(tools=[ping])).run("s1", "hi", metadata={"k": "v"})
    via_run_async = asyncio.run(make_runner(bot(tools=[ping])).run_async("s1", "hi", metadata={"k": "v"}))

    assert via_run == via_run_async


def test_tasks_using_the_pooled_client_when_run_returns_are_cancelled_before_it_closes(clients: Clients) -> None:
    background: list[asyncio.Task[Any]] = []
    done_at_close: list[bool] = []
    blocked: list[httpx.Request] = []
    all_blocked = asyncio.Event()
    main_reply = call_then_text("start_background")

    async def reply(request: httpx.Request) -> httpx.Response:
        if b"BG" not in request.content:
            return await main_reply(request)
        blocked.append(request)
        if len(blocked) == 2:
            all_blocked.set()
        await asyncio.sleep(2)  # bounded, so closing first would fail the test, not hang it
        return httpx.Response(200, json=text_payload("late"))

    clients.reply = reply

    async def start_background() -> str:
        """Start background turns."""
        background.append(runner.spawn_turn("spawned", "BG spawned"))
        background.append(asyncio.create_task(runner.run_async("tool-task", "BG task")))
        await asyncio.wait_for(all_blocked.wait(), 2)
        [pooled] = clients.pooled
        pooled.on_close = lambda: done_at_close.extend(t.done() for t in background)
        return "started"

    runner = make_runner(bot(tools=[start_background]))
    assert runner.run("s1", "hi").content == "done"

    assert [t.cancelled() for t in background] == [True, True]
    assert done_at_close == [True, True]  # cancelled and finished before the client closed
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 4 and pooled.close_calls == 1
    assert pooled.closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._spawned_turns == set()


def test_errors_of_tasks_cancelled_by_run_reach_the_loop_exception_handler_once(clients: Clients) -> None:
    reports: list[dict[str, Any]] = []
    started: list[asyncio.Task[None]] = []

    async def fails_when_cancelled() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise ValueError("failed while cancelled") from None

    async def start_stubborn() -> str:
        """Start a task that fails when cancelled."""
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: reports.append(context))
        started.append(asyncio.create_task(fails_when_cancelled()))
        await asyncio.sleep(0)  # let it reach its try block
        return "started"

    clients.reply = call_then_text("start_stubborn")
    runner = make_runner(bot(tools=[start_stubborn]))
    assert runner.run("s1", "hi").content == "done"
    started.clear()
    # Only the report retrieves the task's error, so an unreported one would be logged here,
    # through the same handler, as "Task exception was never retrieved".
    gc.collect()

    assert [(c["message"], type(c["exception"]), c["task"].get_coro().__name__) for c in reports] == [
        ("unhandled exception during asyncio.run() shutdown", ValueError, "fails_when_cancelled")
    ]
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True


async def bad_request(request: httpx.Request) -> httpx.Response:
    return httpx.Response(400, json={"error": "bad"})


async def too_slow(request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(1)
    return httpx.Response(200, json=text_payload("late"))


async def cancel_turn() -> str:
    """Cancel the running turn."""
    current = asyncio.current_task()
    assert current is not None
    current.cancel()
    await asyncio.sleep(0)
    return "unreachable"


@pytest.mark.parametrize(
    ("reply", "error", "sends"),
    [
        (bad_request, ProviderError, 1),
        (too_slow, ProviderError, 1),  # the timed-out attempt is pooled and not retried
        (call_then_text("cancel_turn"), asyncio.CancelledError, 1),
    ],
    ids=["provider-error", "timeout", "cancelled"],
)
def test_a_failing_run_propagates_its_error_rolls_back_and_closes_the_pooled_client(
    clients: Clients, reply: Reply, error: type[BaseException], sends: int
) -> None:
    runner = make_runner(bot(tools=[cancel_turn]), timeout_seconds=0.05, max_retries=1)
    assert runner.run("s1", "before").content == "ok"

    clients.reply = reply
    with pytest.raises(error):
        runner.run("s1", "fails")

    assert history_of(runner, "s1") == ["before", "ok"]
    first, failed = clients.pooled
    assert len(failed.sent_on) == sends
    assert all(c.close_calls == 1 and c.closed_while_loop_running for c in (first, failed))
    assert clients.attempt == []
    assert runner._client_pools == {}


def test_a_cleanup_error_does_not_replace_the_error_a_run_failed_with(
    clients: Clients, caplog: pytest.LogCaptureFixture
) -> None:
    def fail_close() -> None:
        raise OSError("close failed")

    async def reply(request: httpx.Request) -> httpx.Response:
        clients.pooled[0].on_close = fail_close
        return httpx.Response(400, json={"error": "bad"})

    clients.reply = reply
    runner = make_runner()
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.runner"):
        with pytest.raises(ProviderError, match="\\(400\\)"):
            runner.run("s1", "hi")

    assert "Failed to close the runner's connection scope: OSError('close failed')" in caplog.text
    assert clients.pooled[0].close_calls == 1
    assert runner._client_pools == {}


def test_keyboard_interrupt_outside_the_turn_still_closes_the_pooled_client(clients: Clients) -> None:
    """A KeyboardInterrupt raised by a loop callback leaves the turn's task pending, and
    asyncio.run cancels that task while shutting down. Ctrl-C can take this path on 3.10."""

    def interrupt() -> None:
        raise KeyboardInterrupt

    async def interrupted(request: httpx.Request) -> httpx.Response:
        asyncio.get_running_loop().call_soon(interrupt)
        await asyncio.sleep(2)
        return httpx.Response(200, json=text_payload("late"))

    clients.reply = interrupted
    runner = make_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.run("s1", "hi")

    assert history_of(runner, "s1") == []
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True
    assert runner._client_pools == {}


def raise_keyboard_interrupt() -> None:
    raise KeyboardInterrupt


def raise_interrupt(interrupt: type[BaseException]) -> None:
    raise interrupt


@pytest.fixture(params=["task-cancelling", "py310-fallback"])
def teardown_strategy(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs a test with the observed teardown (3.11+) and with the strategy Python 3.10 uses."""
    if request.param == "py310-fallback":
        monkeypatch.setattr(runner_module, "_TEARDOWN_OBSERVABLE", False)
    elif sys.version_info < (3, 11):
        pytest.skip("Task.cancelling() needs Python 3.11+")
    return str(request.param)


def interrupt_outside_the_turn_with_other_tasks_alive(clients: Clients) -> tuple[list[str], list[dict[str, Any]], Runner]:
    """One run() ended by a KeyboardInterrupt raised outside the turn's task while three other
    tasks are alive: one with awaited cleanup, one failing when cancelled, and a turn holding
    a request on the pooled client. Returns the cleanup log, the loop's error reports, and
    the runner."""
    log: list[str] = []
    reports: list[dict[str, Any]] = []
    bg_entered = asyncio.Event()

    async def cleans_up() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            log.append("cleanup done")
            raise

    async def fails_when_cancelled() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            raise ValueError("failed while cancelled") from None

    async def reply(request: httpx.Request) -> httpx.Response:
        if b"BG" in request.content:
            bg_entered.set()
            await asyncio.sleep(2)  # held on the pooled client until cancelled
            return httpx.Response(200, json=text_payload("late"))
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: reports.append(context))
        background.extend(
            [
                loop.create_task(cleans_up()),
                loop.create_task(fails_when_cancelled()),
                loop.create_task(runner.run_async("bg", "BG")),
            ]
        )
        await asyncio.wait_for(bg_entered.wait(), 2)
        loop.call_soon(raise_keyboard_interrupt)
        await asyncio.sleep(2)
        return httpx.Response(200, json=text_payload("late"))

    background: list[asyncio.Task[Any]] = []
    clients.reply = reply
    runner = make_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.run("s1", "hi")
    assert background[2].cancelled()
    return log, reports, runner


@pytest.mark.usefixtures("teardown_strategy")
def test_interrupt_outside_the_turn_leaves_other_tasks_to_a_single_cancellation(clients: Clients) -> None:
    """A KeyboardInterrupt raised outside the turn's task makes asyncio.run cancel every task
    itself. run() must not cancel them again: their cleanup finishes and their errors are
    reported once. Which task the loop steps first varies, so this is repeated."""
    for i in range(20):
        log, reports, runner = interrupt_outside_the_turn_with_other_tasks_alive(clients)

        assert log == ["cleanup done"], f"iteration {i}"
        assert [(c["message"], type(c.get("exception"))) for c in reports] == [
            ("unhandled exception during asyncio.run() shutdown", ValueError)
        ], f"iteration {i}"
        pooled = clients.pooled[-1]
        assert len(pooled.sent_on) == 2 and pooled.close_calls == 1
        assert pooled.closed_while_loop_running is True
        assert runner._client_pools == {}


class ImmediateEngine(BaseEngine):
    """Calls the agent's first tool, then answers, without ever giving up control."""

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        if history[-1].role == Role.USER:
            return EngineTurnResult(tool_calls=[ToolCall(id="c1", name=agent.get_all_tools()[0].name, args={})])
        return EngineTurnResult(text="done")


@pytest.mark.usefixtures("teardown_strategy")
def test_a_task_created_in_the_last_step_of_a_run_starts_before_it_is_cancelled() -> None:
    log: list[str] = []
    background: list[asyncio.Task[None]] = []

    async def work() -> None:
        log.append("started")
        try:
            await asyncio.sleep(60)
        finally:
            log.append("finally ran")

    async def fire_and_forget() -> str:
        """Start background work and return at once."""
        background.append(asyncio.create_task(work()))
        return "started"

    runner = Runner(bot(tools=[fire_and_forget]), session_store=SessionStore(), engine=ImmediateEngine())
    assert runner.run("s1", "hi").content == "done"

    assert log == ["started", "finally ran"]
    assert background[0].cancelled()


@pytest.mark.usefixtures("teardown_strategy")
def test_a_task_that_once_absorbed_a_cancellation_does_not_hang_a_completed_run() -> None:
    """A task that swallowed an earlier cancellation without uncancel() keeps
    Task.cancelling() > 0 on 3.11+. After a turn that ended normally, run() must cancel it
    again, as asyncio.run does, instead of waiting for it forever."""
    log: list[str] = []
    background: list[asyncio.Task[None]] = []
    responses: list[Any] = []

    async def reconnect_loop() -> None:
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                if log:
                    log.append("cancelled again")
                    raise
                log.append("absorbed")  # carries on, without uncancel()

    async def start_and_interrupt_once() -> str:
        """Start a background loop and cancel it once."""
        task = asyncio.create_task(reconnect_loop())
        background.append(task)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        return "started"

    runner = Runner(bot(tools=[start_and_interrupt_once]), session_store=SessionStore(), engine=ImmediateEngine())
    # In a thread, so a hang fails the test instead of blocking the suite.
    thread = threading.Thread(target=lambda: responses.append(runner.run("s1", "hi")), daemon=True)
    thread.start()
    thread.join(5)

    assert not thread.is_alive(), "run() hung waiting for a task it did not cancel"
    assert responses[0].content == "done"
    assert log == ["absorbed", "cancelled again"]
    assert background[0].cancelled()
    assert runner._client_pools == {}


@pytest.mark.usefixtures("teardown_strategy")
@pytest.mark.parametrize("turn_end", ["swallowed", "cancelled", "keyboard-interrupt"])
def test_a_turn_cancelled_by_its_own_code_does_not_hang_on_a_task_that_once_absorbed_a_cancellation(
    turn_end: str,
) -> None:
    """Without a KeyboardInterrupt outside the turn, asyncio.run's teardown is not running, even
    if the turn's own task was cancelled (and maybe swallowed it) or raised KeyboardInterrupt
    itself. run() must then cancel a task with a leftover cancelling() count again, as
    asyncio.run does, instead of waiting for it forever."""
    log: list[str] = []
    background: list[asyncio.Task[None]] = []
    outcome: list[Any] = []

    async def reconnect_loop() -> None:
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                if log:
                    log.append("cancelled again")
                    raise
                log.append("absorbed")  # carries on, without uncancel()

    class SelfCancellingEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            task = asyncio.get_running_loop().create_task(reconnect_loop())
            background.append(task)
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            if turn_end == "keyboard-interrupt":
                raise KeyboardInterrupt
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            if turn_end == "cancelled":
                await asyncio.sleep(2)
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass  # swallowed, without uncancel()
            return EngineTurnResult(text="done")

    def call_run() -> None:
        try:
            outcome.append(runner.run("s1", "hi").content)
        except BaseException as err:
            outcome.append(type(err))

    runner = Runner(bot(), session_store=SessionStore(), engine=SelfCancellingEngine())
    # In a thread, so a hang fails the test instead of blocking the suite.
    thread = threading.Thread(target=call_run, daemon=True)
    thread.start()
    thread.join(5)

    assert not thread.is_alive(), "run() hung waiting for a task it did not cancel"
    expected = {"swallowed": "done", "cancelled": asyncio.CancelledError, "keyboard-interrupt": KeyboardInterrupt}
    assert outcome == [expected[turn_end]]
    assert log == ["absorbed", "cancelled again"]
    assert background[0].cancelled()
    assert runner._client_pools == {}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the teardown is observed on Python 3.11+")
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_a_turn_that_swallows_the_teardown_cancellation_does_not_cancel_other_tasks_again(
    interrupt: type[BaseException],
) -> None:
    """After a KeyboardInterrupt or SystemExit outside the turn, asyncio.run cancels every task.
    If the turn swallows that cancellation and returns normally, run() must still leave the
    other tasks to asyncio.run instead of cancelling them a second time and cutting their
    cleanup short."""
    log: list[str] = []
    background: list[asyncio.Task[None]] = []

    async def cleans_up_slowly() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            log.append("cancelled")
            await asyncio.sleep(0.05)
            log.append("cleanup done")
            raise

    class SwallowingEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            loop = asyncio.get_running_loop()
            background.append(loop.create_task(cleans_up_slowly()))
            await asyncio.sleep(0)
            loop.call_soon(raise_interrupt, interrupt)
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass  # swallowed, without uncancel()
            return EngineTurnResult(text="done")

    runner = Runner(bot(), session_store=SessionStore(), engine=SwallowingEngine())
    with pytest.raises(interrupt):
        runner.run("s1", "hi")

    assert log == ["cancelled", "cleanup done"]
    assert background[0].cancelled()
    assert runner._client_pools == {}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Task.uncancel() needs Python 3.11+")
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_a_task_that_suppresses_the_teardown_cancellation_with_uncancel_is_not_cancelled_again(
    interrupt: type[BaseException],
) -> None:
    """A task may suppress asyncio.run's teardown cancellation the documented way, with
    uncancel(), and finish its work. run() must not cancel it again; its error is reported
    once, by the teardown, as with asyncio.run."""
    log: list[str] = []
    reports: list[dict[str, Any]] = []

    async def finishes_its_work() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            log.append("cancelled")
            current = asyncio.current_task()
            assert current is not None
            current.uncancel()
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            log.append("cancelled again")
            raise
        log.append("finished")
        raise ValueError("failed after its work")

    class InterruptedEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, context: reports.append(context))
            loop.create_task(finishes_its_work())
            await asyncio.sleep(0)
            loop.call_soon(raise_interrupt, interrupt)
            await asyncio.sleep(2)
            return EngineTurnResult(text="late")

    runner = Runner(bot(), session_store=SessionStore(), engine=InterruptedEngine())
    with pytest.raises(interrupt):
        runner.run("s1", "hi")
    gc.collect()

    assert log == ["cancelled", "finished"]
    assert [(c["message"], type(c.get("exception"))) for c in reports] == [
        ("unhandled exception during asyncio.run() shutdown", ValueError)
    ]
    assert runner._client_pools == {}


@pytest.mark.usefixtures("teardown_strategy")
def test_a_task_created_during_the_teardown_is_left_running_as_asyncio_run_leaves_it() -> None:
    """asyncio.run's teardown does not cancel or wait for tasks created after it started; it
    leaves them running when it closes the loop. run() must not wait for one either, since
    one that absorbs a cancellation would block it forever."""
    created: list[asyncio.Task[None]] = []
    log: list[str] = []
    outcome: list[Any] = []

    async def absorbs_a_cancellation() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            log.append("absorbed")  # carries on, without uncancel()
        await asyncio.sleep(60)

    async def starts_more_work_when_cancelled() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            created.append(asyncio.get_running_loop().create_task(absorbs_a_cancellation()))
            await asyncio.sleep(0)
            raise

    class InterruptedEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            loop = asyncio.get_running_loop()
            loop.create_task(starts_more_work_when_cancelled())
            await asyncio.sleep(0)
            loop.call_soon(raise_keyboard_interrupt)
            await asyncio.sleep(2)
            return EngineTurnResult(text="late")

    def call_run() -> None:
        try:
            outcome.append(runner.run("s1", "hi").content)
        except BaseException as err:
            outcome.append(type(err))

    runner = Runner(bot(), session_store=SessionStore(), engine=InterruptedEngine())
    # In a thread, so a hang fails the test instead of blocking the suite.
    thread = threading.Thread(target=call_run, daemon=True)
    thread.start()
    thread.join(5)

    assert not thread.is_alive(), "run() hung waiting for a task created during the teardown"
    assert outcome == [KeyboardInterrupt]
    assert log == []
    [task] = created
    assert not task.done()  # left running when the loop closed, as asyncio.run leaves it
    assert runner._client_pools == {}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the teardown is observed on Python 3.11+")
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_tasks_failing_while_run_waits_for_them_are_reported_once_when_the_loop_is_interrupted(
    interrupt: type[BaseException],
) -> None:
    """A KeyboardInterrupt or SystemExit outside the turn while run() waits for the tasks it
    cancelled starts asyncio.run's teardown, which then reports those tasks itself. Each
    error must be reported once. Which task the loop steps first varies, so this is
    repeated."""
    for i in range(10):
        reports = interrupt_while_run_waits_for_failing_tasks(interrupt)

        assert sorted((c["message"], str(c.get("exception"))) for c in reports) == [
            ("unhandled exception during asyncio.run() shutdown", f"task {n}") for n in range(3)
        ], f"iteration {i}"


@pytest.mark.usefixtures("teardown_strategy")
def test_tasks_still_running_when_runs_wait_is_interrupted_are_reported_once_by_the_teardown() -> None:
    """Tasks that end only after the interruption has stopped run()'s wait are among those
    asyncio.run's teardown cancels and reports, so run() must not report them as well. On
    Python 3.10 run() can't tell, so it leaves every task still running to asyncio.run."""
    reports = interrupt_while_run_waits_for_failing_tasks(KeyboardInterrupt, finish_later=True)

    assert sorted((c["message"], str(c.get("exception"))) for c in reports) == [
        ("unhandled exception during asyncio.run() shutdown", f"task {n}") for n in range(3)
    ]


def interrupt_while_run_waits_for_failing_tasks(
    interrupt: type[BaseException], finish_later: bool = False
) -> list[dict[str, Any]]:
    """One run() whose turn starts three tasks that fail after a slow cleanup, interrupted
    outside the turn while run() waits for them. With ``finish_later``, the tasks fail only
    well after the teardown has cancelled them. Returns the loop's error reports."""
    reports: list[dict[str, Any]] = []

    async def fails_after_cleanup(n: int) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass  # the teardown's cancellation, absorbed
            if finish_later:
                await asyncio.sleep(0.05)
            raise ValueError(f"task {n}") from None

    class StartsTasksEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, context: reports.append(context))
            for n in range(3):
                loop.create_task(fails_after_cleanup(n))
            loop.call_later(0.01, raise_interrupt, interrupt)  # while run() waits
            return EngineTurnResult(text="done")

    runner = Runner(bot(), session_store=SessionStore(), engine=StartsTasksEngine())
    with pytest.raises(interrupt):
        runner.run("s1", "hi")
    gc.collect()
    assert runner._client_pools == {}
    return reports


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="on Python 3.10 tasks still running are left to asyncio.run"
)
def test_errors_of_tasks_run_cancelled_are_reported_once_when_its_wait_is_interrupted() -> None:
    """The first Ctrl-C cancels the turn's task (asyncio.Runner does that on 3.11+), which
    interrupts run()'s wait. A task that failed before must still be reported, and so must
    one that fails while the scope closes, before asyncio.run's teardown starts: the
    teardown reports neither, since both have already ended."""
    reports: list[dict[str, Any]] = []
    turn_task: list[asyncio.Task[Any]] = []
    scope_closing = asyncio.Event()

    async def fails_at_once() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise ValueError("failed at once") from None

    async def interrupts_the_wait() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            turn_task[0].cancel()  # as asyncio.Runner does on the first Ctrl-C
            raise

    async def fails_while_the_scope_closes() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await scope_closing.wait()
            raise ValueError("failed while the scope closed") from None

    class StartsTasksEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, context: reports.append(context))
            current = asyncio.current_task()
            assert current is not None
            turn_task.append(current)
            for work in (fails_at_once, interrupts_the_wait, fails_while_the_scope_closes):
                loop.create_task(work())
            return EngineTurnResult(text="done")

    runner = Runner(bot(), session_store=SessionStore(), engine=StartsTasksEngine())
    close_scope = runner.__aexit__

    async def close_scope_slowly(*exc_info: Any) -> None:
        scope_closing.set()
        for _ in range(3):
            await asyncio.sleep(0)
        await close_scope(*exc_info)

    runner.__aexit__ = close_scope_slowly  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        runner.run("s1", "hi")
    turn_task.clear()
    # An unreported error would be logged here, through the same handler, as "Task exception
    # was never retrieved".
    gc.collect()

    assert sorted((c["message"], str(c.get("exception"))) for c in reports) == [
        ("unhandled exception during asyncio.run() shutdown", "failed at once"),
        ("unhandled exception during asyncio.run() shutdown", "failed while the scope closed"),
    ]
    assert runner._client_pools == {}


@pytest.mark.usefixtures("teardown_strategy")
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_a_task_ending_with_keyboard_interrupt_or_system_exit_is_not_reported_as_a_task_error(
    interrupt: type[BaseException],
) -> None:
    """A task run() cancelled that ends with KeyboardInterrupt or SystemExit stops the loop;
    that exception propagates from run(). It is not an error to report, as asyncio.run
    doesn't report it either."""
    reports: list[dict[str, Any]] = []

    async def stops_when_cancelled() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise interrupt from None

    class StartsTaskEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, context: reports.append(context))
            loop.create_task(stops_when_cancelled())
            await asyncio.sleep(0)
            return EngineTurnResult(text="done")

    runner = Runner(bot(), session_store=SessionStore(), engine=StartsTaskEngine())
    with pytest.raises(interrupt):
        runner.run("s1", "hi")
    gc.collect()

    assert reports == []
    assert runner._client_pools == {}


@pytest.mark.usefixtures("teardown_strategy")
def test_cancelling_the_teardown_wait_stops_it_without_cancelling_the_tasks_again(clients: Clients) -> None:
    """Ctrl-C while run() waits for a task that resists cancellation cancels the turn's task
    (asyncio.Runner does that on 3.11+). The wait stops at once; the task is cancelled again
    only by asyncio.run's own teardown, after the pooled client is closed."""
    cancellations: list[str] = []
    closed: list[bool] = []
    background: list[asyncio.Task[None]] = []
    turn_task: list[asyncio.Task[Any]] = []

    async def resists_cancellation() -> None:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                cancellations.append("after close" if closed else "before close")
                if len(cancellations) == 1:
                    turn_task[0].cancel()  # as asyncio.Runner does on the first Ctrl-C

    async def start_resistant() -> str:
        """Start a task that resists cancellation."""
        current = asyncio.current_task()
        assert current is not None
        turn_task.append(current)
        background.append(asyncio.create_task(resists_cancellation()))
        clients.pooled[0].on_close = lambda: closed.append(True)
        return "started"

    clients.reply = call_then_text("start_resistant")
    runner = make_runner(bot(tools=[start_resistant]))
    with pytest.raises(asyncio.CancelledError):
        runner.run("s1", "hi")

    assert cancellations == ["before close", "after close"]
    assert background[0].done()
    [pooled] = clients.pooled
    assert pooled.close_calls == 1 and pooled.closed_while_loop_running is True


@pytest.mark.parametrize("teardown_strategy", ["task-cancelling", "py310-fallback"], indirect=True)
def test_tasks_left_running_when_the_turn_itself_is_cancelled(clients: Clients, teardown_strategy: str) -> None:
    """With Task.cancelling(), run() cancels the other tasks as usual. On Python 3.10 it
    can't tell whether asyncio.run is already cancelling them, so it leaves them to
    asyncio.run: scope exit waits for the attempt they have in flight, then closes."""
    background: list[asyncio.Task[Any]] = []
    bg_entered = asyncio.Event()

    async def reply(request: httpx.Request) -> httpx.Response:
        if b"BG" in request.content:
            bg_entered.set()
            await asyncio.sleep(0.2)
            return httpx.Response(200, json=text_payload("late"))
        return await call_then_text("cancel_after_background")(request)

    async def cancel_after_background() -> str:
        """Start a background turn, then cancel this turn."""
        background.append(asyncio.create_task(runner.run_async("bg", "BG")))
        await asyncio.wait_for(bg_entered.wait(), 2)
        return await cancel_turn()

    clients.reply = reply
    runner = make_runner(bot(tools=[cancel_after_background]))
    with pytest.raises(asyncio.CancelledError):
        runner.run("s1", "hi")

    [bg] = background
    if teardown_strategy == "py310-fallback":
        assert bg.result().content == "late"  # not cancelled by run(); finished before close
    else:
        assert bg.cancelled()
    [pooled] = clients.pooled
    assert len(pooled.sent_on) == 2 and pooled.close_calls == 1
    assert pooled.closed_while_loop_running is True
    assert runner._client_pools == {}


async def test_run_on_another_thread_inside_a_scope_uses_its_own_pool(clients: Clients) -> None:
    runner = make_runner()
    async with runner:
        await runner.run_async("s1", "outer")
        assert (await asyncio.to_thread(runner.run, "s2", "inner")).content == "ok"
        outer, inner = clients.pooled
        assert inner.created_on is not outer.created_on
        assert inner.close_calls == 1 and inner.closed_while_loop_running is True
        assert not outer.is_closed
        await runner.run_async("s1", "outer again")

    assert (len(outer.sent_on), len(inner.sent_on)) == (2, 1)
    assert outer.close_calls == 1
    assert clients.attempt == []
    assert runner._client_pools == {}


# --- Real connections over 127.0.0.1 -----------------------------------------------------


class LoopbackServer:
    """HTTP/1.1 keep-alive server that counts TCP connections opened and closed.

    A request whose body contains ``SLOW`` is held until ``release_slow`` is set. With
    ``hold_until`` set, every request is held until that many are in flight at once (or 5s).
    A request whose last message contains ``CALL`` gets a ``ping`` tool call back.
    """

    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self.requests = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.hold_until: int | None = None
        self.slow_entered = threading.Event()
        self.release_slow = threading.Event()
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        server = self
        body = json.dumps(text_payload("ok")).encode()
        call_body = json.dumps(call_payload("ping", {})).encode()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                with server._lock:
                    server.opened += 1

            def finish(self) -> None:
                try:
                    super().finish()
                finally:
                    with server._lock:
                        server.closed += 1

            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                request_body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                contents = json.loads(request_body).get("contents") or [{}]
                reply = call_body if "CALL" in json.dumps(contents[-1]) else body
                with server._changed:
                    server.requests += 1
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                    server._changed.notify_all()
                    if server.hold_until is not None:
                        server._changed.wait_for(lambda: server.max_in_flight >= (server.hold_until or 0), timeout=5)
                try:
                    if b"SLOW" in request_body:
                        server.slow_entered.set()
                        server.release_slow.wait(timeout=30)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    self.wfile.write(reply)
                finally:
                    with server._changed:
                        server.in_flight -= 1

            def log_message(self, format: str, *args: Any) -> None:
                pass

        class Server(ThreadingHTTPServer):
            request_queue_size = 256  # accept bursts of >100 concurrent connections

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1beta"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def wait_all_closed(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.closed == self.opened:
                    return True
            time.sleep(0.01)
        return False

    def stop(self) -> None:
        self.release_slow.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[LoopbackServer]:
    server = LoopbackServer()
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.setattr(gemini_module, "_API_BASE_URL", server.base_url)
    yield server
    server.stop()


def resource_warnings(record: list[warnings.WarningMessage]) -> list[str]:
    gc.collect()
    return [str(w.message) for w in record if issubclass(w.category, ResourceWarning)]


def test_loopback_scope_reuses_one_connection_and_closes_it(loopback: LoopbackServer) -> None:
    async def main() -> None:
        runner = make_runner()
        async with runner:
            for i in range(3):
                assert (await runner.run_async("s1", f"turn {i}")).content == "ok"

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        asyncio.run(main())
        assert resource_warnings(record) == []

    assert loopback.opened == 1
    assert loopback.wait_all_closed()


def test_loopback_without_scope_opens_and_closes_a_connection_per_request(loopback: LoopbackServer) -> None:
    # The README pattern: asyncio.run(main()) around run_async, without `async with runner:`.
    async def main() -> None:
        runner = make_runner()
        for i in range(3):
            assert (await runner.run_async("s1", f"turn {i}")).content == "ok"

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        asyncio.run(main())
        assert resource_warnings(record) == []

    assert loopback.opened == 3
    assert loopback.wait_all_closed()


def test_loopback_an_attempt_that_runs_out_of_time_after_sending_is_not_retried(loopback: LoopbackServer) -> None:
    """Over a real connection, httpcore reports that the request was sent before the deadline."""
    # Long enough for a slow runner to open the connection within it. The attempt's deadline
    # and httpx's read timeout are equal, so either may end the attempt; neither is retried.
    runner = make_runner(timeout_seconds=1.0, max_retries=2)
    with pytest.raises(ProviderError, match=r"Gemini API request failed \((TimeoutError|ReadTimeout)\)"):
        runner.run("s1", "SLOW")

    assert loopback.slow_entered.is_set()
    loopback.release_slow.set()
    assert loopback.requests == 1
    assert loopback.wait_all_closed()


def test_loopback_repeated_run_reuses_one_connection_per_call_and_closes_it(loopback: LoopbackServer) -> None:
    runner = make_runner(bot(tools=[ping]))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        responses = [runner.run("s1", f"CALL {i}") for i in range(3)]
        assert resource_warnings(record) == []

    assert [(r.content, r.tool_calls_executed) for r in responses] == [("ok", ["ping"])] * 3
    assert loopback.requests == 6
    assert loopback.opened == 3  # one connection per run() call, shared by its two requests
    assert loopback.wait_all_closed()


@pytest.mark.parametrize(
    ("interruption", "error"),
    [("cancelled", asyncio.CancelledError), ("keyboard-interrupt", KeyboardInterrupt)],
)
def test_loopback_interrupted_run_closes_every_connection(
    loopback: LoopbackServer, interruption: str, error: type[BaseException]
) -> None:
    """The turn is interrupted while its request is in flight and other keep-alive
    connections are idle; every connection must actually be closed, with no warnings."""

    async def ping() -> str:
        """Open keep-alive connections, then hold the turn's next request in flight."""
        await asyncio.gather(*(runner.run_async(f"warm-{i}", "hi") for i in range(3)))
        loop = asyncio.get_running_loop()
        turn_task = asyncio.current_task()
        assert turn_task is not None
        loop.call_later(0.2, turn_task.cancel if interruption == "cancelled" else raise_keyboard_interrupt)
        return "SLOW"  # in the next request's body, so the server holds that request

    runner = make_runner(bot(tools=[ping]), max_retries=0, timeout_seconds=10)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(error):
            runner.run("s1", "CALL")
        assert resource_warnings(record) == []

    assert loopback.slow_entered.is_set()
    assert loopback.opened == 3  # the turn's connection plus two more for the concurrent warm-up
    loopback.release_slow.set()
    assert loopback.wait_all_closed()
    assert history_of(runner, "s1") == []


SIGINT_DURING_TEARDOWN = """
import asyncio, json, signal, time
import httpx
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import GeminiEngine
from chatflow_agent.core.engines import gemini as gemini_module
from chatflow_agent.core.memory import SessionStore
from chatflow_agent.core.runner import Runner

events = []

def reply(request):
    if b"functionResponse" in request.content:
        part = {"text": "done"}
    else:
        part = {"functionCall": {"name": "start_resistant", "args": {}}}
    return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [part]}}]})

class Pooled(httpx.AsyncClient):
    async def aclose(self):
        events.append("pool closed")
        await super().aclose()

gemini_module._new_pooled_client = lambda: Pooled(transport=httpx.MockTransport(reply))

async def resists_cancellation():
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            events.append("cancelled")
            signal.raise_signal(signal.SIGINT)  # Ctrl-C, once per cancellation received

background = []

async def start_resistant() -> str:
    \"\"\"Start a task that resists cancellation.\"\"\"
    background.append(asyncio.create_task(resists_cancellation()))
    return "started"

agent = Agent(name="Bot", instructions="x", model="gemini-2.5-flash", tools=[start_resistant])
runner = Runner(agent, session_store=SessionStore(), engine=GeminiEngine(api_key="k"))
started = time.monotonic()
try:
    runner.run("s1", "hi")
    outcome = "returned"
except KeyboardInterrupt:
    outcome = "KeyboardInterrupt"
print(json.dumps({"outcome": outcome, "events": events, "seconds": time.monotonic() - started}))
"""


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.Runner turns Ctrl-C into a cancellation on 3.11+")
@pytest.mark.skipif(sys.platform == "win32", reason="relies on POSIX SIGINT delivery")
def test_ctrl_c_during_teardown_closes_the_client_and_a_second_ctrl_c_exits(tmp_path: Any) -> None:
    """Real SIGINTs, in a subprocess so a stray KeyboardInterrupt can't end the test session.

    A background task resists cancellation for 5 s and sends a Ctrl-C each time it is
    cancelled. The first Ctrl-C (during run()'s teardown wait) must stop that wait without
    cancelling the task again, so the pooled client closes next. The second Ctrl-C (during
    asyncio.run's own teardown) must end the call well before the 5 s are up."""
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_GEMINI_BASE_URL"}
    result = subprocess.run(
        [sys.executable, "-c", SIGINT_DURING_TEARDOWN],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])

    assert outcome["outcome"] == "KeyboardInterrupt"
    assert outcome["events"] == ["cancelled", "pool closed", "cancelled"]
    assert outcome["seconds"] < 3


@pytest.mark.parametrize("yields_between_cancels", [1, 2, 3, 5])
def test_repeated_cancellation_of_scope_exit_still_closes_every_connection(
    loopback: LoopbackServer, yields_between_cancels: int
) -> None:
    """Cancel the exit while it waits for an in-flight request, then again while the client
    is closing: every keep-alive connection must still be closed, with nothing left running."""

    async def main() -> None:
        runner = make_runner(max_retries=0)
        await runner.__aenter__()
        await asyncio.gather(*(runner.run_async(f"s{i}", "hi") for i in range(5)))  # idle keep-alive connections
        slow = asyncio.create_task(runner.run_async("slow", "SLOW"))
        while not loopback.slow_entered.is_set():
            await asyncio.sleep(0.005)

        exiting = asyncio.create_task(runner.__aexit__(None, None, None))
        await asyncio.sleep(0.05)
        assert not exiting.done()  # waiting for the slow request
        exiting.cancel()
        for _ in range(yields_between_cancels):
            await asyncio.sleep(0)
        exiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await exiting

        with pytest.raises(ProviderError):
            await slow  # its connection was closed under it
        assert runner._client_pools == {}
        assert {t for t in asyncio.all_tasks() if t is not asyncio.current_task()} == set()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        asyncio.run(main())
        assert resource_warnings(record) == []

    assert loopback.opened >= 2  # several keep-alive connections were open at exit
    loopback.release_slow.set()
    assert loopback.wait_all_closed()


def test_more_than_100_concurrent_requests_succeed_in_a_scope(loopback: LoopbackServer) -> None:
    concurrent = 120
    loopback.hold_until = concurrent  # replies only once all requests are in flight together

    async def main() -> list[str]:
        runner = make_runner(timeout_seconds=15, max_retries=0)
        async with runner:
            responses = await asyncio.gather(*(runner.run_async(f"s{i}", "hi") for i in range(concurrent)))
        return [r.content for r in responses]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        assert asyncio.run(main()) == ["ok"] * concurrent
        assert resource_warnings(record) == []

    assert loopback.max_in_flight == concurrent
    assert loopback.wait_all_closed()


@pytest.mark.skipif(sys.platform == "win32", reason="unclosed-socket warnings depend on the event loop implementation")
def test_warning_check_detects_an_abandoned_client(loopback: LoopbackServer) -> None:
    """Negative control: the warning check above does catch a client left open."""

    async def leak() -> None:
        client = httpx.AsyncClient()
        await client.post(f"{loopback.base_url}/models/x:generateContent", content=b"{}")
        # Not closed: its connection outlives the event loop.

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        asyncio.run(leak())
        assert resource_warnings(record) != []
