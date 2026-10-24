"""Lifecycle tests for the Gemini HTTP client scope that ``WhatsAppChannel``'s app lifespan opens.

The app is driven by Starlette's ``TestClient`` (only its ``with`` form runs the lifespan), by
raw ASGI lifespan messages for cancellation, and by a real uvicorn server on 127.0.0.1 for its
shutdown order. Gemini clients are the recording clients of ``test_gemini_lifecycle``. No
external network is used.
"""

import asyncio
import contextlib
import logging
import socket
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from starlette.testclient import TestClient  # noqa: E402

from chatflow_agent.channels.whatsapp import WhatsAppChannel  # noqa: E402
from chatflow_agent.core.engines import gemini as gemini_module  # noqa: E402
from chatflow_agent.core.runner import Runner  # noqa: E402
from tests.test_gemini_lifecycle import (  # noqa: E402
    Clients,
    Gate,
    LoopbackServer,
    history_of,
    make_runner,
    resource_warnings,
    text_payload,
)

PHONE = "5491100001111"


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> Clients:
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    return Clients(monkeypatch)


def make_channel(runner: Runner | None = None) -> WhatsAppChannel:
    channel = WhatsAppChannel(verify_token="tok", verify_signature=False)
    if runner is not None:
        channel.attach(runner)
    return channel


def meta_payload(wamid: str, text: str = "hi") -> dict[str, Any]:
    message = {"id": wamid, "from": PHONE, "type": "text", "text": {"body": text}}
    return {"object": "whatsapp_business_account", "entry": [{"changes": [{"value": {"messages": [message]}}]}]}


def post_message(client: TestClient | httpx.Client, wamid: str, text: str = "hi") -> None:
    response = client.post("/webhook", json=meta_payload(wamid, text))
    assert response.status_code == 200
    assert response.json()["messages_queued"] == 1


def recording_close(events: list[str]) -> Callable[[], None]:
    return lambda: events.append("pooled client closed")


# --- Startup and normal shutdown (TestClient) ----------------------------------------------


def test_webhook_turns_share_one_pooled_client_that_closes_when_the_lifespan_ends(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)  # attached after the app was built, as users do

    with TestClient(channel.app) as client:
        for i in range(3):
            post_message(client, f"wamid.{i}")
        assert len(clients.pooled) == 1
        assert clients.pooled[0].close_calls == 0

    pooled = clients.pooled[0]
    assert clients.attempt == []
    assert pooled.sent_on == [pooled.created_on] * 3
    assert pooled.close_calls == 1
    assert pooled.closed_while_loop_running is True
    assert runner._client_pools == {}
    assert history_of(runner, PHONE) == ["hi", "ok"] * 3


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_without_an_attached_runner_the_lifespan_only_runs_the_apps_own(clients: Clients) -> None:
    channel = make_channel()
    events: list[str] = []
    channel.app.on_event("startup")(lambda: events.append("startup handler"))
    channel.app.on_event("shutdown")(lambda: events.append("shutdown handler"))

    with TestClient(channel.app) as client:
        assert client.get("/health").status_code == 200

    assert events == ["startup handler", "shutdown handler"]
    assert clients.created == []


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_startup_and_shutdown_handlers_run_inside_the_scope(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)
    events: list[str] = []

    @channel.app.on_event("startup")
    async def startup() -> None:
        await runner.run_async("startup", "warm up")
        events.append("startup handler")

    @channel.app.on_event("shutdown")
    async def shutdown() -> None:
        await runner.run_async("shutdown", "bye")
        events.append("shutdown handler")

    with TestClient(channel.app) as client:
        clients.pooled[0].on_close = recording_close(events)
        post_message(client, "wamid.1")

    assert events == ["startup handler", "shutdown handler", "pooled client closed"]
    assert len(clients.pooled) == 1
    assert len(clients.pooled[0].sent_on) == 3
    assert clients.attempt == []
    assert runner._client_pools == {}


def test_without_the_lifespan_each_attempt_uses_and_closes_its_own_client(clients: Clients) -> None:
    # The bare TestClient form (as the existing WhatsApp tests use it) runs no lifespan.
    runner = make_runner()
    client = TestClient(make_channel(runner).app)

    for i in range(2):
        post_message(client, f"wamid.{i}")

    assert clients.pooled == []
    assert len(clients.attempt) == 2
    assert all(c.close_calls == 1 for c in clients.attempt)
    assert runner._client_pools == {}


def test_each_lifespan_opens_and_closes_its_own_pool(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)

    for i in range(2):
        with TestClient(channel.app) as client:
            post_message(client, f"wamid.{i}")

    first, second = clients.pooled
    assert first.created_on is not second.created_on
    assert first.close_calls == second.close_calls == 1
    assert clients.attempt == []
    assert runner._client_pools == {}


def test_the_runner_entered_at_startup_is_the_one_closed(clients: Clients) -> None:
    runner = make_runner()
    replacement = make_runner()
    channel = make_channel(runner)

    with TestClient(channel.app) as client:
        post_message(client, "wamid.1")
        channel.attach(replacement)

    assert clients.pooled[0].close_calls == 1
    assert runner._client_pools == {}
    assert replacement._client_pools == {}


def test_an_included_routers_lifespan_runs_inside_the_scope_and_its_state_reaches_requests(clients: Clients) -> None:
    from contextlib import asynccontextmanager

    from fastapi import APIRouter, Request

    runner = make_runner()
    channel = make_channel(runner)
    events: list[str] = []

    @asynccontextmanager
    async def router_lifespan(app: Any) -> Any:
        await runner.run_async("router", "warm up")
        yield {"marker": "from the router"}
        events.append("router shutdown")

    router = APIRouter(lifespan=router_lifespan)

    @router.get("/marker")
    async def marker(request: Request) -> dict[str, str]:
        return {"marker": request.state.marker}

    channel.app.include_router(router)

    with TestClient(channel.app) as client:
        clients.pooled[0].on_close = recording_close(events)
        assert client.get("/marker").json() == {"marker": "from the router"}
        post_message(client, "wamid.1")

    assert events == ["router shutdown", "pooled client closed"]
    assert len(clients.pooled) == 1
    assert len(clients.pooled[0].sent_on) == 2
    assert clients.attempt == []
    assert runner._client_pools == {}


def test_an_injected_client_is_used_and_never_closed(clients: Clients) -> None:
    async def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=text_payload("injected"))

    async def check_open(injected: httpx.AsyncClient) -> None:
        assert not injected.is_closed
        await injected.aclose()

    injected = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    runner = make_runner(http_client=injected)

    with TestClient(make_channel(runner).app) as client:
        for i in range(2):
            post_message(client, f"wamid.{i}")

    assert history_of(runner, PHONE) == ["hi", "injected"] * 2
    assert clients.created == []
    assert runner._client_pools == {}
    asyncio.run(check_open(injected))


# --- Startup and shutdown failures ---------------------------------------------------------


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_a_failing_startup_handler_still_closes_the_scope(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)

    @channel.app.on_event("startup")
    async def startup() -> None:
        await runner.run_async("startup", "warm up")
        raise ValueError("startup failed")

    with pytest.raises(ValueError, match="startup failed"):
        with TestClient(channel.app):
            pass

    assert clients.pooled[0].close_calls == 1
    assert clients.pooled[0].closed_while_loop_running is True
    assert runner._client_pools == {}


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_a_failing_shutdown_handler_still_closes_the_scope(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)

    @channel.app.on_event("shutdown")
    async def shutdown() -> None:
        raise ValueError("shutdown failed")

    with pytest.raises(ValueError, match="shutdown failed"):
        with TestClient(channel.app) as client:
            post_message(client, "wamid.1")

    assert clients.pooled[0].close_calls == 1
    assert runner._client_pools == {}


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_a_cleanup_error_does_not_replace_the_error_startup_failed_with(
    clients: Clients, caplog: pytest.LogCaptureFixture
) -> None:
    runner = make_runner()
    channel = make_channel(runner)

    @channel.app.on_event("startup")
    async def startup() -> None:
        await runner.run_async("startup", "warm up")

        def broken_close() -> None:
            raise OSError("close failed")

        clients.pooled[0].on_close = broken_close
        raise ValueError("startup failed")

    with pytest.raises(ValueError, match="startup failed"):
        with TestClient(channel.app):
            pass

    assert "Failed to close the runner's connection scope" in caplog.text
    assert "close failed" in caplog.text
    assert clients.pooled[0].close_calls == 1
    assert runner._client_pools == {}


# --- Cancellation (raw ASGI lifespan) ------------------------------------------------------


LIFESPAN_SCOPE = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}


async def until_sent(sent: list[str], message_type: str) -> None:
    async def wait() -> None:
        while message_type not in sent:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=5)  # fail, rather than hang, if it never comes


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_a_lifespan_cancelled_during_startup_closes_the_pooled_client(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)
    gate = Gate()

    @channel.app.on_event("startup")
    async def startup() -> None:
        await runner.run_async("startup", "warm up")
        await gate.release.wait()  # never released: startup is still running when cancelled

    async def main() -> None:
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        sent: list[str] = []

        async def send(message: dict[str, Any]) -> None:
            sent.append(message["type"])

        await messages.put({"type": "lifespan.startup"})
        lifespan = asyncio.create_task(channel.app(dict(LIFESPAN_SCOPE, state={}), messages.get, send))
        while not clients.pooled or not clients.pooled[0].sent_on:
            await asyncio.sleep(0)
        lifespan.cancel()
        with pytest.raises(asyncio.CancelledError):
            await lifespan
        assert sent == ["lifespan.startup.failed"]

    asyncio.run(asyncio.wait_for(main(), timeout=10))

    assert clients.pooled[0].close_calls == 1
    assert clients.pooled[0].closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}


def test_a_cancelled_lifespan_closes_the_pooled_client_while_the_loop_runs(clients: Clients) -> None:
    runner = make_runner()
    channel = make_channel(runner)

    async def main() -> None:
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        sent: list[str] = []

        async def send(message: dict[str, Any]) -> None:
            sent.append(message["type"])

        await messages.put({"type": "lifespan.startup"})
        lifespan = asyncio.create_task(channel.app(dict(LIFESPAN_SCOPE, state={}), messages.get, send))
        await until_sent(sent, "lifespan.startup.complete")
        await runner.run_async("s1", "hi")
        assert clients.pooled[0].close_calls == 0

        lifespan.cancel()
        with pytest.raises(asyncio.CancelledError):
            await lifespan
        assert sent == ["lifespan.startup.complete", "lifespan.shutdown.failed"]

    asyncio.run(main())

    assert clients.pooled[0].close_calls == 1
    assert clients.pooled[0].closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}


# --- A real uvicorn server -----------------------------------------------------------------


class UvicornServer:
    """Serves ``app`` with uvicorn on 127.0.0.1 in a thread (so no signal handlers), via ``asyncio.run``."""

    def __init__(self, app: Any, log_level: str = "critical") -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        # lifespan left at uvicorn's default ("auto"), as WhatsAppChannel.run() uses it.
        self.server = uvicorn.Server(uvicorn.Config(app, log_level=log_level))
        self.loop: asyncio.AbstractEventLoop | None = None
        self.error: BaseException | None = None
        self.on_teardown: list[Callable[[], Any]] = []  # run on the server's loop by the fixture
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        async def serve() -> None:
            self.loop = asyncio.get_running_loop()
            await self.server.serve(sockets=[self.sock])

        try:
            asyncio.run(serve())
        except BaseException as err:  # reported by the test
            self.error = err

    def start(self) -> None:
        self.thread.start()
        assert wait_until(lambda: self.server.started)

    def call_soon(self, callback: Callable[[], Any]) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(callback)

    def join(self) -> None:
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
        assert self.error is None


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def uvicorn_server() -> Iterator[Callable[[Any], UvicornServer]]:
    servers: list[UvicornServer] = []

    def start(app: Any, **kwargs: Any) -> UvicornServer:
        server = UvicornServer(app, **kwargs)
        servers.append(server)
        server.start()
        return server

    yield start
    for server in servers:  # never leave a server thread behind a failed assertion
        server.server.should_exit = server.server.force_exit = True
        if server.thread.is_alive():
            for callback in server.on_teardown:
                with contextlib.suppress(RuntimeError):  # the loop may close meanwhile
                    server.call_soon(callback)
            server.thread.join(timeout=10)
        server.sock.close()  # uvicorn closes it at shutdown; this covers a failed start
        assert not server.thread.is_alive()


class DrainStarted(logging.Handler):
    """Set once uvicorn logs that its graceful shutdown is waiting for request tasks to end."""

    def __init__(self) -> None:
        super().__init__()
        self.event = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Waiting for background tasks to complete"):
            self.event.set()


@pytest.fixture
def uvicorn_drain_started() -> Iterator[DrainStarted]:
    """Attach it to ``uvicorn.error`` after creating the server: uvicorn.Config's logging
    setup replaces that logger's handlers."""
    names = ["uvicorn.error", "uvicorn.access", "uvicorn.asgi"]
    levels = {name: logging.getLogger(name).level for name in names}
    handler = DrainStarted()
    yield handler
    logging.getLogger("uvicorn.error").removeHandler(handler)
    for name, level in levels.items():  # uvicorn.Config sets them from log_level
        logging.getLogger(name).setLevel(level)


@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_uvicorn_finishes_a_turn_in_flight_before_the_lifespan_shutdown_starts(
    clients: Clients,
    uvicorn_server: Callable[..., UvicornServer],
    uvicorn_drain_started: DrainStarted,
) -> None:
    events: list[str] = []
    gate = Gate()

    async def held(request: httpx.Request) -> httpx.Response:
        events.append("request in flight")
        response = await gate(request)
        events.append("released")
        return response

    clients.reply = held
    runner = make_runner()
    channel = make_channel(runner)
    # The pool itself waits for attempts in flight before closing, so the shutdown handler
    # (which runs when uvicorn starts the lifespan shutdown) is what shows uvicorn's order.
    channel.app.on_event("shutdown")(lambda: events.append("lifespan shutdown"))
    server = uvicorn_server(channel.app, log_level="info")  # for the drain log record
    server.on_teardown.append(gate.release.set)  # after a failed assertion, let the turn end
    logging.getLogger("uvicorn.error").addHandler(uvicorn_drain_started)
    with httpx.Client(base_url=server.url) as http:
        post_message(http, "wamid.1")  # acknowledged at once; the turn runs as a background task
    assert wait_until(gate.entered.is_set)
    clients.pooled[0].on_close = recording_close(events)

    server.server.should_exit = True
    # Logged once uvicorn's shutdown waits for the turn's task; it keeps waiting until the task
    # ends, so the lifespan shutdown can't have started yet.
    assert uvicorn_drain_started.event.wait(5)
    assert events == ["request in flight"]
    server.call_soon(gate.release.set)
    server.join()

    assert events == ["request in flight", "released", "lifespan shutdown", "pooled client closed"]
    assert history_of(runner, PHONE) == ["hi", "released"]
    assert clients.pooled[0].closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}


def test_uvicorn_force_exit_skips_lifespan_shutdown_but_the_client_is_still_closed(
    clients: Clients, uvicorn_server: Callable[[Any], UvicornServer]
) -> None:
    # A second Ctrl-C: uvicorn skips the lifespan shutdown, and asyncio.run then cancels the
    # lifespan task and the turn in flight.
    gate = Gate()
    clients.reply = gate
    runner = make_runner()
    server = uvicorn_server(make_channel(runner).app)
    server.on_teardown.append(gate.release.set)
    with httpx.Client(base_url=server.url) as http:
        post_message(http, "wamid.1")
    assert wait_until(gate.entered.is_set)

    server.server.force_exit = server.server.should_exit = True
    server.join()

    assert not gate.release.is_set()
    assert clients.pooled[0].close_calls == 1
    assert clients.pooled[0].closed_while_loop_running is True
    assert clients.attempt == []
    assert runner._client_pools == {}
    session = runner.get_session(PHONE)
    assert session is None or session.history == []  # the cancelled turn was rolled back


# --- Real connections over 127.0.0.1 -------------------------------------------------------


@pytest.fixture
def loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[LoopbackServer]:
    server = LoopbackServer()
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.setattr(gemini_module, "_API_BASE_URL", server.base_url)
    yield server
    server.stop()


def test_loopback_webhook_turns_reuse_one_connection_closed_at_shutdown(loopback: LoopbackServer) -> None:
    runner = make_runner()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        with TestClient(make_channel(runner).app) as client:
            for i in range(3):
                post_message(client, f"wamid.{i}")
        assert resource_warnings(record) == []

    assert loopback.requests == 3
    assert loopback.opened == 1
    assert loopback.wait_all_closed()
    assert runner._client_pools == {}
