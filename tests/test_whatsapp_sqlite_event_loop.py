"""WhatsApp webhook with a file-backed SQLiteSessionStore under SQLite contention (F-c).

The deduplication write still happens before the HTTP acknowledgement (at-most-once
semantics unchanged), but it runs in a worker thread: while it waits for the database the
event loop keeps serving other requests and sessions. Report 38 tests 2, 3, 11 and 12.
"""

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Message, Role

SAFETY = 30.0
# A dedup write blocking the loop stalls it for its whole wait on the lock: busy_timeout
# (10 s) or more. 1.0 s keeps a 10x margin over that and stays clear of scheduler jitter
# on loaded runners; dedup_threads() checks deterministically where the write runs.
MAX_STALL = 1.0
PHONE = "5491100000077"


class Engine(BaseEngine):
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[str] = []

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        text = next(m.content for m in reversed(history) if m.role == Role.USER) or ""
        self.calls.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return EngineTurnResult(text=f"re:{text}")


class ExternalWriter:
    def __init__(self, db: Path, hold: float | None = None) -> None:
        self.db = db
        self.hold = hold
        self.release = threading.Event()
        self._taken = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        conn = sqlite3.connect(self.db, check_same_thread=False)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO chatflow_processed_messages (message_id) VALUES ('external-writer')")
        self._taken.set()
        self.release.wait(self.hold if self.hold is not None else 60)
        conn.rollback()
        conn.close()

    def __enter__(self) -> "ExternalWriter":
        self._thread.start()
        assert self._taken.wait(5)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release.set()
        self._thread.join(SAFETY)


class Heartbeat:
    def __init__(self) -> None:
        self.gaps: list[float] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            self.gaps.append(now - last)
            last = now

    async def __aenter__(self) -> "Heartbeat":
        self._task = asyncio.ensure_future(self._run())
        await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._stop.set()
        assert self._task is not None
        await self._task

    @property
    def max_gap(self) -> float:
        return max(self.gaps) if self.gaps else 0.0


def dedup_threads(store: SQLiteSessionStore) -> list[str]:
    """Record the name of the thread each ``record_processed_message`` call runs on."""
    names: list[str] = []
    real = store.record_processed_message

    def spy(message_id: str) -> bool:
        names.append(threading.current_thread().name)
        return real(message_id)

    store.record_processed_message = spy  # type: ignore[method-assign]
    return names


def on_sqlite_workers(names: list[str]) -> bool:
    """Whether every call ran on the store's worker threads, never on the loop's thread."""
    loop_thread = threading.current_thread().name
    return bool(names) and all(
        n.startswith("chatflow-sqlite") and n != loop_thread for n in names
    )


def channel(store: Any, engine: BaseEngine | None = None) -> WhatsAppChannel:
    ch = WhatsAppChannel(verify_token="tok", verify_signature=False)
    ch.attach(Runner(Agent(name="Bot", instructions="Test"), session_store=store, engine=engine or Engine()))
    return ch


def meta(wamid: str, text: str = "hola") -> dict[str, Any]:
    return {"object": "whatsapp_business_account", "entry": [{"changes": [{"value": {
        "messages": [{"id": wamid, "from": PHONE, "type": "text", "text": {"body": text}}]}}]}]}


async def request(app: Any, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Drive the ASGI app directly. Returns status, body, the time the response was
    complete (what Meta sees) and the app's own exception, if any."""
    body = json.dumps(payload).encode() if payload is not None else b""
    out: dict[str, Any] = {"status": None, "body": b"", "ack": None, "exc": None}
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")
            if not message.get("more_body"):
                out["ack"] = time.perf_counter()

    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 1), "server": ("testserver", 80),
        "headers": headers,
    }
    try:
        await app(scope, receive, send)
    except Exception as err:  # ServerErrorMiddleware re-raises after sending its 500
        out["exc"] = err
    return out


def queued(res: dict[str, Any]) -> int:
    return int(json.loads(res["body"])["messages_queued"])


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "wa.db"
    SQLiteSessionStore(path).close()  # schema
    return path


@pytest.fixture
def store(db: Path) -> Iterator[SQLiteSessionStore]:
    with SQLiteSessionStore(db) as s:  # closed at teardown: shuts down its worker threads
        yield s


@pytest.mark.asyncio
async def test_2_ack_waits_for_its_dedup_write_but_the_loop_keeps_serving(
    db: Path, store: SQLiteSessionStore
) -> None:
    threads = dedup_threads(store)
    ch = channel(store)
    with ExternalWriter(db) as writer:
        async with Heartbeat() as hb:
            post = asyncio.ensure_future(request(ch.app, "POST", "/webhook", meta("wamid.2")))
            await asyncio.sleep(0.2)
            assert not post.done(), "the ACK waits for the deduplication write"
            health = await asyncio.wait_for(request(ch.app, "GET", "/health"), 1.0)
            assert health["status"] == 200
            t_release = time.perf_counter()
            writer.release.set()
            res = await asyncio.wait_for(post, SAFETY)
    assert res["status"] == 200 and queued(res) == 1
    assert res["ack"] >= t_release
    assert len(threads) == 1 and on_sqlite_workers(threads), threads
    assert hb.max_gap < MAX_STALL
    assert store.is_message_processed("wamid.2")


@pytest.mark.asyncio
async def test_12_concurrent_deliveries_of_one_wamid_are_queued_once(db: Path) -> None:
    store = SQLiteSessionStore(db)
    engine = Engine()
    ch = channel(store, engine)
    for round_ in range(10):
        wamid = f"wamid.dup{round_}"
        with ExternalWriter(db) as writer:
            posts = [asyncio.ensure_future(request(ch.app, "POST", "/webhook", meta(wamid, f"t{round_}")))
                     for _ in range(3)]
            await asyncio.sleep(0.05)  # all three wait on the deduplication write
            writer.release.set()
            results = await asyncio.wait_for(asyncio.gather(*posts), SAFETY)
        assert sorted(queued(r) for r in results) == [0, 0, 1]
    assert engine.calls == [f"t{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_3_11_lock_beyond_busy_timeout_fails_the_webhook_not_the_loop_and_retry_works(
    db: Path, store: SQLiteSessionStore
) -> None:
    """Lock held well beyond busy_timeout (10 s; more on Windows): that webhook ends in
    HTTP 500 and records nothing, so Meta's retry is processed once; meanwhile the loop
    and an unrelated runner keep working."""
    threads = dedup_threads(store)
    engine = Engine()
    ch = channel(store, engine)
    other_runner = Runner(Agent(name="Bot", instructions="x"), session_store=SessionStore(), engine=Engine())
    with ExternalWriter(db, hold=14.0):
        async with Heartbeat() as hb:
            t0 = time.perf_counter()
            post = asyncio.ensure_future(request(ch.app, "POST", "/webhook", meta("wamid.3")))
            await asyncio.sleep(0.5)
            other = await asyncio.wait_for(other_runner.run_async("other", "ping"), 1.0)
            res = await asyncio.wait_for(post, SAFETY)
            elapsed = time.perf_counter() - t0
    assert other.content == "re:ping"
    assert res["status"] == 500
    assert isinstance(res["exc"], sqlite3.OperationalError)
    assert elapsed >= 9.0
    assert len(threads) == 1 and on_sqlite_workers(threads), threads
    assert hb.max_gap < MAX_STALL
    assert not store.is_message_processed("wamid.3")
    # 11: Meta's retry is processed once; a further duplicate is skipped.
    retry = await request(ch.app, "POST", "/webhook", meta("wamid.3"))
    assert retry["status"] == 200 and queued(retry) == 1
    again = await request(ch.app, "POST", "/webhook", meta("wamid.3"))
    assert queued(again) == 0
    assert engine.calls == ["hola"]
    assert len(threads) == 3 and on_sqlite_workers(threads), threads
