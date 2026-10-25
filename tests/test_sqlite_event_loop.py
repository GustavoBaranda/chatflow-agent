"""SQLite calls of a file-backed SQLiteSessionStore no longer block the event loop (F-c).

Covers the Runner integration (report 38 tests 1, 4-10, 14-16) and the rollback-failure
condition. Synchronization is deterministic where it can be: an external writer holds the
SQLite write lock until the test releases it (threading.Event), and slow writes are
simulated inside the worker thread with events. Timing assertions only bound the event
loop stall and deadline delivery, with tolerances for slow CI runners.
"""

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core import memory
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import (
    SessionContext,
    SessionStore,
    SQLiteSessionContext,
    SQLiteSessionStore,
)
from chatflow_agent.core.runner import (
    Runner,
    _dispatch_limits,
    _DispatchLimits,
    _SessionWaitTimeout,
    _TurnTimeout,
)
from chatflow_agent.types import Message, Role

SAFETY = 15.0
MAX_STALL = 0.1  # longest acceptable event-loop stall (Windows timer granularity is ~16 ms)


async def bounded(aw: Any, timeout: float = SAFETY) -> Any:
    return await asyncio.wait_for(aw, timeout)


async def wait_event(ev: threading.Event) -> None:
    deadline = time.monotonic() + SAFETY
    while not ev.is_set():
        assert time.monotonic() < deadline, "event never set"
        await asyncio.sleep(0.005)


class ExternalWriter:
    """Another connection holding the SQLite write lock until released."""

    def __init__(self, db: Path) -> None:
        self.db = db
        self.release = threading.Event()
        self._taken = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        conn = sqlite3.connect(self.db, check_same_thread=False)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO chatflow_processed_messages (message_id) VALUES ('external-writer')")
        self._taken.set()
        self.release.wait(60)
        conn.rollback()
        conn.close()

    def __enter__(self) -> "ExternalWriter":
        self._thread.start()
        assert self._taken.wait(5)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release.set()
        self._thread.join(10)


class Heartbeat:
    """Longest gap between event-loop iterations while active."""

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


class Engine(BaseEngine):
    """Replies ``re:<text>``; texts in ``gates`` wait for their asyncio.Event."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.gates: dict[str, asyncio.Event] = {}
        self.entered: dict[str, asyncio.Event] = {}
        self.cancelled_at: dict[str, float] = {}

    def entered_event(self, text: str) -> asyncio.Event:
        return self.entered.setdefault(text, asyncio.Event())

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        text = next(m.content for m in reversed(history) if m.role == Role.USER) or ""
        self.entered_event(text).set()
        try:
            gate = self.gates.get(text)
            if gate is not None:
                await gate.wait()
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled_at[text] = time.perf_counter()
            raise
        return EngineTurnResult(text=f"re:{text}")


def runner_for(store: Any, engine: BaseEngine | None = None) -> tuple[Runner, Any]:
    engine = engine or Engine()
    return Runner(Agent(name="Bot", instructions="Test"), session_store=store, engine=engine), engine


def rows(db: Path, session_id: str) -> list[str | None]:
    conn = sqlite3.connect(db)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT content FROM chatflow_messages WHERE session_id = ? ORDER BY id", (session_id,)
            )
        ]
    finally:
        conn.close()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "fc.db"


@pytest.fixture
def thread_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Records (class, thread name) of every add_message call on either context class."""
    seen: list[tuple[str, str]] = []
    for cls in (SessionContext, SQLiteSessionContext):
        original = cls.__dict__["add_message"]

        def spy(self: Any, message: Message, _orig: Any = original, _cls: type = cls) -> None:
            seen.append((_cls.__name__, threading.current_thread().name))
            _orig(self, message)

        monkeypatch.setattr(cls, "add_message", spy)
    return seen


# ---------------------------------------------------------------- 1, 14, 15: where calls run


@pytest.mark.asyncio
async def test_1_no_contention_turn_runs_store_calls_in_worker_threads(db: Path, thread_spy: Any) -> None:
    runner, _ = runner_for(SQLiteSessionStore(db))
    async with Heartbeat() as hb:
        response = await bounded(runner.run_async("s", "hello"))
    assert response.content == "re:hello"
    assert rows(db, "s") == ["hello", "re:hello"]
    assert thread_spy and all(t.startswith("chatflow-sqlite") for _, t in thread_spy)
    assert hb.max_gap < MAX_STALL


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["session_store", "sqlite_memory", "sqlite_subclass"])
async def test_14_15_other_stores_keep_running_on_the_event_loop(
    kind: str, tmp_path: Path, thread_spy: Any
) -> None:
    class Subclass(SQLiteSessionStore):
        pass

    store: Any = {
        "session_store": SessionStore,
        "sqlite_memory": lambda: SQLiteSessionStore(":memory:"),
        "sqlite_subclass": lambda: Subclass(tmp_path / "sub.db"),
    }[kind]()
    runner, _ = runner_for(store)
    assert (await bounded(runner.run_async("s", "hello"))).content == "re:hello"
    loop_thread = threading.current_thread().name
    assert thread_spy and all(t == loop_thread for _, t in thread_spy)
    if isinstance(store, SQLiteSessionStore):
        assert store._executor is None


# ---------------------------------------------------------------- 4, 5: independent progress


@pytest.mark.asyncio
async def test_4_independent_turn_progresses_while_another_waits_for_sqlite(db: Path) -> None:
    sqlite_runner, _ = runner_for(SQLiteSessionStore(db))
    memory_runner, _ = runner_for(SessionStore(), Engine(delay=0.05))
    SQLiteSessionStore(db)  # schema
    with ExternalWriter(db) as writer:
        async with Heartbeat() as hb:
            blocked = asyncio.ensure_future(sqlite_runner.run_async("sqlite-user", "hola"))
            await asyncio.sleep(0.05)
            t0 = time.perf_counter()
            other = await bounded(memory_runner.run_async("other-user", "ping"))
            other_elapsed = time.perf_counter() - t0
            assert not blocked.done()
        writer.release.set()
        assert (await bounded(blocked)).content == "re:hola"
    assert other.content == "re:ping"
    assert other_elapsed < 0.5
    assert hb.max_gap < MAX_STALL


@pytest.mark.asyncio
async def test_5_concurrent_turns_on_several_sessions_complete_in_order_after_contention(db: Path) -> None:
    store = SQLiteSessionStore(db)
    runner, _ = runner_for(store)
    sessions = [f"user{i}" for i in range(5)]
    with ExternalWriter(db) as writer:
        tasks = [
            asyncio.ensure_future(runner.run_async(sid, f"{sid}-m{n}"))
            for n in range(2)
            for sid in sessions
        ]
        await asyncio.sleep(0.2)
        assert not any(t.done() for t in tasks)
        writer.release.set()
        await bounded(asyncio.gather(*tasks))
    for sid in sessions:
        assert rows(db, sid) == [f"{sid}-m0", f"re:{sid}-m0", f"{sid}-m1", f"re:{sid}-m1"]


# ---------------------------------------------------------------- 6: timeouts


@pytest.mark.asyncio
async def test_6_turn_timeout_is_detected_on_time_and_delivered_after_its_rollback(db: Path) -> None:
    """J1: the turn is cancelled at its deadline even while another session's write is
    stuck; its TimeoutError is raised once its own rollback (which needs the write lock)
    has been stored."""
    store = SQLiteSessionStore(db)
    engine = Engine()
    engine.gates["slow"] = asyncio.Event()  # never released
    runner, _ = runner_for(store, engine)
    await bounded(runner.run_async("a", "first"))
    turn_a = asyncio.ensure_future(runner.run_async("a", "slow", turn_timeout=0.3))
    await bounded(engine.entered_event("slow").wait())
    with ExternalWriter(db) as writer:
        t0 = time.perf_counter()
        blocked = asyncio.ensure_future(runner.run_async("b", "hola"))
        async with Heartbeat() as hb:
            await asyncio.sleep(0.8)
            assert "slow" in engine.cancelled_at, "timeout not detected while the loop is free"
            detected = engine.cancelled_at["slow"] - t0
            assert not turn_a.done(), "TimeoutError must wait for the rollback (J1)"
        writer.release.set()
        with pytest.raises(_TurnTimeout):
            await bounded(turn_a)
        await bounded(blocked)
    assert detected < 0.6
    assert hb.max_gap < MAX_STALL
    assert rows(db, "a") == ["first", "re:first"]


# ---------------------------------------------------------------- 7, 8: cancellation vs rollback


@contextmanager
def slow_model_reply(release: threading.Event, started: threading.Event):
    """Make the model-reply add_message wait in its worker thread until released."""
    original = SQLiteSessionContext.add_message

    def add_message(self: Any, message: Message) -> None:
        if message.role == Role.MODEL and not message.tool_calls:
            started.set()
            assert release.wait(SAFETY)
        original(self, message)

    SQLiteSessionContext.add_message = add_message  # type: ignore[method-assign]
    try:
        yield
    finally:
        SQLiteSessionContext.add_message = original  # type: ignore[method-assign]


@pytest.mark.asyncio
@pytest.mark.parametrize("round_", range(20))
async def test_7_8_cancel_during_a_running_write_rolls_back_after_it(db: Path, round_: int) -> None:
    store = SQLiteSessionStore(db)
    runner, _ = runner_for(store)
    sid = f"s{round_}"
    await bounded(runner.run_async(sid, "before"))
    release, started = threading.Event(), threading.Event()
    with slow_model_reply(release, started):
        task = asyncio.ensure_future(runner.run_async(sid, "cancelled"))
        await wait_event(started)  # the reply's write is running in its worker
        task.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done(), "must wait for the running write before rolling back"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await bounded(task)
    assert rows(db, sid) == ["before", "re:before"], "no write may outlive the rollback"
    assert (await bounded(runner.run_async(sid, "next"))).content == "re:next"


@pytest.mark.asyncio
async def test_8_rollback_waiting_for_an_external_lock_completes_before_returning(db: Path) -> None:
    store = SQLiteSessionStore(db)
    engine = Engine()
    engine.gates["stuck"] = asyncio.Event()
    runner, _ = runner_for(store, engine)
    await bounded(runner.run_async("s", "before"))
    task = asyncio.ensure_future(runner.run_async("s", "stuck"))
    await bounded(engine.entered_event("stuck").wait())
    with ExternalWriter(db) as writer:
        task.cancel()  # the rollback now needs the write lock
        await asyncio.sleep(0.3)
        assert not task.done()
        writer.release.set()
        with pytest.raises(asyncio.CancelledError):
            await bounded(task)
    assert rows(db, "s") == ["before", "re:before"]


# ---------------------------------------------------------------- 9, 10: write and commit errors


@pytest.mark.asyncio
async def test_9_failed_write_propagates_rolls_back_and_releases(db: Path, monkeypatch: Any) -> None:
    store = SQLiteSessionStore(db)
    runner, _ = runner_for(store)
    await bounded(runner.run_async("s", "before"))
    original = SQLiteSessionContext.add_message

    def failing(self: Any, message: Message) -> None:
        if message.role == Role.MODEL:
            raise sqlite3.OperationalError("disk I/O error")
        original(self, message)

    monkeypatch.setattr(SQLiteSessionContext, "add_message", failing)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        await bounded(runner.run_async("s", "broken"))
    monkeypatch.setattr(SQLiteSessionContext, "add_message", original)
    assert rows(db, "s") == ["before", "re:before"]
    assert (await bounded(runner.run_async("s", "next"))).content == "re:next"


@pytest.mark.asyncio
async def test_10_failed_commit_leaves_no_partial_state(db: Path, monkeypatch: Any) -> None:
    store = SQLiteSessionStore(db)
    runner, _ = runner_for(store)
    await bounded(runner.run_async("s", "before"))
    real = memory._get_sqlite_connection
    fail_next_reply = {"armed": True}

    @contextmanager
    def connection(db_path: str):
        with real(db_path) as conn:
            yield conn
            pending = conn.execute(
                "SELECT COUNT(*) FROM chatflow_messages WHERE content = 're:broken'"
            ).fetchone()[0]
            if pending and fail_next_reply["armed"]:
                fail_next_reply["armed"] = False
                conn.rollback()
                raise sqlite3.OperationalError("commit failed")

    monkeypatch.setattr(memory, "_get_sqlite_connection", connection)
    with pytest.raises(sqlite3.OperationalError, match="commit failed"):
        await bounded(runner.run_async("s", "broken"))
    assert rows(db, "s") == ["before", "re:before"]
    assert (await bounded(runner.run_async("s", "next"))).content == "re:next"


# ---------------------------------------------------------------- condition 3: failed rollback


@pytest.mark.asyncio
async def test_failed_rollback_is_reported_as_such_not_as_a_rolled_back_timeout(
    db: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    store = SQLiteSessionStore(db)
    engine = Engine()
    engine.gates["stuck"] = asyncio.Event()
    runner, _ = runner_for(store, engine)

    def failing_restore(self: Any, *args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SQLiteSessionContext, "restore_history", failing_restore)
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.runner"):
        with pytest.raises(sqlite3.OperationalError, match="locked") as raised:
            await bounded(runner.run_async("s", "stuck", turn_timeout=0.2))
    assert not isinstance(raised.value, asyncio.TimeoutError)  # never reported as a timeout
    assert "Rolling back the failed turn of session 's' failed" in caplog.text
    assert rows(db, "s") == ["stuck"]  # what really is stored: the rollback did not happen


# ---------------------------------------------------------------- 16 / R5: queue limit


@pytest.mark.asyncio
async def test_r5_expired_queued_turn_never_runs_nor_loads_the_session(db: Path) -> None:
    store = SQLiteSessionStore(db)
    engine = Engine()
    engine.gates["hold"] = asyncio.Event()
    runner, _ = runner_for(store, engine)
    holder = runner.spawn_turn("s", "hold")
    await bounded(engine.entered_event("hold").wait())
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() + 0.2, None))
    try:
        with pytest.raises(_SessionWaitTimeout):
            await bounded(runner.run_async("s", "late"))
    finally:
        _dispatch_limits.reset(token)
    engine.gates["hold"].set()
    await bounded(holder)
    assert rows(db, "s") == ["hold", "re:hold"]


@pytest.mark.asyncio
async def test_fifo_order_is_kept_with_worker_threads(db: Path) -> None:
    """Turns start waiting for the session in call order even though loading the session
    runs in worker threads (the lock is taken before loading)."""
    store = SQLiteSessionStore(db)
    engine = Engine()
    engine.gates["m0"] = asyncio.Event()
    runner, _ = runner_for(store, engine)
    tasks = [asyncio.ensure_future(runner.run_async("s", f"m{i}")) for i in range(10)]
    await bounded(engine.entered_event("m0").wait())
    engine.gates["m0"].set()
    await bounded(asyncio.gather(*tasks))
    expected: list[str | None] = []
    for i in range(10):
        expected += [f"m{i}", f"re:m{i}"]
    assert rows(db, "s") == expected
