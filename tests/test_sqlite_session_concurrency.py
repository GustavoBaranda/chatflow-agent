"""Concurrency, refresh and rollback integrity of SQLite sessions (#31).

Interleavings are driven by asyncio/threading/multiprocessing events, never by sleeps.
"""

import asyncio
import gc
import logging
import multiprocessing
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import (
    BaseSessionStore,
    SessionContext,
    SessionStore,
    SQLiteSessionStore,
)
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ChatFlowError, ReentrantTurnError, SessionDeletedError
from chatflow_agent.types import Handoff, Message, Role, ToolCall


class GatedEngine(BaseEngine):
    """Answers ``re:<last user message>``.

    Calls for a message listed in ``gates`` wait until its gate is set; messages listed
    in ``fail`` raise after their gate. Records what each call saw.
    """

    def __init__(self, gates: tuple[str, ...] = (), fail: tuple[str, ...] = ()) -> None:
        self.gates = {name: asyncio.Event() for name in gates}
        self.entered = {name: asyncio.Event() for name in gates}
        self.fail = set(fail)
        self.active = 0
        self.max_active = 0
        self.seen: dict[str, list[str | None]] = {}

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        last = [m.content for m in history if m.role == Role.USER][-1] or ""
        self.seen[last] = [m.content for m in history]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)  # yield, so unserialized turns would interleave
            if last in self.gates:
                self.entered[last].set()
                await self.gates[last].wait()
            if last in self.fail:
                raise RuntimeError(f"turn {last} failed")
            return EngineTurnResult(text=f"re:{last}")
        finally:
            self.active -= 1


@pytest.fixture(params=["disk", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path):
    db = tmp_path / "sessions.db" if request.param == "disk" else ":memory:"
    sqlite_store = SQLiteSessionStore(db)
    yield sqlite_store
    sqlite_store.close()


def contents(store: SQLiteSessionStore, session_id: str = "s") -> list[str | None]:
    session = store.get(session_id)
    assert session is not None
    return [m.content for m in session.history]


def raw_rows(db: Path, session_id: str = "s") -> list[tuple[str, str]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT role, content FROM chatflow_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()


async def queue_turn(runner: Runner, message: str, session_id: str = "s") -> asyncio.Task[Any]:
    """Start a turn and let it run until it blocks on the session lock."""
    task = asyncio.create_task(runner.run_async(session_id, message))
    for _ in range(5):
        await asyncio.sleep(0)
    return task


def bot() -> Agent:
    return Agent(name="Bot", instructions="Test")


# --- Serialization -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turns_of_one_session_are_serialized(store: SQLiteSessionStore) -> None:
    engine = GatedEngine(gates=("A",))
    runner = Runner(bot(), session_store=store, engine=engine)

    turn_a = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    turn_b = await queue_turn(runner, "B")
    assert not turn_b.done()
    assert engine.max_active == 1

    engine.gates["A"].set()
    await asyncio.gather(turn_a, turn_b)
    assert engine.max_active == 1
    assert contents(store) == ["A", "re:A", "B", "re:B"]


@pytest.mark.asyncio
async def test_ten_concurrent_turns_keep_order_and_see_previous_replies(
    store: SQLiteSessionStore,
) -> None:
    engine = GatedEngine()
    runner = Runner(bot(), session_store=store, engine=engine)

    await asyncio.gather(*(runner.run_async("s", f"m{i}") for i in range(10)))

    session = store.get("s")
    assert session is not None
    assert [m.role for m in session.history] == [Role.USER, Role.MODEL] * 10
    assert engine.max_active == 1
    for i in range(1, 10):
        assert f"re:m{i - 1}" in engine.seen[f"m{i}"]


@pytest.mark.asyncio
async def test_different_sessions_run_concurrently(store: SQLiteSessionStore) -> None:
    engine = GatedEngine(gates=("A", "B"))
    runner = Runner(bot(), session_store=store, engine=engine)

    turn_a = asyncio.create_task(runner.run_async("s1", "A"))
    turn_b = asyncio.create_task(runner.run_async("s2", "B"))
    await engine.entered["A"].wait()
    await engine.entered["B"].wait()
    assert engine.max_active == 2

    engine.gates["A"].set()
    engine.gates["B"].set()
    await asyncio.gather(turn_a, turn_b)
    assert contents(store, "s1") == ["A", "re:A"]
    assert contents(store, "s2") == ["B", "re:B"]


# --- Refresh inside the lock ---------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_queued_turn_keeps_previous_successful_turn(
    store: SQLiteSessionStore,
) -> None:
    engine = GatedEngine(gates=("A", "B"), fail=("B",))
    runner = Runner(bot(), session_store=store, engine=engine)

    turn_a = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    turn_b = await queue_turn(runner, "B")
    engine.gates["A"].set()
    assert (await turn_a).content == "re:A"

    engine.gates["B"].set()
    with pytest.raises(RuntimeError, match="turn B failed"):
        await turn_b

    assert engine.seen["B"] == ["A", "re:A", "B"]
    assert contents(store) == ["A", "re:A"]


@pytest.mark.asyncio
async def test_cancelled_turn_reverts_only_itself_and_queued_turn_proceeds(
    store: SQLiteSessionStore,
) -> None:
    engine = GatedEngine(gates=("A",))
    runner = Runner(bot(), session_store=store, engine=engine)
    await runner.run_async("s", "hello")

    turn_a = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    turn_b = await queue_turn(runner, "B")
    turn_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn_a
    await turn_b

    assert engine.seen["B"] == ["hello", "re:hello", "B"]
    assert contents(store) == ["hello", "re:hello", "B", "re:B"]


@pytest.mark.asyncio
async def test_queued_turn_sees_committed_handoff_and_metadata(
    store: SQLiteSessionStore,
) -> None:
    billing = Agent(name="Billing", instructions="Billing")

    def route() -> Handoff:
        """Route to billing."""
        return Handoff(target_agent_name="Billing", context_updates={"ticket": "T1"})

    triage = Agent(name="Triage", instructions="Triage", tools=[route], handoffs=[billing])
    gate, entered = asyncio.Event(), asyncio.Event()
    seen_agent: dict[str, str] = {}

    class HandoffEngine(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            last = history[-1]
            if last.role == Role.USER and last.content == "A":
                entered.set()
                await gate.wait()
                return EngineTurnResult(tool_calls=[ToolCall(id="c1", name="route", args={})])
            if last.role == Role.USER and last.content == "B":
                seen_agent["B"] = agent.name
            return EngineTurnResult(text="ok")

    runner = Runner(triage, session_store=store, engine=HandoffEngine())
    turn_a = asyncio.create_task(runner.run_async("s", "A", metadata={"channel": "wa"}))
    await entered.wait()
    turn_b = await queue_turn(runner, "B")
    gate.set()
    await asyncio.gather(turn_a, turn_b)

    assert seen_agent["B"] == "Billing"
    session = store.get("s")
    assert session is not None
    assert session.active_agent.name == "Billing"
    assert session.metadata == {"channel": "wa", "ticket": "T1"}


# --- Rollback and pruning (#34) ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("db_kind", ["disk", "memory"])
async def test_pruned_history_is_restored_when_queued_turn_fails(
    db_kind: str, tmp_path: Path
) -> None:
    db = tmp_path / "pruned.db" if db_kind == "disk" else ":memory:"
    store = SQLiteSessionStore(db, default_max_turns=1)
    engine = GatedEngine(gates=("A", "B"), fail=("B",))
    runner = Runner(bot(), session_store=store, engine=engine)
    await runner.run_async("s", "first")

    turn_a = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    turn_b = await queue_turn(runner, "B")
    engine.gates["A"].set()
    await turn_a

    engine.gates["B"].set()
    with pytest.raises(RuntimeError):
        await turn_b

    # Turn B pruned turn A from the one-turn window before failing; rollback restores it.
    assert contents(store) == ["A", "re:A"]
    store.close()


@pytest.mark.asyncio
async def test_failed_turn_keeps_rows_committed_by_another_store(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "shared.db"
    failing = GatedEngine(gates=("B",), fail=("B",))
    store_1 = SQLiteSessionStore(db)
    store_2 = SQLiteSessionStore(db)
    runner_1 = Runner(bot(), session_store=store_1, engine=GatedEngine())
    runner_2 = Runner(bot(), session_store=store_2, engine=failing)

    turn_b = asyncio.create_task(runner_2.run_async("s", "B"))
    await failing.entered["B"].wait()
    # Another store instance does not share the lock: its turn commits meanwhile.
    await runner_1.run_async("s", "A")
    failing.gates["B"].set()
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.memory"):
        with pytest.raises(RuntimeError):
            await turn_b

    assert raw_rows(db) == [("user", "A"), ("model", "re:A")]
    assert "modified outside its lock" in caplog.text


@pytest.mark.asyncio
async def test_failed_turn_after_external_prune_keeps_rows_of_other_writer(
    tmp_path: Path,
) -> None:
    """Out-of-model writer + pruning: this turn's own pruned rows come back, theirs stay."""
    db = tmp_path / "prune_shared.db"
    failing = GatedEngine(gates=("B",), fail=("B",))
    store_2 = SQLiteSessionStore(db, default_max_turns=2)
    runner_2 = Runner(bot(), session_store=store_2, engine=failing)
    runner_1 = Runner(
        bot(), session_store=SQLiteSessionStore(db, default_max_turns=50), engine=GatedEngine()
    )
    await runner_2.run_async("s", "old")
    await runner_2.run_async("s", "mid")

    # Turn B prunes "old" (window of 2), then waits; another store commits "A".
    turn_b = asyncio.create_task(runner_2.run_async("s", "B"))
    await failing.entered["B"].wait()
    await runner_1.run_async("s", "A")
    failing.gates["B"].set()
    with pytest.raises(RuntimeError):
        await turn_b

    assert raw_rows(db) == [
        ("user", "old"),
        ("model", "re:old"),
        ("user", "mid"),
        ("model", "re:mid"),
        ("user", "A"),
        ("model", "re:A"),
    ]


@pytest.mark.asyncio
async def test_failed_turn_does_not_undo_an_external_reset(tmp_path: Path) -> None:
    db = tmp_path / "reset.db"
    failing = GatedEngine(gates=("B",), fail=("B",))
    store = SQLiteSessionStore(db)
    runner = Runner(bot(), session_store=store, engine=failing)
    await runner.run_async("s", "secret")

    turn_b = asyncio.create_task(runner.run_async("s", "B"))
    await failing.entered["B"].wait()
    # A reset made through another context while the turn runs.
    other = store.get("s")
    assert other is not None
    other.clear_history()
    failing.gates["B"].set()
    with pytest.raises(RuntimeError):
        await turn_b

    assert raw_rows(db) == []


@pytest.mark.asyncio
async def test_rollback_check_and_revert_run_in_one_immediate_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "txn.db"
    engine = GatedEngine(gates=("B",), fail=("B",))
    runner = Runner(bot(), session_store=SQLiteSessionStore(db), engine=engine)
    await runner.run_async("s", "A")

    real_connect = sqlite3.connect
    traces: list[list[str]] = []

    def tracing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        statements: list[str] = []
        traces.append(statements)
        conn.set_trace_callback(lambda sql: statements.append(" ".join(sql.split())))
        return conn

    monkeypatch.setattr("chatflow_agent.core.memory.sqlite3.connect", tracing_connect)
    turn_b = asyncio.create_task(runner.run_async("s", "B"))
    await engine.entered["B"].wait()

    # No write transaction is held while the model is called.
    probe = real_connect(db, timeout=0)
    try:
        probe.execute("BEGIN IMMEDIATE")
        probe.rollback()
    finally:
        probe.close()

    engine.gates["B"].set()
    with pytest.raises(RuntimeError):
        await turn_b

    rollback = [t for t in traces if "BEGIN IMMEDIATE" in t]
    assert len(rollback) == 1
    statements = rollback[0]
    begin = statements.index("BEGIN IMMEDIATE")
    after = statements[begin + 1 :]
    assert any(s.startswith("SELECT id FROM chatflow_messages") for s in after)
    assert any(s.startswith("DELETE FROM chatflow_messages") for s in after)
    assert after[-1] == "COMMIT"
    assert raw_rows(db) == [("user", "A"), ("model", "re:A")]


# Two processes: module-level helpers so a spawned child process can import them.


class _ChildFailingEngine(BaseEngine):
    def __init__(self, entered: Any, release: Any) -> None:
        self.entered = entered
        self.release = release

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        self.entered.set()
        await asyncio.to_thread(self.release.wait, 60)
        raise RuntimeError("child turn failed")


def _run_failing_turn_in_child(db: str, entered: Any, release: Any) -> None:
    runner = Runner(
        Agent(name="Bot", instructions="Test"),
        session_store=SQLiteSessionStore(db),
        engine=_ChildFailingEngine(entered, release),
    )
    try:
        runner.run("s", "child")
    except RuntimeError:
        pass


@pytest.mark.asyncio
async def test_failed_turn_in_another_process_keeps_committed_rows(tmp_path: Path) -> None:
    db = tmp_path / "processes.db"
    SQLiteSessionStore(db).close()  # create the schema before the child starts
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    child = context.Process(target=_run_failing_turn_in_child, args=(str(db), entered, release))
    child.start()
    try:
        assert await asyncio.to_thread(entered.wait, 60), "child turn did not start"
        runner = Runner(bot(), session_store=SQLiteSessionStore(db), engine=GatedEngine())
        await runner.run_async("s", "parent")
    finally:
        release.set()
        await asyncio.to_thread(child.join, 60)
    assert child.exitcode == 0
    assert raw_rows(db) == [("user", "parent"), ("model", "re:parent")]


# --- Event loops, threads and lock lifecycle -----------------------------------------


def test_repeated_asyncio_run_with_contention_and_sync_runs(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "loops.db")
    runner = Runner(bot(), session_store=store, engine=GatedEngine())

    for i in range(3):

        async def burst(i: int = i) -> None:
            await asyncio.gather(runner.run_async("s", f"a{i}"), runner.run_async("s", f"b{i}"))

        asyncio.run(burst())
        runner.run("s", f"sync{i}")

    assert len(contents(store)) == 18


@pytest.mark.asyncio
async def test_contexts_of_one_session_share_a_lock_per_event_loop(
    store: SQLiteSessionStore,
) -> None:
    # Adapted from the lock-identity test in PR #33 by @KeuqS.
    first = store.get_or_create("shared", default_agent=bot())
    second = store.get_or_create("shared", default_agent=bot())
    other = store.get_or_create("other", default_agent=bot())

    assert first is not second
    assert first.lock is second.lock
    assert first.lock is not other.lock

    async def lock_in_new_loop() -> asyncio.Lock:
        return store.get_or_create("shared", default_agent=bot()).lock

    other_loop_lock = await asyncio.to_thread(asyncio.run, lock_in_new_loop())
    assert other_loop_lock is not first.lock


def test_lock_registry_does_not_retain_locks(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "gc.db")
    runner = Runner(bot(), session_store=store, engine=GatedEngine())
    for i in range(5):
        runner.run(f"s{i}", "hi")
    gc.collect()
    assert sum(len(locks) for locks in store._session_locks._locks.values()) == 0


@pytest.mark.asyncio
async def test_delete_and_clear_keep_the_lock_of_a_running_turn(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "delete.db")
    engine = GatedEngine(gates=("A",))
    runner = Runner(bot(), session_store=store, engine=engine)

    turn_a = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    store.delete("s")
    store.clear()
    turn_b = await queue_turn(runner, "B")
    assert not turn_b.done()
    assert engine.max_active == 1

    engine.gates["A"].set()
    # Documented contract ("Deleting a session"): the running turn fails with
    # SessionDeletedError and leaves nothing behind; the queued turn waits for it and then
    # starts a new, empty session. (This used to pass only when the queued turn happened to
    # recreate the session before the running one stored its reply.)
    with pytest.raises(SessionDeletedError):
        await turn_a
    await turn_b
    assert engine.max_active == 1
    assert contents(store) == ["B", "re:B"]


@pytest.mark.asyncio
async def test_turn_queued_behind_a_deleted_session_starts_a_new_session(
    tmp_path: Path,
) -> None:
    billing = Agent(name="Billing", instructions="Billing")
    triage = Agent(name="Triage", instructions="Triage", handoffs=[billing])
    db = tmp_path / "deleted.db"
    store = SQLiteSessionStore(db)
    engine = GatedEngine(gates=("A",))
    runner = Runner(triage, session_store=store, engine=engine)
    await runner.run_async("s", "old secret", metadata={"user": "u1"})
    old = store.get("s")
    assert old is not None
    old.set_active_agent(billing)

    turn_a = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    turn_b = await queue_turn(runner, "B")
    store.delete("s")
    engine.gates["A"].set()
    # The running turn cannot store its reply into the deleted session: it fails and
    # rolls back without leaving rows. The queued turn then starts a new, empty session.
    with pytest.raises(SessionDeletedError):
        await turn_a
    await turn_b

    assert engine.seen["B"] == ["B"]
    assert raw_rows(db) == [("user", "B"), ("model", "re:B")]
    session = store.get("s")
    assert session is not None
    assert session.active_agent.name == "Triage"
    assert session.metadata == {}


def test_threads_with_their_own_loops_are_not_serialized(tmp_path: Path) -> None:
    """Documents a limitation: turns from two threads (two loops) are not serialized.

    The rollback guard still keeps the other thread's committed rows.
    """
    db = tmp_path / "threads.db"
    store = SQLiteSessionStore(db)
    a_inside, b_done, release_a = threading.Event(), threading.Event(), threading.Event()

    class ThreadEngine(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            last = history[-1].content
            if last == "A":
                a_inside.set()
                await asyncio.to_thread(release_a.wait, 60)
                return EngineTurnResult(text="re:A")
            raise RuntimeError("turn B failed")

    runner = Runner(bot(), session_store=store, engine=ThreadEngine())
    results: dict[str, str] = {}

    def run(message: str) -> None:
        try:
            results[message] = runner.run("s", message).content
        except RuntimeError as err:
            results[message] = str(err)
        finally:
            if message == "B":
                b_done.set()

    thread_a = threading.Thread(target=run, args=("A",))
    thread_a.start()
    assert a_inside.wait(60)
    thread_b = threading.Thread(target=run, args=("B",))
    thread_b.start()
    ran_while_a_held_lock = b_done.wait(60)
    release_a.set()
    thread_a.join(60)
    thread_b.join(60)

    assert ran_while_a_held_lock
    assert results == {"A": "re:A", "B": "turn B failed"}
    assert raw_rows(db) == [("user", "A"), ("model", "re:A")]


# --- Re-entrancy ---------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["sqlite", "memory"])
async def test_reentrant_turn_for_the_same_session_raises(store_kind: str, tmp_path: Path) -> None:
    session_store = (
        SQLiteSessionStore(tmp_path / "reentry.db") if store_kind == "sqlite" else SessionStore()
    )
    holder: dict[str, Runner] = {}

    async def ask_again() -> str:
        """Run another turn for the same session."""
        try:
            await holder["runner"].run_async("s", "nested")
        except RuntimeError as err:
            return f"refused: {err}"
        return "nested turn ran"

    tool_results: list[Any] = []

    class ToolEngine(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            last = history[-1]
            if last.role == Role.USER:
                return EngineTurnResult(tool_calls=[ToolCall(id="c1", name="ask_again", args={})])
            assert last.tool_results is not None
            tool_results.extend(r.content for r in last.tool_results)
            return EngineTurnResult(text="done")

    agent = Agent(name="Bot", instructions="Test", tools=[ask_again])
    holder["runner"] = Runner(agent, session_store=session_store, engine=ToolEngine())

    response = await asyncio.wait_for(holder["runner"].run_async("s", "outer"), timeout=30)

    assert response.content == "done"
    assert len(tool_results) == 1
    assert tool_results[0].startswith("refused:")
    assert "already running a turn" in tool_results[0]


@pytest.mark.asyncio
async def test_turn_started_from_another_task_waits_instead_of_raising(
    store: SQLiteSessionStore,
) -> None:
    """An independent task started by a tool queues behind the running turn.

    Contract (#120): an independent follow-up turn is started with ``spawn_turn()``; it waits
    for the running turn and then runs. A task created with ``create_task()`` inherits the
    turn and is refused with ``ReentrantTurnError`` while that turn runs (see
    ``test_task_started_inside_a_running_turn_is_refused_by_contract``). The create_task()
    form used to pass here only because the fake engine never yielded to the event loop.
    """
    background: list[asyncio.Task[Any]] = []

    async def follow_up() -> str:
        """Schedule a follow-up turn for the same session."""
        background.append(holder["runner"].spawn_turn("s", "follow-up"))
        return "scheduled"

    class ToolEngine(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            last = history[-1]
            if last.role == Role.USER and last.content == "outer":
                return EngineTurnResult(tool_calls=[ToolCall(id="c1", name="follow_up", args={})])
            return EngineTurnResult(text=f"re:{last.content}")

    holder = {
        "runner": Runner(
            Agent(name="Bot", instructions="Test", tools=[follow_up]),
            session_store=store,
            engine=ToolEngine(),
        )
    }
    await holder["runner"].run_async("s", "outer")
    assert len(background) == 1
    assert (await background[0]).content == "re:follow-up"
    assert contents(store)[-2:] == ["follow-up", "re:follow-up"]


# --- Rollback keeps agent and metadata written by others (H-01) ----------------------


class BlockThenFail(BaseEngine):
    """The call for message 'bad' waits on `release`, then raises."""

    def __init__(self) -> None:
        self.inside = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        if history[-1].content == "bad":
            self.inside.set()
            await self.release.wait()
            raise RuntimeError("turn failed")
        return EngineTurnResult(text="ok")


def session_row(db: Path) -> tuple[str, str]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT active_agent_name, metadata_json FROM chatflow_sessions WHERE session_id = 's'"
        ).fetchone()
    finally:
        conn.close()


async def failing_turn_with_foreign_write(
    tmp_path: Path, write: Any, turn_metadata: dict[str, Any] | None = None
) -> tuple[Any, Any]:
    db = tmp_path / "foreign.db"
    billing = Agent(name="Billing", instructions="x")
    triage = Agent(name="Triage", instructions="x", handoffs=[billing])
    store = SQLiteSessionStore(db)
    engine = BlockThenFail()
    runner = Runner(triage, session_store=store, engine=engine)
    await runner.run_async("s", "hello", metadata={"channel": "wa"})
    turn = asyncio.create_task(runner.run_async("s", "bad", metadata=turn_metadata))
    await engine.inside.wait()
    write(store, db, billing)
    during = session_row(db)
    engine.release.set()
    with pytest.raises(RuntimeError, match="turn failed"):
        await turn
    return during, session_row(db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "write",
    [
        pytest.param(
            lambda store, db, billing: store.get("s").update_metadata({"vip": True}),
            id="metadata-same-store",
        ),
        pytest.param(
            lambda store, db, billing: (
                SQLiteSessionStore(db).get("s").update_metadata({"vip": True})
            ),
            id="metadata-other-store",
        ),
        pytest.param(
            lambda store, db, billing: store.get("s").set_active_agent(billing),
            id="agent-same-store",
        ),
    ],
)
async def test_failed_turn_keeps_agent_and_metadata_written_by_another_context(
    tmp_path: Path, write: Any
) -> None:
    during, after = await failing_turn_with_foreign_write(tmp_path, write)
    assert after == during


@pytest.mark.asyncio
async def test_failed_turn_that_wrote_metadata_keeps_later_foreign_metadata(tmp_path: Path) -> None:
    def write(store: SQLiteSessionStore, db: Path, billing: Agent) -> None:
        store.get("s").update_metadata({"vip": True})

    during, after = await failing_turn_with_foreign_write(tmp_path, write, {"wamid": "w2"})
    assert after == during


@pytest.mark.asyncio
async def test_failed_turn_alone_still_reverts_its_own_metadata(tmp_path: Path) -> None:
    during, after = await failing_turn_with_foreign_write(
        tmp_path, lambda *args: None, {"wamid": "w2"}
    )
    assert after == ("Triage", '{"channel": "wa"}')


# --- Writes into a deleted session (D2) ---------------------------------------------


def test_session_deleted_error_is_a_chatflow_error() -> None:
    assert issubclass(SessionDeletedError, ChatFlowError)


def test_add_message_to_a_deleted_session_stores_nothing(tmp_path: Path) -> None:
    db = tmp_path / "gone.db"
    store = SQLiteSessionStore(db)
    context = store.get_or_create("s", default_agent=bot())
    context.add_message(Message(role=Role.USER, content="kept until delete"))
    store.delete("s")
    before = list(context.history)
    with pytest.raises(SessionDeletedError):
        context.add_message(Message(role=Role.MODEL, content="late"))
    assert raw_rows(db) == []
    assert context.history == before


@pytest.mark.asyncio
async def test_delete_during_a_running_turn_leaves_no_rows(tmp_path: Path) -> None:
    db = tmp_path / "delete.db"
    store = SQLiteSessionStore(db)
    engine = GatedEngine(gates=("A",))
    runner = Runner(bot(), session_store=store, engine=engine)
    await runner.run_async("s", "old secret")
    turn = asyncio.create_task(runner.run_async("s", "A"))
    await engine.entered["A"].wait()
    store.delete("s")
    engine.gates["A"].set()
    with pytest.raises(SessionDeletedError):
        await turn
    assert raw_rows(db) == []
    # A new message later starts an empty session; nothing deleted comes back (D3).
    assert (await runner.run_async("s", "new")).content == "re:new"
    assert raw_rows(db) == [("user", "new"), ("model", "re:new")]


# --- Nested turns of the same session (H-02) -----------------------------------------

SAFETY = 10.0


async def no_deadlock(awaitable: Any) -> Any:
    """Await with a safety net: hitting it means a real deadlock, not a slow wait."""
    try:
        return await asyncio.wait_for(awaitable, SAFETY)
    except asyncio.TimeoutError:
        pytest.fail("deadlock: no progress within the safety timeout")


@pytest.fixture(params=["sqlite", "memory"])
def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    return SQLiteSessionStore(tmp_path / "s.db") if request.param == "sqlite" else SessionStore()


class ToolTurnEngine(BaseEngine):
    """'outer' calls the tool; after the tool the parent turn keeps working until released."""

    def __init__(self) -> None:
        self.parent_may_finish = asyncio.Event()
        self.parent_may_finish.set()

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        last = history[-1]
        if last.role == Role.USER and last.content == "outer":
            return EngineTurnResult(tool_calls=[ToolCall(id="c1", name="tool", args={})])
        if last.role == Role.TOOL:
            await self.parent_may_finish.wait()
        return EngineTurnResult(text=f"re:{last.content}")


def build_tool_runner(store: Any, body: Any) -> tuple[Runner, list[Any], ToolTurnEngine]:
    outcomes: list[Any] = []
    engine = ToolTurnEngine()

    async def tool() -> str:
        """tool"""
        try:
            value = await body(runner)
        except Exception as err:  # noqa: BLE001 - recorded for the assertions
            outcomes.append(err)
            return "refused"
        outcomes.append(value)
        return "done"

    runner = Runner(
        Agent(name="Bot", instructions="x", tools=[tool]), session_store=store, engine=engine
    )
    return runner, outcomes, engine


def is_reentrant_refusal(value: Any) -> bool:
    if isinstance(value, ReentrantTurnError):
        return True
    return any(isinstance(e, ReentrantTurnError) for e in getattr(value, "exceptions", ()))


async def _nested_turn(runner: Runner) -> Any:
    return await runner.run_async("s", "nested")


async def _await_task(runner: Runner) -> Any:
    return await asyncio.create_task(_nested_turn(runner))


async def _gather(runner: Runner) -> Any:
    return (await asyncio.gather(_nested_turn(runner)))[0]


async def _wait_for(runner: Runner) -> Any:
    return await asyncio.wait_for(_nested_turn(runner), 60)


async def _task_group(runner: Runner) -> Any:
    async with asyncio.TaskGroup() as group:
        task = group.create_task(_nested_turn(runner))
    return task.result()


async def _sync_tool_thread(runner: Runner) -> Any:
    return await asyncio.to_thread(runner.run, "s", "nested")


CIRCULAR_WAITS = [_await_task, _gather, _wait_for, _sync_tool_thread]
if sys.version_info >= (3, 11):
    CIRCULAR_WAITS.append(_task_group)


def test_reentrant_turn_error_keeps_runtime_error_compatibility() -> None:
    assert issubclass(ReentrantTurnError, ChatFlowError)
    assert issubclass(ReentrantTurnError, RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", CIRCULAR_WAITS, ids=lambda f: f.__name__.strip("_"))
async def test_turn_waiting_on_a_turn_of_its_own_session_is_refused(
    any_store: Any, body: Any
) -> None:
    runner, outcomes, _ = build_tool_runner(any_store, body)
    await no_deadlock(runner.run_async("s", "outer"))
    assert len(outcomes) == 1 and is_reentrant_refusal(outcomes[0])


@pytest.mark.asyncio
async def test_task_started_inside_a_running_turn_is_refused_by_contract(any_store: Any) -> None:
    spawned: list[asyncio.Task[Any]] = []
    scheduled = asyncio.Event()

    async def body(runner: Runner) -> str:
        spawned.append(asyncio.create_task(_nested_turn(runner)))
        scheduled.set()
        return "scheduled"

    runner, _, engine = build_tool_runner(any_store, body)
    engine.parent_may_finish.clear()
    outer = asyncio.create_task(runner.run_async("s", "outer"))
    await no_deadlock(scheduled.wait())
    with pytest.raises(ReentrantTurnError):
        await no_deadlock(spawned[0])  # the error is retrievable from the task
    engine.parent_may_finish.set()
    await no_deadlock(outer)


@pytest.mark.asyncio
async def test_spawn_turn_queues_a_follow_up_after_the_running_turn(any_store: Any) -> None:
    spawned: list[asyncio.Task[Any]] = []
    scheduled = asyncio.Event()

    async def body(runner: Runner) -> str:
        spawned.append(runner.spawn_turn("s", "follow-up"))
        scheduled.set()
        return "scheduled"

    runner, _, engine = build_tool_runner(any_store, body)
    engine.parent_may_finish.clear()
    outer = asyncio.create_task(runner.run_async("s", "outer"))
    await no_deadlock(scheduled.wait())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not spawned[0].done()  # waiting for the running turn, not refused
    engine.parent_may_finish.set()
    await no_deadlock(outer)
    assert (await no_deadlock(spawned[0])).content == "re:follow-up"
    session = runner.get_session("s")
    assert session is not None
    assert [m.content for m in session.history if m.role == Role.USER] == ["outer", "follow-up"]


@pytest.mark.asyncio
async def test_spawned_turn_survives_without_caller_reference_and_reports_errors(
    any_store: Any,
) -> None:
    class FailFollowUp(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            await asyncio.sleep(0)
            if history[-1].content == "follow-up":
                raise RuntimeError("follow-up failed")
            return EngineTurnResult(text="ok")

    runner = Runner(bot(), session_store=any_store, engine=FailFollowUp())
    task_id = id(runner.spawn_turn("s", "follow-up"))  # the caller keeps no reference
    gc.collect()
    assert [id(t) for t in runner._spawned_turns] == [task_id]
    (task,) = runner._spawned_turns
    with pytest.raises(RuntimeError, match="follow-up failed"):
        await no_deadlock(task)
    assert runner._spawned_turns == set()


@pytest.mark.asyncio
async def test_task_that_starts_its_turn_after_the_parent_ended_runs(any_store: Any) -> None:
    go = asyncio.Event()
    spawned: list[asyncio.Task[Any]] = []

    async def later(runner: Runner) -> Any:
        await go.wait()
        return await _nested_turn(runner)

    async def body(runner: Runner) -> str:
        spawned.append(asyncio.create_task(later(runner)))
        return "scheduled"

    runner, _, _ = build_tool_runner(any_store, body)
    await no_deadlock(runner.run_async("s", "outer"))
    go.set()
    assert (await no_deadlock(spawned[0])).content == "re:nested"


@pytest.mark.asyncio
async def test_turn_context_does_not_leak_into_later_turns(any_store: Any) -> None:
    runner = Runner(bot(), session_store=any_store, engine=GatedEngine())
    await runner.run_async("s", "first")
    # Same task, sequential turns: no refusal once the previous turn has ended.
    await no_deadlock(runner.run_async("s", "second"))
    third = asyncio.create_task(runner.run_async("s", "third"))
    assert (await no_deadlock(third)).content == "re:third"


@pytest.mark.asyncio
async def test_independent_turn_of_the_same_session_waits(any_store: Any) -> None:
    runner, _, engine = build_tool_runner(any_store, lambda r: asyncio.sleep(0, "noop"))
    engine.parent_may_finish.clear()
    first = asyncio.create_task(runner.run_async("s", "outer"))
    for _ in range(5):
        await asyncio.sleep(0)
    second = asyncio.create_task(runner.run_async("s", "independent"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not second.done()
    engine.parent_may_finish.set()
    await no_deadlock(first)
    assert (await no_deadlock(second)).content == "re:independent"


@pytest.mark.asyncio
async def test_cancelling_the_parent_lets_a_spawned_turn_run(any_store: Any) -> None:
    spawned: list[asyncio.Task[Any]] = []
    scheduled = asyncio.Event()

    async def body(runner: Runner) -> str:
        spawned.append(runner.spawn_turn("s", "follow-up"))
        scheduled.set()
        return "scheduled"

    runner, _, engine = build_tool_runner(any_store, body)
    engine.parent_may_finish.clear()
    outer = asyncio.create_task(runner.run_async("s", "outer"))
    await no_deadlock(scheduled.wait())
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert (await no_deadlock(spawned[0])).content == "re:follow-up"
    session = runner.get_session("s")
    assert session is not None
    assert [m.content for m in session.history if m.role == Role.USER] == ["follow-up"]


@pytest.mark.asyncio
async def test_cancelling_a_waiting_spawned_turn_leaves_the_lock_usable(any_store: Any) -> None:
    spawned: list[asyncio.Task[Any]] = []
    scheduled = asyncio.Event()

    async def body(runner: Runner) -> str:
        spawned.append(runner.spawn_turn("s", "follow-up"))
        scheduled.set()
        return "scheduled"

    runner, _, engine = build_tool_runner(any_store, body)
    engine.parent_may_finish.clear()
    outer = asyncio.create_task(runner.run_async("s", "outer"))
    await no_deadlock(scheduled.wait())
    for _ in range(3):
        await asyncio.sleep(0)
    spawned[0].cancel()
    engine.parent_may_finish.set()
    await no_deadlock(outer)
    with pytest.raises(asyncio.CancelledError):
        await spawned[0]
    assert (await no_deadlock(runner.run_async("s", "after"))).content == "re:after"


@pytest.mark.asyncio
async def test_nested_turn_of_another_session_is_allowed(any_store: Any) -> None:
    async def body(runner: Runner) -> Any:
        return (await runner.run_async("other", "hi")).content

    runner, outcomes, _ = build_tool_runner(any_store, body)
    await no_deadlock(runner.run_async("s", "outer"))
    assert outcomes == ["re:hi"]


@pytest.mark.asyncio
async def test_cross_session_cycle_is_refused(any_store: Any) -> None:
    holder: dict[str, Runner] = {}

    async def back_to_first() -> str:
        """Run a turn of the first session again."""
        try:
            await holder["runner"].run_async("s", "again")
        except ReentrantTurnError:
            return "refused"
        return "ran"

    async def to_other() -> str:
        """Run a turn of another session."""
        return (await holder["runner"].run_async("other", "go-back")).content

    class Chain(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            last = history[-1]
            if last.role == Role.USER and last.content == "outer":
                return EngineTurnResult(tool_calls=[ToolCall(id="c1", name="to_other", args={})])
            if last.role == Role.USER and last.content == "go-back":
                return EngineTurnResult(
                    tool_calls=[ToolCall(id="c2", name="back_to_first", args={})]
                )
            if last.role == Role.TOOL and last.tool_results:
                return EngineTurnResult(text=str(last.tool_results[0].content))
            return EngineTurnResult(text="ok")

    agent = Agent(name="Bot", instructions="x", tools=[to_other, back_to_first])
    holder["runner"] = Runner(agent, session_store=any_store, engine=Chain())
    result = await no_deadlock(holder["runner"].run_async("s", "outer"))
    assert result.content == "refused"


class DelegatingStore(BaseSessionStore):
    """A custom store that is not one of the built-in classes."""

    def __init__(self) -> None:
        self._inner = SessionStore()

    def get_or_create(
        self, session_id: str, default_agent: Agent, metadata: dict[str, Any] | None = None
    ) -> SessionContext:
        return self._inner.get_or_create(session_id, default_agent, metadata)

    def get(self, session_id: str) -> SessionContext | None:
        return self._inner.get(session_id)

    def delete(self, session_id: str) -> bool:
        return self._inner.delete(session_id)

    def list_sessions(self) -> list[str]:
        return self._inner.list_sessions()

    def clear(self) -> None:
        self._inner.clear()


@pytest.mark.asyncio
async def test_custom_store_gets_the_same_reentrancy_contract() -> None:
    runner, outcomes, _ = build_tool_runner(DelegatingStore(), _await_task)
    await no_deadlock(runner.run_async("s", "outer"))
    assert is_reentrant_refusal(outcomes[0])
    follow_up = runner.spawn_turn("s", "follow-up")
    assert (await no_deadlock(follow_up)).content == "re:follow-up"
