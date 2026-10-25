"""Telegram /reset with a file-backed SQLiteSessionStore under SQLite contention (F-c, test 13).

/reset loads the session and clears its history in worker threads: a locked database no
longer stops the bot's event loop. Its interaction with a running turn is unchanged (not
coordinated, as documented under "Reset and Delete During a Turn").
"""

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import SQLiteSessionContext, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Message, Role

SAFETY = 15.0
MAX_STALL = 0.1
CHAT = 424242


class Engine(BaseEngine):
    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.entered: dict[str, asyncio.Event] = {}

    def entered_event(self, text: str) -> asyncio.Event:
        return self.entered.setdefault(text, asyncio.Event())

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        text = next(m.content for m in reversed(history) if m.role == Role.USER) or ""
        self.entered_event(text).set()
        gate = self.gates.get(text)
        if gate is not None:
            await gate.wait()
        return EngineTurnResult(text=f"re:{text}")


class ExternalWriter:
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
        self._thread.join(SAFETY)


async def max_loop_gap_while(until: asyncio.Event) -> float:
    """Longest gap between event-loop iterations until ``until`` is set."""
    gaps: list[float] = []
    last = time.perf_counter()
    while not until.is_set():
        await asyncio.sleep(0.005)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    return max(gaps) if gaps else 0.0


def rows(db: Path, session_id: str) -> list[str | None]:
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute(
            "SELECT content FROM chatflow_messages WHERE session_id = ? ORDER BY id", (session_id,))]
    finally:
        conn.close()


def reset_update() -> tuple[Any, Any]:
    update = MagicMock()
    update.effective_chat = MagicMock(id=CHAT, type="private")
    update.message = AsyncMock()
    context = MagicMock()
    context.bot = AsyncMock()
    return update, context


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "tg.db"


def build(db: Path, engine: BaseEngine | None = None) -> tuple[TelegramChannel, Runner]:
    runner = Runner(Agent(name="Bot", instructions="x"), session_store=SQLiteSessionStore(db),
                    engine=engine or Engine())
    channel = TelegramChannel(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
    channel.attach(runner)
    return channel, runner


@pytest.mark.asyncio
async def test_13_reset_under_a_locked_database_keeps_the_loop_free(db: Path) -> None:
    channel, runner = build(db)
    await runner.run_async(str(CHAT), "hello")
    update, context = reset_update()
    with ExternalWriter(db) as writer:
        reset = asyncio.ensure_future(channel._handle_reset(update, context))
        stop = asyncio.Event()
        gap_task = asyncio.ensure_future(max_loop_gap_while(stop))
        await asyncio.sleep(0.3)
        assert not reset.done(), "the reset waits for the database"
        writer.release.set()
        await asyncio.wait_for(reset, SAFETY)
        stop.set()
        gap = await gap_task
    assert gap < MAX_STALL
    assert rows(db, str(CHAT)) == []
    update.message.reply_text.assert_called_once_with("Conversation memory has been reset.")


@pytest.mark.asyncio
async def test_13_reset_cancelled_while_clearing_finishes_the_clear_and_does_not_reply(db: Path) -> None:
    channel, runner = build(db)
    await runner.run_async(str(CHAT), "hello")
    update, context = reset_update()
    started, release = threading.Event(), threading.Event()
    original = SQLiteSessionContext.clear_history

    def slow_clear(self: Any) -> None:
        started.set()
        assert release.wait(SAFETY)
        original(self)

    SQLiteSessionContext.clear_history = slow_clear  # type: ignore[method-assign]
    try:
        reset = asyncio.ensure_future(channel._handle_reset(update, context))
        deadline = time.monotonic() + SAFETY
        while not started.is_set():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        reset.cancel()
        await asyncio.sleep(0.05)
        assert not reset.done(), "a started clear is waited for"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reset, SAFETY)
    finally:
        SQLiteSessionContext.clear_history = original  # type: ignore[method-assign]
    assert rows(db, str(CHAT)) == []  # the clear that had started is complete, not partial
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_13_reset_during_a_running_turn_behaves_as_before(db: Path) -> None:
    """Not coordinated with the running turn (documented Known Issue): the reset clears
    what is stored so far and the turn then stores its reply. Same outcome as before F-c."""
    engine = Engine()
    engine.gates["A"] = asyncio.Event()
    channel, runner = build(db, engine)
    await runner.run_async(str(CHAT), "before")
    turn = asyncio.ensure_future(runner.run_async(str(CHAT), "A"))
    await asyncio.wait_for(engine.entered_event("A").wait(), SAFETY)
    update, context = reset_update()
    await asyncio.wait_for(channel._handle_reset(update, context), SAFETY)
    assert rows(db, str(CHAT)) == []
    engine.gates["A"].set()
    assert (await asyncio.wait_for(turn, SAFETY)).content == "re:A"
    assert rows(db, str(CHAT)) == ["re:A"]
