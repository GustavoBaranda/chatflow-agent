"""Residual synchronous tools keep their session fenced after turn cancellation."""

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core._sync_work import (
    _registry,
    _run_sync_tool,
    _tasks,
    _track_sync_work,
    _wait_for_sync_work,
)
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import (
    Runner,
    _dispatch_limits,
    _DispatchLimits,
    _SessionWaitTimeout,
    _TurnTimeout,
)
from chatflow_agent.core.tools import Tool
from chatflow_agent.exceptions import ReentrantTurnError
from chatflow_agent.types import Message, Role, ToolCall

SAFETY = 10.0


@pytest.fixture(params=["memory", "sqlite_memory", "sqlite_file"])
def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "sqlite_file":
        return SQLiteSessionStore(tmp_path / "sessions.db")
    if request.param == "sqlite_memory":
        return SQLiteSessionStore(":memory:")
    return SessionStore()


async def bounded(awaitable: Any) -> Any:
    return await asyncio.wait_for(awaitable, SAFETY)


async def wait_thread_event(event: threading.Event) -> None:
    assert await bounded(asyncio.to_thread(event.wait, SAFETY))


class CallingEngine(BaseEngine):
    """Call one tool for every user message, then return after its result."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.started: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []
        self.results: list[Any] = []

    def started_event(self, text: str) -> asyncio.Event:
        return self.started.setdefault(text, asyncio.Event())

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        last = history[-1]
        if last.role == Role.USER:
            text = last.content or ""
            self.calls.append(text)
            self.started_event(text).set()
            return EngineTurnResult(
                tool_calls=[
                    ToolCall(
                        id=f"call-{len(self.calls)}", name=self.tool_name, args={"label": text}
                    )
                ]
            )
        assert last.role == Role.TOOL
        assert last.tool_results is not None
        self.results.extend(result.content for result in last.tool_results)
        return EngineTurnResult(text="done")


class OneToolEngine(BaseEngine):
    """Call one tool only for the outer message; nested messages return directly."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.user_messages: list[str] = []

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        last = history[-1]
        if last.role == Role.USER:
            text = last.content or ""
            self.user_messages.append(text)
            if text == "outer":
                return EngineTurnResult(
                    tool_calls=[ToolCall(id="outer-call", name=self.tool_name, args={})]
                )
            return EngineTurnResult(text=f"reply:{text}")
        return EngineTurnResult(text="outer-done")


class BlockingSyncTool:
    def __init__(self) -> None:
        self.entered: dict[str, threading.Event] = {}
        self.release: dict[str, threading.Event] = {}
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def entered_event(self, label: str) -> threading.Event:
        return self.entered.setdefault(label, threading.Event())

    def release_event(self, label: str) -> threading.Event:
        return self.release.setdefault(label, threading.Event())

    def __call__(self, label: str) -> str:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered_event(label).set()
        try:
            self.release_event(label).wait()
            return f"released:{label}"
        finally:
            with self._guard:
                self.active -= 1


def make_runner(store: Any, function: Any, name: str = "blocking") -> tuple[Runner, CallingEngine]:
    agent = Agent(name="Bot", instructions="Test", tools=[Tool(function, name=name)])
    engine = CallingEngine(name)
    return Runner(agent, session_store=store, engine=engine), engine


def make_one_tool_runner(store: Any, function: Any, name: str) -> tuple[Runner, OneToolEngine]:
    agent = Agent(name="Bot", instructions="Test", tools=[Tool(function, name=name)])
    engine = OneToolEngine(name)
    return Runner(agent, session_store=store, engine=engine), engine


async def assert_sync_work_clean(store: object, session_id: str) -> None:
    await bounded(_wait_for_sync_work(store, session_id))
    for _ in range(100):
        if (id(store), session_id) not in _registry._entries and not _tasks:
            return
        await asyncio.sleep(0)
    assert (id(store), session_id) not in _registry._entries
    assert not _tasks


async def run_with_queue_deadline(runner: Runner, session_id: str, text: str, wait: float) -> Any:
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() + wait, None))
    try:
        return await runner.run_async(session_id, text)
    finally:
        _dispatch_limits.reset(token)


@pytest.mark.asyncio
async def test_sync_timeout_fences_same_session_until_callable_finishes(any_store: Any) -> None:
    tool = BlockingSyncTool()
    runner, engine = make_runner(any_store, tool)

    first = asyncio.create_task(runner.run_async("s", "first", turn_timeout=1.0))
    await wait_thread_event(tool.entered_event("first"))
    with pytest.raises(_TurnTimeout):
        await bounded(first)
    assert tool.active == 1

    second = asyncio.create_task(runner.run_async("s", "second"))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not engine.started_event("second").is_set()
    assert not tool.entered_event("second").is_set()

    tool.release_event("first").set()
    await wait_thread_event(tool.entered_event("second"))
    assert tool.max_active == 1
    tool.release_event("second").set()
    assert (await bounded(second)).content == "done"
    assert tool.active == 0


@pytest.mark.asyncio
async def test_queue_deadline_includes_residual_sync_work(any_store: Any) -> None:
    tool = BlockingSyncTool()
    runner, engine = make_runner(any_store, tool)

    first = asyncio.create_task(runner.run_async("s", "first", turn_timeout=1.0))
    await wait_thread_event(tool.entered_event("first"))
    with pytest.raises(_TurnTimeout):
        await bounded(first)

    with pytest.raises(_SessionWaitTimeout):
        await bounded(run_with_queue_deadline(runner, "s", "queued", 0.1))
    assert not engine.started_event("queued").is_set()
    assert any_store.get("s").history == []

    tool.release_event("first").set()
    await bounded(_wait_for_sync_work(any_store, "s"))
    tool.release_event("next").set()
    assert (await bounded(runner.run_async("s", "next"))).content == "done"


@pytest.mark.asyncio
async def test_residual_sync_work_does_not_block_another_session(any_store: Any) -> None:
    first_entered = threading.Event()
    first_release = threading.Event()

    def selective(label: str) -> str:
        if label == "a:first":
            first_entered.set()
            first_release.wait()
        return label

    runner, _ = make_runner(any_store, selective, name="selective")
    first = asyncio.create_task(runner.run_async("a", "a:first", turn_timeout=1.0))
    await wait_thread_event(first_entered)
    with pytest.raises(_TurnTimeout):
        await bounded(first)

    assert (await bounded(runner.run_async("b", "b:other"))).content == "done"
    first_release.set()
    await bounded(_wait_for_sync_work(any_store, "a"))


@pytest.mark.asyncio
async def test_async_tool_cancellation_is_unchanged(any_store: Any) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def async_blocker(label: str) -> str:
        if label == "first":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return label

    runner, _ = make_runner(any_store, async_blocker, name="async_blocker")
    first = asyncio.create_task(runner.run_async("s", "first", turn_timeout=1.0))
    await bounded(entered.wait())
    with pytest.raises(_TurnTimeout):
        await bounded(first)
    await bounded(cancelled.wait())
    assert (await bounded(runner.run_async("s", "next"))).content == "done"


@pytest.mark.asyncio
async def test_sync_exception_before_timeout_does_not_leave_a_barrier(any_store: Any) -> None:
    def fail(label: str) -> str:
        raise ValueError(f"failed:{label}")

    runner, engine = make_runner(any_store, fail, name="fail")
    assert (await bounded(runner.run_async("s", "first", turn_timeout=2.0))).content == "done"
    assert "Tool execution failed" in engine.results[0]
    assert (await bounded(runner.run_async("s", "second"))).content == "done"


@pytest.mark.asyncio
async def test_sync_exception_after_timeout_opens_barrier_without_task_warning(
    any_store: Any,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    reports: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda unused_loop, context: reports.append(context))

    def fail_late(label: str) -> str:
        if label == "first":
            entered.set()
            release.wait()
            raise ValueError("late failure")
        return label

    try:
        runner, _ = make_runner(any_store, fail_late, name="fail_late")
        first = asyncio.create_task(runner.run_async("s", "first", turn_timeout=1.0))
        await wait_thread_event(entered)
        with pytest.raises(_TurnTimeout):
            await bounded(first)

        second = asyncio.create_task(runner.run_async("s", "second"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not second.done()
        release.set()
        assert (await bounded(second)).content == "done"
        await asyncio.sleep(0)
        assert not [
            report
            for report in reports
            if "exception was never retrieved" in report.get("message", "").lower()
        ]
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_cancellation_before_executor_worker_starts_withdraws_call() -> None:
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    blocker_started = threading.Event()
    blocker_release = threading.Event()
    called = threading.Event()

    def occupy_worker() -> None:
        blocker_started.set()
        blocker_release.wait()

    def should_not_run() -> str:
        called.set()
        return "ran"

    blocker = loop.run_in_executor(None, occupy_worker)
    while not blocker_started.is_set():
        await asyncio.sleep(0)
    try:
        with _track_sync_work(object_store := object(), "s"):
            pending = asyncio.create_task(_run_sync_tool(should_not_run, {}))
            # First pass creates the to_thread task; the second submits it behind blocker.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await bounded(_wait_for_sync_work(object_store, "s"))
    finally:
        blocker_release.set()
        await bounded(blocker)
    assert not called.is_set()


@pytest.mark.asyncio
async def test_blocked_sync_tool_keeps_only_its_session_fenced(any_store: Any) -> None:
    entered = threading.Event()
    release = threading.Event()

    def stays_blocked(label: str) -> str:
        if label == "blocked:first":
            entered.set()
            release.wait()
        return label

    runner, engine = make_runner(any_store, stays_blocked, name="stays_blocked")
    first = asyncio.create_task(runner.run_async("blocked", "blocked:first", turn_timeout=1.0))
    await wait_thread_event(entered)
    with pytest.raises(_TurnTimeout):
        await bounded(first)

    heartbeat = False

    async def beat() -> None:
        nonlocal heartbeat
        await asyncio.sleep(0)
        heartbeat = True

    try:
        await bounded(beat())
        assert heartbeat
        with pytest.raises(_SessionWaitTimeout):
            await bounded(run_with_queue_deadline(runner, "blocked", "blocked:queued", 0.1))
        assert not engine.started_event("blocked:queued").is_set()
        assert (await bounded(runner.run_async("other", "other:turn"))).content == "done"
    finally:
        release.set()
        await bounded(_wait_for_sync_work(any_store, "blocked"))


def test_residual_barrier_is_visible_from_a_different_event_loop() -> None:
    store = SessionStore()
    entered = threading.Event()
    release = threading.Event()
    timed_out = threading.Event()
    first_errors: list[BaseException] = []

    def selective(label: str) -> str:
        if label == "first":
            entered.set()
            release.wait()
        return label

    runner, engine = make_runner(store, selective, name="selective")

    async def first_loop() -> None:
        try:
            await runner.run_async("s", "first", turn_timeout=0.2)
        except _TurnTimeout:
            timed_out.set()
        except BaseException as error:
            first_errors.append(error)

    thread = threading.Thread(target=lambda: asyncio.run(first_loop()))
    thread.start()
    assert entered.wait(SAFETY)
    assert timed_out.wait(SAFETY)

    try:
        with pytest.raises(_SessionWaitTimeout):
            asyncio.run(run_with_queue_deadline(runner, "s", "queued", 0.1))
        assert not engine.started_event("queued").is_set()
    finally:
        release.set()
        thread.join(SAFETY)

    assert not thread.is_alive()
    assert first_errors == []
    assert asyncio.run(runner.run_async("s", "next")).content == "done"


@pytest.mark.asyncio
async def test_residual_sync_tool_rejects_same_session_after_turn_timeout() -> None:
    store = SessionStore()
    entered = threading.Event()
    reenter = threading.Event()
    finished = threading.Event()
    outcomes: list[type[BaseException] | str] = []
    holder: dict[str, Runner] = {}

    def reenter_same_session() -> str:
        entered.set()
        reenter.wait()
        try:
            holder["runner"].run("s", "nested")
        except BaseException as error:
            outcomes.append(type(error))
        else:
            outcomes.append("nested-ran")
        finally:
            finished.set()
        return "tool-finished"

    runner, engine = make_one_tool_runner(store, reenter_same_session, "reenter_same_session")
    holder["runner"] = runner
    outer = asyncio.create_task(runner.run_async("s", "outer", turn_timeout=0.2))
    await wait_thread_event(entered)
    with pytest.raises(_TurnTimeout):
        await bounded(outer)

    reenter.set()
    await wait_thread_event(finished)
    assert outcomes == [ReentrantTurnError]
    assert engine.user_messages == ["outer"]
    await assert_sync_work_clean(store, "s")


@pytest.mark.asyncio
async def test_residual_sync_tool_can_run_a_different_session_after_timeout() -> None:
    store = SessionStore()
    entered = threading.Event()
    reenter = threading.Event()
    finished = threading.Event()
    outcomes: list[str] = []
    holder: dict[str, Runner] = {}

    def run_other_session() -> str:
        entered.set()
        reenter.wait()
        try:
            response = holder["runner"].run("other", "nested")
            outcomes.append(response.content)
        finally:
            finished.set()
        return "tool-finished"

    runner, engine = make_one_tool_runner(store, run_other_session, "run_other_session")
    holder["runner"] = runner
    outer = asyncio.create_task(runner.run_async("s", "outer", turn_timeout=0.2))
    await wait_thread_event(entered)
    with pytest.raises(_TurnTimeout):
        await bounded(outer)

    reenter.set()
    await wait_thread_event(finished)
    assert outcomes == ["reply:nested"]
    assert engine.user_messages == ["outer", "nested"]
    await assert_sync_work_clean(store, "s")


@pytest.mark.asyncio
async def test_residual_sync_tool_rejects_same_session_after_external_cancellation() -> None:
    store = SessionStore()
    entered = threading.Event()
    reenter = threading.Event()
    finished = threading.Event()
    outcomes: list[type[BaseException] | str] = []
    holder: dict[str, Runner] = {}

    def reenter_after_cancellation() -> str:
        entered.set()
        reenter.wait()
        try:
            holder["runner"].run("s", "nested")
        except BaseException as error:
            outcomes.append(type(error))
        else:
            outcomes.append("nested-ran")
        finally:
            finished.set()
        return "tool-finished"

    runner, engine = make_one_tool_runner(
        store, reenter_after_cancellation, "reenter_after_cancellation"
    )
    holder["runner"] = runner
    outer = asyncio.create_task(runner.run_async("s", "outer"))
    await wait_thread_event(entered)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    reenter.set()
    await wait_thread_event(finished)
    assert outcomes == [ReentrantTurnError]
    assert engine.user_messages == ["outer"]
    await assert_sync_work_clean(store, "s")


@pytest.mark.asyncio
async def test_sync_tool_reentrancy_is_still_rejected_while_turn_is_active() -> None:
    store = SessionStore()
    outcomes: list[type[BaseException] | str] = []
    holder: dict[str, Runner] = {}

    def reenter_immediately() -> str:
        try:
            holder["runner"].run("s", "nested")
        except BaseException as error:
            outcomes.append(type(error))
        else:
            outcomes.append("nested-ran")
        return "tool-finished"

    runner, engine = make_one_tool_runner(store, reenter_immediately, "reenter_immediately")
    holder["runner"] = runner
    assert (await bounded(runner.run_async("s", "outer"))).content == "outer-done"
    assert outcomes == [ReentrantTurnError]
    assert engine.user_messages == ["outer"]
    await assert_sync_work_clean(store, "s")


@pytest.mark.asyncio
async def test_finished_ticket_in_copied_context_does_not_cause_false_reentrancy() -> None:
    store = SessionStore()
    copied: list[contextvars.Context] = []
    holder: dict[str, Runner] = {}

    def copy_running_context() -> str:
        copied.append(contextvars.copy_context())
        return "copied"

    runner, engine = make_one_tool_runner(store, copy_running_context, "copy_running_context")
    holder["runner"] = runner
    assert (await bounded(runner.run_async("s", "outer"))).content == "outer-done"
    await assert_sync_work_clean(store, "s")

    response = await bounded(asyncio.to_thread(copied[0].run, holder["runner"].run, "s", "later"))
    assert response.content == "reply:later"
    assert engine.user_messages == ["outer", "later"]
