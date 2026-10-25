"""Runner.run_async(turn_timeout=...) and the session-wait deadline channels set (finding F).

The turn limit starts once the turn holds the session lock, so waiting behind earlier
turns of the same session never consumes it. Tests are deterministic: turns hold the
lock on asyncio.Event gates released by the test; sleeps only ever establish a lower
bound on how long a turn waited.
"""

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import (
    Runner,
    _dispatch_limits,
    _DispatchLimits,
    _hold_session_lock,
    _SessionWaitTimeout,
    _TurnTimeout,
    _within,
)
from chatflow_agent.exceptions import ReentrantTurnError
from chatflow_agent.types import Message, Role, ToolCall

SAFETY = 10.0  # bound for waits that must finish; never reached when the code is correct


@pytest.fixture(params=["sqlite_file", "sqlite_memory", "memory"])
def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "sqlite_file":
        return SQLiteSessionStore(tmp_path / "s.db")
    if request.param == "sqlite_memory":
        return SQLiteSessionStore(":memory:")
    return SessionStore()


class GateEngine(BaseEngine):
    """Replies ``re:<text>``; texts listed in ``gates`` wait for their event first."""

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.fail: set[str] = set()
        self.started: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []
        self.cancelled: list[str] = []

    def started_event(self, text: str) -> asyncio.Event:
        return self.started.setdefault(text, asyncio.Event())

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        text = next(m.content for m in reversed(history) if m.role == Role.USER) or ""
        self.calls.append(text)
        self.started_event(text).set()
        try:
            gate = self.gates.get(text)
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        if text in self.fail:
            raise RuntimeError("engine failure")
        return EngineTurnResult(text=f"re:{text}")


def history(store: Any, session_id: str = "s") -> list[str | None]:
    session = store.get(session_id)
    return [] if session is None else [m.content for m in session.history]


async def bounded(aw: Any) -> Any:
    return await asyncio.wait_for(aw, SAFETY)


def make_runner(store: Any, tools: list[Any] | None = None, engine: BaseEngine | None = None):
    engine = engine or GateEngine()
    agent = Agent(name="Bot", instructions="Test", tools=tools or [])
    return Runner(agent, session_store=store, engine=engine), engine


# ---------------------------------------------------------------- public API


def test_turn_timeout_is_keyword_only_and_defaults_to_none() -> None:
    param = inspect.signature(Runner.run_async).parameters["turn_timeout"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
async def test_turn_timeout_rejects_non_positive_values_before_touching_the_session(
    any_store: Any, value: float
) -> None:
    runner, engine = make_runner(any_store)
    with pytest.raises(ValueError, match="turn_timeout"):
        await runner.run_async("s", "hello", turn_timeout=value)
    assert any_store.get("s") is None
    assert engine.calls == []


@pytest.mark.asyncio
async def test_turn_timeout_none_keeps_previous_behaviour(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["slow"] = asyncio.Event()
    task = asyncio.create_task(runner.run_async("s", "slow"))
    await bounded(engine.started_event("slow").wait())
    await asyncio.sleep(0.05)
    assert not task.done()
    engine.gates["slow"].set()
    assert (await bounded(task)).content == "re:slow"


# ---------------------------------------------------------------- turn limit


@pytest.mark.asyncio
async def test_turn_timeout_cancels_the_turn_rolls_back_and_releases_the_session(
    any_store: Any,
) -> None:
    runner, engine = make_runner(any_store)
    await runner.run_async("s", "first")
    before = history(any_store)

    engine.gates["stuck"] = asyncio.Event()  # never released
    # 1.0 s covers the turn's disk writes before its engine call (file store, slow CI disks):
    # the limit must expire while the engine runs.
    turn = asyncio.ensure_future(runner.run_async("s", "stuck", turn_timeout=1.0))
    try:
        await asyncio.wait_for(engine.started_event("stuck").wait(), timeout=5.0)
    except asyncio.TimeoutError:
        turn.cancel()
        await asyncio.wait_for(asyncio.gather(turn, return_exceptions=True), timeout=SAFETY)
        pytest.fail("the turn limit expired before the engine started")
    with pytest.raises(asyncio.TimeoutError):
        await bounded(turn)
    assert turn.done()  # the timed-out turn has finished, rollback included

    assert engine.cancelled == ["stuck"]
    assert history(any_store) == before
    assert not any_store.get("s").lock.locked()
    assert (await bounded(runner.run_async("s", "next"))).content == "re:next"
    assert history(any_store) == before + ["next", "re:next"]


@pytest.mark.asyncio
async def test_waiting_for_the_session_does_not_count_toward_turn_timeout(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn("s", "hold")
    await bounded(engine.started_event("hold").wait())

    # The turn limit must cover the queued turn's own work, which with a file store
    # includes real disk writes (slow on CI runners); the wait below still exceeds it.
    queued = asyncio.create_task(runner.run_async("s", "queued", turn_timeout=1.0))
    await asyncio.sleep(1.2)  # queued has waited longer than its turn limit
    assert not queued.done()
    assert "queued" not in engine.calls

    engine.gates["hold"].set()
    assert (await bounded(queued)).content == "re:queued"
    await bounded(holder)
    assert history(any_store) == ["hold", "re:hold", "queued", "re:queued"]


@pytest.mark.asyncio
async def test_external_cancellation_under_turn_timeout_stays_a_cancellation(
    any_store: Any,
) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["stuck"] = asyncio.Event()
    task = asyncio.create_task(runner.run_async("s", "stuck", turn_timeout=5.0))
    await bounded(engine.started_event("stuck").wait())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert history(any_store) == []
    assert not any_store.get("s").lock.locked()
    assert (await bounded(runner.run_async("s", "next"))).content == "re:next"


@pytest.mark.asyncio
async def test_error_under_turn_timeout_propagates_restores_and_releases(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    engine.fail.add("boom")
    with pytest.raises(RuntimeError, match="engine failure"):
        await bounded(runner.run_async("s", "boom", turn_timeout=5.0))
    assert history(any_store) == []
    assert not any_store.get("s").lock.locked()
    assert (await bounded(runner.run_async("s", "next", turn_timeout=5.0))).content == "re:next"


# ---------------------------------------------------------------- reentrancy and spawn_turn


class ToolCallingEngine(BaseEngine):
    """The first model call of a turn calls ``tool_name``; the next one reports the result."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.tool_results: list[Any] = []

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        last = history[-1]
        if last.role == Role.USER and last.content == "outer":
            return EngineTurnResult(tool_calls=[ToolCall(id="c1", name=self.tool_name, args={})])
        if last.role == Role.TOOL:
            assert last.tool_results is not None
            self.tool_results.extend(r.content for r in last.tool_results)
            return EngineTurnResult(text="done")
        return EngineTurnResult(text=f"re:{last.content}")


@pytest.mark.asyncio
async def test_reentrant_turn_is_still_refused_under_turn_timeout(any_store: Any) -> None:
    """The turn runs inside the timeout scope (a separate task on Python 3.10); the turn
    lineage must still reach the tool."""
    holder: dict[str, Runner] = {}

    async def ask_again() -> str:
        """Run another turn for the same session."""
        try:
            await holder["runner"].run_async("s", "nested", turn_timeout=5.0)
        except ReentrantTurnError as err:
            return f"refused: {err}"
        return "nested turn ran"

    engine = ToolCallingEngine("ask_again")
    holder["runner"], _ = make_runner(any_store, tools=[ask_again], engine=engine)
    response = await bounded(holder["runner"].run_async("s", "outer", turn_timeout=5.0))
    assert response.content == "done"
    assert len(engine.tool_results) == 1
    assert engine.tool_results[0].startswith("refused:")


@pytest.mark.asyncio
async def test_spawn_turn_from_a_tool_runs_after_a_turn_with_turn_timeout(any_store: Any) -> None:
    holder: dict[str, Runner] = {}
    spawned: list[asyncio.Task[Any]] = []

    async def follow_up() -> str:
        """Queue a follow-up turn for the same session."""
        spawned.append(holder["runner"].spawn_turn("s", "follow-up"))
        return "scheduled"

    engine = ToolCallingEngine("follow_up")
    holder["runner"], _ = make_runner(any_store, tools=[follow_up], engine=engine)
    assert (await bounded(holder["runner"].run_async("s", "outer", turn_timeout=5.0))).content == "done"
    assert (await bounded(spawned[0])).content == "re:follow-up"
    users = [m.content for m in any_store.get("s").history if m.role == Role.USER]
    assert users == ["outer", "follow-up"]


# ---------------------------------------------------------------- session-wait deadline (channels)


async def run_with_limits(runner: Runner, text: str, wait: float, turn: float | None = None):
    """Dispatch like a channel does: limits in the context of the run_async() call."""
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() + wait, turn))
    try:
        return await runner.run_async("s", text)
    finally:
        _dispatch_limits.reset(token)


@pytest.mark.asyncio
async def test_wait_deadline_expires_and_the_turn_never_runs(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn("s", "hold")
    await bounded(engine.started_event("hold").wait())

    with pytest.raises(_SessionWaitTimeout):
        await bounded(run_with_limits(runner, "late", wait=0.1))
    assert issubclass(_SessionWaitTimeout, asyncio.TimeoutError)

    engine.gates["hold"].set()
    await bounded(holder)
    for _ in range(10):
        await asyncio.sleep(0)
    assert "late" not in engine.calls  # an expired turn is never run afterwards
    assert history(any_store) == ["hold", "re:hold"]
    assert not any_store.get("s").lock.locked()
    assert (await bounded(run_with_limits(runner, "next", wait=5.0))).content == "re:next"


@pytest.mark.asyncio
async def test_wait_deadline_does_not_limit_the_turn(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["slow"] = asyncio.Event()
    task = asyncio.create_task(run_with_limits(runner, "slow", wait=0.5, turn=None))
    await bounded(engine.started_event("slow").wait())
    await asyncio.sleep(0.6)  # past the wait deadline, but the turn had already started
    assert not task.done()
    engine.gates["slow"].set()
    assert (await bounded(task)).content == "re:slow"


@pytest.mark.asyncio
async def test_cancelling_a_turn_waiting_with_a_deadline_does_not_leak_the_lock(
    any_store: Any,
) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn("s", "hold")
    await bounded(engine.started_event("hold").wait())
    waiter = asyncio.create_task(run_with_limits(runner, "waiting", wait=5.0))
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    engine.gates["hold"].set()
    await bounded(holder)
    assert not any_store.get("s").lock.locked()
    assert (await bounded(runner.run_async("s", "next"))).content == "re:next"
    assert "waiting" not in engine.calls


@pytest.mark.asyncio
async def test_lock_obtained_after_the_wait_deadline_is_released_at_once() -> None:
    """The loop can be busy (e.g. synchronous I/O) when the lock is handed over, so the
    turn resumes after its deadline. It must give the lock back instead of running."""

    class SlowAcquireLock(asyncio.Lock):
        async def acquire(self) -> bool:
            time.sleep(0.1)  # blocks the event loop, as synchronous I/O would
            return await super().acquire()

    lock = SlowAcquireLock()
    deadline = asyncio.get_running_loop().time() + 0.05
    ran = False
    with pytest.raises(_SessionWaitTimeout):
        async with _hold_session_lock(object(), "s", lock, wait_deadline=deadline):
            ran = True
    assert not ran
    assert not lock.locked()


@pytest.mark.asyncio
async def test_dispatch_limits_are_consumed_once_and_never_reach_tools(any_store: Any) -> None:
    seen_in_tool: list[Any] = []

    async def peek() -> str:
        """Report the limits visible to tool code."""
        seen_in_tool.append(_dispatch_limits.get())
        return "ok"

    engine = ToolCallingEngine("peek")
    runner, _ = make_runner(any_store, tools=[peek], engine=engine)
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() + 5.0, 5.0))
    try:
        # Awaited directly (no wait_for): on Python 3.10/3.11 wait_for() would run the call
        # in a child task, and the consumption would happen in that task's context copy.
        assert (await runner.run_async("s", "outer")).content == "done"
        assert _dispatch_limits.get() is None  # consumed by the first run_async()
    finally:
        _dispatch_limits.reset(token)
    assert seen_in_tool == [None]


@pytest.mark.asyncio
async def test_spawn_turn_does_not_inherit_dispatch_limits(any_store: Any) -> None:
    runner, _ = make_runner(any_store)
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() - 1.0, 0.001))  # already expired
    try:
        task = runner.spawn_turn("s", "spawned")
    finally:
        _dispatch_limits.reset(token)
    assert (await bounded(task)).content == "re:spawned"


# ---------------------------------------------------------------- _within() and timeout kinds


async def _sleep_then(value: str, delay: float = 0.05) -> str:
    await asyncio.sleep(delay)
    return value


async def _raise_timeout() -> str:
    raise asyncio.TimeoutError("raised by the awaited code")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, float("inf"), float("nan")])
async def test_within_sets_no_limit_like_wait_for(value: float | None) -> None:
    """Like wait_for(); asyncio.timeout(nan) would expire at once on Python 3.11+."""
    assert await bounded(_within(value, _sleep_then("done"), _TurnTimeout)) == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, float("-inf")])
async def test_within_expires_at_once_for_non_positive_values(value: float) -> None:
    with pytest.raises(_TurnTimeout):
        await bounded(_within(value, _sleep_then("never"), _TurnTimeout))


@pytest.mark.asyncio
async def test_within_raises_its_own_kind_only_when_its_limit_expires() -> None:
    with pytest.raises(_TurnTimeout):
        await bounded(_within(0.05, _sleep_then("late", delay=5.0), _TurnTimeout))
    with pytest.raises(asyncio.TimeoutError) as raised:
        await bounded(_within(5.0, _raise_timeout(), _TurnTimeout))
    assert not isinstance(raised.value, _TurnTimeout)
    assert "raised by the awaited code" in str(raised.value)


@pytest.mark.asyncio
async def test_within_classifies_its_limit_even_when_the_loop_fires_timers_early() -> None:
    """The event loop may run a timer up to one clock tick before it is due (about 16 ms on
    Windows). Classifying by elapsed time misreported such expiries as foreign
    TimeoutErrors on Python 3.10; this loop fires every new timer at half its delay."""
    loop = asyncio.get_running_loop()
    call_later, call_at = loop.call_later, loop.call_at

    def early_call_later(delay: float, callback: Any, *args: Any, **kwargs: Any) -> Any:
        return call_later(delay / 2, callback, *args, **kwargs)

    def early_call_at(when: float, callback: Any, *args: Any, **kwargs: Any) -> Any:
        now = loop.time()
        return call_at(now + (when - now) / 2, callback, *args, **kwargs)

    loop.call_later = early_call_later  # type: ignore[method-assign]
    loop.call_at = early_call_at  # type: ignore[method-assign]
    try:
        with pytest.raises(_TurnTimeout):
            await _within(0.2, asyncio.Event().wait(), _TurnTimeout)
    finally:
        del loop.call_later, loop.call_at  # back to the class methods
    assert loop.call_later == call_later


@pytest.mark.asyncio
async def test_turn_timeout_raises_a_timeout_error_of_the_turn_kind(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    engine.gates["stuck"] = asyncio.Event()
    with pytest.raises(_TurnTimeout) as raised:
        await bounded(runner.run_async("s", "stuck", turn_timeout=0.1))
    assert isinstance(raised.value, asyncio.TimeoutError)  # the documented contract


@pytest.mark.asyncio
async def test_timeout_error_from_the_engine_is_not_reported_as_turn_timeout(
    any_store: Any,
) -> None:
    class TimeoutEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            raise asyncio.TimeoutError("provider timeout")

    runner, _ = make_runner(any_store, engine=TimeoutEngine())
    with pytest.raises(asyncio.TimeoutError) as raised:
        await bounded(runner.run_async("s", "hi", turn_timeout=5.0))
    assert not isinstance(raised.value, _TurnTimeout)
    assert history(any_store) == []  # still rolled back


# ---------------------------------------------------------------- _dispatch_limits contract


@pytest.mark.asyncio
async def test_dispatch_limits_are_consumed_even_when_the_call_is_rejected(any_store: Any) -> None:
    runner, _ = make_runner(any_store)
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() + 5.0, 5.0))
    try:
        with pytest.raises(ValueError):
            await runner.run_async("s", "hi", turn_timeout=0)
        assert _dispatch_limits.get() is None
    finally:
        _dispatch_limits.reset(token)


@pytest.mark.asyncio
async def test_concurrent_dispatches_keep_their_own_limits(any_store: Any) -> None:
    runner, engine = make_runner(any_store)
    gates = {}
    for sid in ("a", "b"):
        gates[sid] = engine.gates[f"hold-{sid}"] = asyncio.Event()
    holders = [runner.spawn_turn(sid, f"hold-{sid}") for sid in ("a", "b")]
    for sid in ("a", "b"):
        await bounded(engine.started_event(f"hold-{sid}").wait())

    async def dispatch(sid: str, wait: float) -> Any:
        loop = asyncio.get_running_loop()
        token = _dispatch_limits.set(_DispatchLimits(loop.time() + wait, None))
        try:
            return await runner.run_async(sid, f"queued-{sid}")
        finally:
            _dispatch_limits.reset(token)

    short = asyncio.create_task(dispatch("a", 0.2))
    long = asyncio.create_task(dispatch("b", 5.0))
    with pytest.raises(_SessionWaitTimeout):
        await bounded(short)
    assert not long.done()  # b's limit is its own, not a's
    for sid in ("a", "b"):
        gates[sid].set()
    assert (await bounded(long)).content == "re:queued-b"
    await bounded(asyncio.gather(*holders))
    assert "queued-a" not in engine.calls


@pytest.mark.asyncio
async def test_task_created_before_run_async_gets_its_own_copy_of_the_limits(any_store: Any) -> None:
    """Contract: a task created while the limits are set (e.g. by a dispatch_async()
    override) starts with them, and consumes them in that task only."""
    runner, engine = make_runner(any_store)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn("s", "hold")
    await bounded(engine.started_event("hold").wait())
    loop = asyncio.get_running_loop()
    token = _dispatch_limits.set(_DispatchLimits(loop.time() + 0.2, None))
    try:
        child = asyncio.create_task(runner.run_async("s", "via-child"))
        with pytest.raises(_SessionWaitTimeout):
            await bounded(child)
        assert _dispatch_limits.get() is not None  # the creator's copy is untouched
    finally:
        _dispatch_limits.reset(token)
    assert _dispatch_limits.get() is None
    engine.gates["hold"].set()
    await bounded(holder)
