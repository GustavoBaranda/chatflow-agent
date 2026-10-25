"""WhatsApp queue limit vs. turn limit (finding F).

``queue_timeout_seconds`` bounds the wait behind earlier turns of the same session;
``llm_timeout_seconds`` bounds the turn once it holds the session. Before the fix a single
``llm_timeout_seconds`` covered both, so valid messages queued behind slow turns timed out
without running. Concurrent ``_process_message_background()`` calls stand in for the
background tasks of separate webhook requests. Turns hold the session on asyncio.Event
gates released by the test; sleeps only establish lower bounds on waiting time.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner, _dispatch_limits
from chatflow_agent.types import Message, Role, ToolCall

SAFETY = 10.0
FALLBACK = "fallback-reply"
PHONE = "5491100000001"
MASKED = "5491***001"


@pytest.fixture(params=["sqlite_file", "sqlite_memory", "memory"])
def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "sqlite_file":
        return SQLiteSessionStore(tmp_path / "wa.db")
    if request.param == "sqlite_memory":
        return SQLiteSessionStore(":memory:")
    return SessionStore()


class GateEngine(BaseEngine):
    """Replies ``re:<text>``; gated texts wait for their event; tracks per-session overlap."""

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.started: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.active = 0
        self.max_active = 0

    def started_event(self, text: str) -> asyncio.Event:
        return self.started.setdefault(text, asyncio.Event())

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        text = next(m.content for m in reversed(history) if m.role == Role.USER) or ""
        self.calls.append(text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started_event(text).set()
        try:
            gate = self.gates.get(text)
            if gate is not None:
                await gate.wait()
            return EngineTurnResult(text=f"re:{text}")
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        finally:
            self.active -= 1


def build(
    store: Any,
    *,
    queue: float,
    turn: float,
    engine: BaseEngine | None = None,
    tools: list[Any] | None = None,
) -> tuple[WhatsAppChannel, Runner, Any, list[tuple[str, str]]]:
    engine = engine or GateEngine()
    runner = Runner(
        Agent(name="Bot", instructions="Test", tools=tools or []),
        session_store=store,
        engine=engine,
    )
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test-token",
        phone_number_id="12345",
        verify_signature=False,
        fallback_message=FALLBACK,
        llm_timeout_seconds=turn,
        queue_timeout_seconds=queue,
    )
    channel.attach(runner)
    sent: list[tuple[str, str]] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append((to_phone, text))

    channel._send_outbound_whatsapp = capture  # type: ignore[method-assign]
    return channel, runner, engine, sent


def process(channel: WhatsAppChannel, wamid: str, text: str, phone: str = PHONE) -> "asyncio.Task[None]":
    return asyncio.create_task(
        channel._process_message_background(wamid=wamid, sender_phone=phone, user_text=text)
    )


def history(store: Any, session_id: str = PHONE) -> list[str | None]:
    session = store.get(session_id)
    return [] if session is None else [m.content for m in session.history]


async def bounded(aw: Any) -> Any:
    return await asyncio.wait_for(aw, SAFETY)


# ---------------------------------------------------------------- configuration


def test_queue_timeout_defaults_to_120_seconds() -> None:
    channel = WhatsAppChannel(verify_token="tok", verify_signature=False)
    assert channel.queue_timeout_seconds == 120.0
    assert channel.llm_timeout_seconds == 120.0


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_queue_timeout_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="queue_timeout_seconds"):
        WhatsAppChannel(verify_token="tok", verify_signature=False, queue_timeout_seconds=value)


# ---------------------------------------------------------------- the defect (audit case 1)


@pytest.mark.asyncio
async def test_waiting_behind_a_turn_does_not_consume_llm_timeout(any_store: Any) -> None:
    """Audit case 1a: a message that needs ~0 s used to time out while queued."""
    # The turn limit must cover the queued turn's own work, which with a file store
    # includes real disk writes (slow on CI runners); the wait below still exceeds it.
    channel, runner, engine, sent = build(any_store, queue=5.0, turn=1.0)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn(PHONE, "hold")  # a turn of the same session, not bounded
    await bounded(engine.started_event("hold").wait())

    task = process(channel, "w2", "queued")
    await asyncio.sleep(1.2)  # queued has waited longer than llm_timeout_seconds
    assert not task.done()
    engine.gates["hold"].set()
    await bounded(asyncio.gather(holder, task))

    assert sent == [(PHONE, "re:queued")]
    assert history(any_store) == ["hold", "re:hold", "queued", "re:queued"]


@pytest.mark.asyncio
async def test_queue_timeout_answers_with_fallback_and_never_runs_the_turn(
    any_store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    channel, runner, engine, sent = build(any_store, queue=0.3, turn=5.0)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn(PHONE, "hold")
    await bounded(engine.started_event("hold").wait())

    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        await bounded(process(channel, "w2", "late"))
    assert sent == [(PHONE, FALLBACK)]
    assert "Session busy" in caplog.text
    assert "queue_timeout_seconds" in caplog.text
    assert "Turn timed out" not in caplog.text
    assert MASKED in caplog.text and PHONE not in caplog.text

    engine.gates["hold"].set()
    await bounded(holder)
    for _ in range(10):
        await asyncio.sleep(0)
    assert "late" not in engine.calls  # never runs, not even after the session is free
    assert history(any_store) == ["hold", "re:hold"]
    assert not any_store.get(PHONE).lock.locked()

    await bounded(process(channel, "w3", "next"))
    assert sent[-1] == (PHONE, "re:next")


@pytest.mark.asyncio
async def test_turn_timeout_is_logged_separately_and_rolls_back(
    any_store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # 1.0 s covers the turn's disk writes before its engine call (file store, slow CI disks):
    # the limit must expire while the engine runs.
    channel, runner, engine, sent = build(any_store, queue=5.0, turn=1.0)
    engine.gates["stuck"] = asyncio.Event()  # never released
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        task = process(channel, "w1", "stuck")
        try:
            await asyncio.wait_for(engine.started_event("stuck").wait(), timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=SAFETY)
            pytest.fail("the turn limit expired before the engine started")
        await bounded(task)
    assert task.done()  # the timed-out turn has finished, rollback included
    assert sent == [(PHONE, FALLBACK)]
    assert "Turn timed out" in caplog.text
    assert "llm_timeout_seconds" in caplog.text
    assert "Session busy" not in caplog.text
    assert MASKED in caplog.text and PHONE not in caplog.text
    assert engine.cancelled == ["stuck"]
    assert history(any_store) == []
    assert not any_store.get(PHONE).lock.locked()

    # The follow-up turn must succeed and does real disk writes with a file store (slow on
    # CI runners): give it its own realistic limit, independent of the one "stuck" exceeded.
    channel.llm_timeout_seconds = 5.0
    await bounded(process(channel, "w2", "next"))
    assert sent[-1] == (PHONE, "re:next")


# ---------------------------------------------------------------- burst (audit case 2)


@pytest.mark.asyncio
async def test_burst_longer_than_llm_timeout_is_fully_answered_in_order(any_store: Any) -> None:
    """Ten concurrent messages queue behind a running turn for longer than
    llm_timeout_seconds; each of their turns fits in it. Before the fix most of them timed
    out while queued (audit case 2)."""
    # Each turn must fit in its own limit even on slow CI runners (SQLite on disk), while
    # the queue still lasts longer than that limit: turn=1.0 s, wait before release 1.2 s.
    channel, runner, engine, sent = build(any_store, queue=30.0, turn=1.0)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn(PHONE, "hold")
    await bounded(engine.started_event("hold").wait())
    texts = [f"m{i}" for i in range(10)]
    for text in texts:
        engine.gates[text] = asyncio.Event()
    tasks = [process(channel, f"w{i}", text) for i, text in enumerate(texts)]

    await asyncio.sleep(1.2)  # every message has now waited longer than llm_timeout_seconds
    engine.gates["hold"].set()
    await bounded(holder)
    for text in texts:  # release one turn at a time, as soon as it is the running one
        await bounded(engine.started_event(text).wait())
        assert engine.active == 1
        engine.gates[text].set()
    await bounded(asyncio.gather(*tasks))

    assert engine.calls == ["hold"] + texts  # FIFO
    assert engine.max_active == 1  # one turn of the session at a time
    assert sent == [(PHONE, f"re:{text}") for text in texts]  # one reply each, no fallback
    expected: list[str | None] = ["hold", "re:hold"]
    for text in texts:
        expected += [text, f"re:{text}"]
    assert history(any_store) == expected


@pytest.mark.asyncio
async def test_burst_beyond_queue_timeout_fails_only_the_queued_messages(any_store: Any) -> None:
    channel, runner, engine, sent = build(any_store, queue=0.3, turn=5.0)
    engine.gates["m0"] = asyncio.Event()
    first = process(channel, "w0", "m0")
    await bounded(engine.started_event("m0").wait())
    rest = [process(channel, f"w{i}", f"m{i}") for i in range(1, 5)]
    await bounded(asyncio.gather(*rest))  # all expire while m0 holds the session
    engine.gates["m0"].set()
    await bounded(first)

    assert engine.calls == ["m0"]
    assert sorted(sent) == sorted([(PHONE, FALLBACK)] * 4 + [(PHONE, "re:m0")])
    assert history(any_store) == ["m0", "re:m0"]


# ---------------------------------------------------------------- isolation


@pytest.mark.asyncio
async def test_queue_limit_does_not_reach_turns_started_by_tools(any_store: Any) -> None:
    """A tool of a WhatsApp turn that awaits a turn of another, busy session waits for it;
    the WhatsApp message's queue limit applies to its own turn only."""
    holder: dict[str, Runner] = {}
    tool_started = asyncio.Event()
    limits_seen_by_tool: list[Any] = []
    queue_limit = 0.3

    async def ask_other() -> str:
        """Run a turn for another session."""
        limits_seen_by_tool.append(_dispatch_limits.get())
        tool_started.set()
        response = await holder["runner"].run_async("other", "sub")
        return response.content

    class Engine(BaseEngine):
        def __init__(self) -> None:
            self.gate = asyncio.Event()
            self.other_started = asyncio.Event()

        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            last = history[-1]
            if last.role == Role.USER and last.content == "hold-other":
                self.other_started.set()
                await self.gate.wait()
                return EngineTurnResult(text="held")
            if last.role == Role.USER and last.content == "outer":
                return EngineTurnResult(tool_calls=[ToolCall(id="c1", name="ask_other", args={})])
            if last.role == Role.TOOL:
                assert last.tool_results is not None
                return EngineTurnResult(text=f"tool said {last.tool_results[0].content}")
            return EngineTurnResult(text=f"re:{last.content}")

    engine = Engine()
    channel, runner, _, sent = build(
        any_store, queue=queue_limit, turn=5.0, engine=engine, tools=[ask_other]
    )
    holder["runner"] = runner
    # Keep session creation outside the deliberately short channel deadline. This test is
    # about whether that deadline reaches a tool, not about SQLite cold-start latency.
    any_store.get_or_create(PHONE, runner.starting_agent)
    other = runner.spawn_turn("other", "hold-other")
    await bounded(engine.other_started.wait())

    task = process(channel, "w1", "outer")
    await bounded(tool_started.wait())  # the outer turn holds PHONE and has entered its tool
    assert limits_seen_by_tool == [None]
    # The nested turn is waiting for "other". It must remain active beyond the channel's
    # queue limit; shielding keeps this observation from cancelling it.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=queue_limit + 0.1)
    assert not task.done()
    engine.gate.set()
    await bounded(asyncio.gather(other, task))
    assert sent == [(PHONE, "tool said re:sub")]
    assert all(text != FALLBACK for _, text in sent)


@pytest.mark.asyncio
async def test_dispatch_limits_are_consumed_before_a_tool_runs() -> None:
    """A tool deterministically observes no limits from the channel that started its turn."""
    limits_seen_by_tool: list[Any] = []

    async def inspect_limits() -> str:
        limits_seen_by_tool.append(_dispatch_limits.get())
        return "isolated"

    class Engine(BaseEngine):
        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            if history[-1].role == Role.USER:
                return EngineTurnResult(
                    tool_calls=[ToolCall(id="c1", name="inspect_limits", args={})]
                )
            return EngineTurnResult(text="done")

    channel, _, _, sent = build(
        SessionStore(), queue=5.0, turn=5.0, engine=Engine(), tools=[inspect_limits]
    )
    await bounded(process(channel, "w1", "outer"))

    assert limits_seen_by_tool == [None]
    assert sent == [(PHONE, "done")]


@pytest.mark.asyncio
async def test_dispatch_limits_do_not_outlive_the_dispatch(any_store: Any) -> None:
    channel, _, _, sent = build(any_store, queue=5.0, turn=5.0)
    await channel._process_message_background(wamid="w1", sender_phone=PHONE, user_text="hi")
    assert sent == [(PHONE, "re:hi")]
    assert _dispatch_limits.get() is None


@pytest.mark.asyncio
async def test_dispatch_async_seam_is_unchanged(any_store: Any) -> None:
    """Subclasses and tests intercept dispatch_async(session_id, message, metadata)."""
    channel, _, _, sent = build(any_store, queue=5.0, turn=5.0)
    calls: list[tuple[str, str, Any]] = []
    original = channel.dispatch_async

    async def spy(*, session_id: str, message: str, metadata: Any = None) -> Any:
        calls.append((session_id, message, metadata))
        return await original(session_id=session_id, message=message, metadata=metadata)

    channel.dispatch_async = spy  # type: ignore[method-assign]
    await bounded(process(channel, "w1", "hi"))
    assert calls == [(PHONE, "hi", {"channel": "whatsapp", "phone": PHONE, "wamid": "w1"})]
    assert sent == [(PHONE, "re:hi")]


@pytest.mark.asyncio
async def test_backstop_bounds_a_dispatch_async_that_never_reaches_the_runner() -> None:
    channel, _, _, sent = build(SessionStore(), queue=0.05, turn=0.05)

    async def hang(**_: Any) -> Any:
        await asyncio.Event().wait()

    with patch.object(channel, "dispatch_async", hang):
        await bounded(process(channel, "w1", "hi"))
    assert sent == [(PHONE, FALLBACK)]


# ---------------------------------------------------------------- llm_timeout_seconds values


def build_llm(store: Any, llm: Any) -> tuple[WhatsAppChannel, GateEngine, list[tuple[str, str]]]:
    engine = GateEngine()
    channel, _, _, sent = build(store, queue=5.0, turn=1.0, engine=engine)
    channel.llm_timeout_seconds = llm  # as passed to the constructor; see the tests below
    return channel, engine, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [None, float("inf"), float("nan")])
async def test_llm_timeout_without_limit_keeps_working(any_store: Any, llm: Any) -> None:
    """None and inf meant "no limit" with the old wait_for() and still do; nan now does
    too. (With the first F patch None crashed the backstop sum and asyncio.timeout(nan)
    expired at once; before F, wait_for(nan) timed out or stopped the loop.)"""
    channel, engine, sent = build_llm(any_store, llm)
    engine.gates["slow"] = asyncio.Event()
    task = process(channel, "w1", "slow")
    await bounded(engine.started_event("slow").wait())
    await asyncio.sleep(0.1)  # the turn has to wait: a broken limit would fire or hang here
    engine.gates["slow"].set()
    await bounded(task)
    assert sent == [(PHONE, "re:slow")]


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [0, -1, float("-inf")])
async def test_non_positive_llm_timeout_times_out_every_turn_as_before(
    any_store: Any, llm: float, caplog: pytest.LogCaptureFixture
) -> None:
    channel, engine, sent = build_llm(any_store, llm)
    engine.gates["slow"] = asyncio.Event()
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        await bounded(process(channel, "w1", "slow"))
    assert sent == [(PHONE, FALLBACK)]
    assert "Turn timed out" in caplog.text
    assert "backstop" not in caplog.text
    assert history(any_store) == []


@pytest.mark.parametrize("llm", [0, -5, float("nan"), float("-inf")])
def test_constructor_warns_but_accepts_suspicious_llm_timeouts(
    llm: float, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.whatsapp"):
        channel = WhatsAppChannel(verify_token="tok", verify_signature=False, llm_timeout_seconds=llm)
    assert channel.llm_timeout_seconds is llm
    assert "llm_timeout_seconds=" in caplog.text


@pytest.mark.parametrize("llm", [None, 30.0, float("inf")])
def test_constructor_does_not_warn_for_valid_llm_timeouts(
    llm: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.whatsapp"):
        WhatsAppChannel(verify_token="tok", verify_signature=False, llm_timeout_seconds=llm)
    assert "llm_timeout_seconds=" not in caplog.text


# ---------------------------------------------------------------- log kinds


@pytest.mark.asyncio
async def test_backstop_is_logged_apart_from_the_turn_timeout(caplog: pytest.LogCaptureFixture) -> None:
    channel, _, _, sent = build(SessionStore(), queue=0.05, turn=0.05)

    async def hang(**_: Any) -> Any:
        await asyncio.Event().wait()

    with patch.object(channel, "dispatch_async", hang):
        with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
            await bounded(process(channel, "w1", "hi"))
    assert sent == [(PHONE, FALLBACK)]
    assert "Dispatch backstop expired" in caplog.text
    assert "Turn timed out" not in caplog.text
    assert "Session busy" not in caplog.text
    assert MASKED in caplog.text and PHONE not in caplog.text


@pytest.mark.asyncio
async def test_timeout_error_from_the_engine_is_logged_as_such(
    any_store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    class TimeoutEngine(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            raise asyncio.TimeoutError("provider timeout")

    channel, _, _, sent = build(any_store, queue=5.0, turn=5.0, engine=TimeoutEngine())
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        await bounded(process(channel, "w1", "hi"))
    assert sent == [(PHONE, FALLBACK)]
    assert "TimeoutError raised while dispatching" in caplog.text
    for other in ("Turn timed out", "Session busy", "backstop"):
        assert other not in caplog.text
    assert history(any_store) == []


@pytest.mark.asyncio
async def test_dispatch_async_override_raising_timeout_error_is_handled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    channel, _, _, sent = build(SessionStore(), queue=5.0, turn=5.0)

    async def override(*, session_id: str, message: str, metadata: Any = None) -> Any:
        raise asyncio.TimeoutError("override's own deadline")

    channel.dispatch_async = override  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        await bounded(process(channel, "w1", "hi"))
    assert sent == [(PHONE, FALLBACK)]
    assert "TimeoutError raised while dispatching" in caplog.text
    assert "Turn timed out" not in caplog.text


# ---------------------------------------------------------------- limits isolation


@pytest.mark.asyncio
async def test_dispatch_async_override_running_the_turn_in_a_task_keeps_the_limits(
    any_store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    channel, runner, engine, sent = build(any_store, queue=0.3, turn=5.0)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn(PHONE, "hold")
    await bounded(engine.started_event("hold").wait())

    async def override(*, session_id: str, message: str, metadata: Any = None) -> Any:
        return await asyncio.create_task(runner.run_async(session_id, message, metadata=metadata))

    channel.dispatch_async = override  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="chatflow_agent.whatsapp"):
        await bounded(process(channel, "w1", "late"))
    assert sent == [(PHONE, FALLBACK)]
    assert "Session busy" in caplog.text
    engine.gates["hold"].set()
    await bounded(holder)
    assert _dispatch_limits.get() is None


@pytest.mark.asyncio
async def test_sequential_dispatches_in_one_task_get_fresh_limits(any_store: Any) -> None:
    """Starlette runs the background tasks of one request one after another in the same
    task: a message that expired in the queue must not affect the next one."""
    channel, runner, engine, sent = build(any_store, queue=0.3, turn=5.0)
    engine.gates["hold"] = asyncio.Event()
    holder = runner.spawn_turn(PHONE, "hold")
    await bounded(engine.started_event("hold").wait())

    async def batch() -> None:
        await channel._process_message_background(wamid="w1", sender_phone=PHONE, user_text="late")
        engine.gates["hold"].set()
        await holder
        await channel._process_message_background(wamid="w2", sender_phone=PHONE, user_text="next")

    await bounded(batch())
    assert sent == [(PHONE, FALLBACK), (PHONE, "re:next")]
    assert _dispatch_limits.get() is None
