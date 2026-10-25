import asyncio
from pathlib import Path

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.engines.base import EngineTurnResult
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Handoff, Message, Role, ToolCall


class ControlledEngine(BaseEngine):
    def __init__(self) -> None:
        self.block = False
        self.entered = asyncio.Event()

    async def generate_turn_async(
        self, agent: Agent, history: list[Message]
    ) -> EngineTurnResult:
        if self.block:
            self.entered.set()
            await asyncio.Event().wait()
        return EngineTurnResult(text="valid response")


def make_store(kind: str, tmp_path: Path):
    if kind == "memory":
        return SessionStore(default_max_turns=1)
    return SQLiteSessionStore(tmp_path / "sessions.db", default_max_turns=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
async def test_cancelled_turn_restores_pruned_history_and_propagates(
    store_kind: str, tmp_path: Path
) -> None:
    agent = Agent(name="A", instructions="test")
    engine = ControlledEngine()
    store = make_store(store_kind, tmp_path)
    runner = Runner(starting_agent=agent, engine=engine, session_store=store)

    await runner.run_async("s", "valid turn", metadata={"source": "original"})
    session = store.get("s")
    assert session is not None
    before = [message.content for message in session.history]
    assert before == ["valid turn", "valid response"]

    engine.block = True
    task = asyncio.create_task(runner.run_async("s", "cancelled turn"))
    await engine.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    restored = store.get("s")
    assert restored is not None
    assert [message.content for message in restored.history] == before
    assert restored.active_agent.name == "A"
    assert restored.metadata == {"source": "original"}

    if store_kind == "sqlite":
        reloaded_store = SQLiteSessionStore(tmp_path / "sessions.db", default_max_turns=1)
        reloaded_runner = Runner(
            starting_agent=agent,
            session_store=reloaded_store,
            engine=ControlledEngine(),
        )
        reloaded = reloaded_runner.get_session("s")
        assert reloaded is not None
        assert [message.content for message in reloaded.history] == before
        assert reloaded.metadata == {"source": "original"}


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
async def test_cancellation_restores_active_agent_and_metadata_after_handoff(
    store_kind: str, tmp_path: Path
) -> None:
    specialist = Agent(name="Specialist", instructions="handle the request")
    triage = Agent(name="Triage", instructions="route the request", handoffs=[specialist])
    store = make_store(store_kind, tmp_path)

    class HandoffThenBlockEngine(BaseEngine):
        def __init__(self) -> None:
            self.calls = 0
            self.entered = asyncio.Event()

        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            self.calls += 1
            if self.calls == 1:
                return EngineTurnResult(
                    tool_calls=[
                        ToolCall(
                            id="handoff-1",
                            name="transfer_to_specialist",
                            args={},
                        )
                    ]
                )
            self.entered.set()
            await asyncio.Event().wait()

    @triage.tool
    def transfer_to_specialist() -> Handoff:
        return Handoff(
            target_agent_name="Specialist",
            context_updates={"handoff_marker": "temporary"},
        )

    engine = HandoffThenBlockEngine()
    runner = Runner(starting_agent=triage, engine=engine, session_store=store)
    session = store.get_or_create(
        "handoff-session", triage, metadata={"account": "original"}
    )
    session.add_message(Message(role=Role.USER, content="previous turn"))
    session.add_message(Message(role=Role.MODEL, content="previous response"))

    task = asyncio.create_task(runner.run_async("handoff-session", "new request"))
    await engine.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    restored = store.get("handoff-session")
    assert restored is not None
    assert restored.active_agent.name == "Triage"
    assert restored.metadata == {"account": "original"}
    assert [message.content for message in restored.history] == [
        "previous turn",
        "previous response",
    ]

    if store_kind == "sqlite":
        reloaded_store = SQLiteSessionStore(tmp_path / "sessions.db", default_max_turns=1)
        reloaded_runner = Runner(
            starting_agent=triage,
            session_store=reloaded_store,
            engine=ControlledEngine(),
        )
        reloaded = reloaded_runner.get_session("handoff-session")
        assert reloaded is not None
        assert reloaded.active_agent.name == "Triage"
        assert reloaded.metadata == {"account": "original"}
        assert [message.content for message in reloaded.history] == [
            "previous turn",
            "previous response",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
async def test_timeout_cancellation_restores_session_state(
    store_kind: str, tmp_path: Path
) -> None:
    agent = Agent(name="A", instructions="test")
    engine = ControlledEngine()
    store = make_store(store_kind, tmp_path)
    runner = Runner(starting_agent=agent, engine=engine, session_store=store)

    await runner.run_async("timeout-session", "valid turn", metadata={"source": "original"})
    engine.block = True
    task = asyncio.create_task(runner.run_async("timeout-session", "timed-out turn"))
    # Start the timeout only once the turn runs its engine call: with a file store its disk
    # writes come first, and a timeout firing during them would never reach the engine.
    try:
        await asyncio.wait_for(engine.entered.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=15.0)
        pytest.fail("the turn never reached its engine call")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=0.1)
    # wait_for returns only after the cancelled turn has finished (rolled back).
    assert task.cancelled()

    restored = store.get("timeout-session")
    assert restored is not None
    assert [message.content for message in restored.history] == [
        "valid turn",
        "valid response",
    ]
    assert restored.metadata == {"source": "original"}
