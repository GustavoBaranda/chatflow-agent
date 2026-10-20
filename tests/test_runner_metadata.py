"""Per-turn session metadata must be merged, persisted, and rolled back safely."""

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine, EngineTurnResult
from chatflow_agent.core.memory import BaseSessionStore, SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Handoff, Message, Role, ToolCall


class MetadataEngine(BaseEngine):
    def __init__(self, store: BaseSessionStore) -> None:
        self.store = store
        self.observed: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.block = False
        self.entered = asyncio.Event()

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        session = self.store.get("session")
        assert session is not None
        self.observed.append(copy.deepcopy(session.metadata))
        if self.error is not None:
            raise self.error
        if self.block:
            self.entered.set()
            await asyncio.Event().wait()
        return EngineTurnResult(text="Reply")


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> BaseSessionStore:
    if request.param == "memory":
        return SessionStore()
    return SQLiteSessionStore(tmp_path / "sessions.db")


@pytest.fixture
def runner(store: BaseSessionStore) -> Runner:
    return Runner(
        starting_agent=Agent(name="Assistant", instructions="Help the user."),
        session_store=store,
        engine=MetadataEngine(store),
    )


@pytest.mark.parametrize("updates", [{}, {"wamid": "second", "new_key": False}])
def test_metadata_update_preserves_agent_and_history(
    store: BaseSessionStore, updates: dict[str, Any]
) -> None:
    agent = Agent(name="Assistant", instructions="Help the user.")
    original = {"channel": "whatsapp", "wamid": "first"}
    session = store.get_or_create("session", agent, metadata=original.copy())
    session.add_message(Message(role=Role.USER, content="Previous message"))
    history = list(session.history)

    session.update_metadata(updates)

    assert session.metadata == {**original, **updates}
    assert session.active_agent is agent
    assert session.history == history
    restored = store.get("session")
    assert restored is not None
    assert restored.metadata == session.metadata
    assert restored.active_agent.name == agent.name


@pytest.mark.asyncio
async def test_next_turn_metadata_preserves_handoff_context(
    store: BaseSessionStore, tmp_path: Path
) -> None:
    billing = Agent(name="Billing", instructions="Review invoices.")
    triage = Agent(name="Triage", instructions="Route the request.", handoffs=[billing])

    @triage.tool
    def transfer_with_context() -> Handoff:
        return Handoff(
            target_agent_name="Billing",
            context_updates={"customer_tier": "VIP", "invoice_id": "INV-101"},
        )

    class HandoffEngine(MetadataEngine):
        def __init__(self) -> None:
            super().__init__(store)
            self.calls = 0

        async def generate_turn_async(
            self, agent: Agent, history: list[Message]
        ) -> EngineTurnResult:
            self.calls += 1
            if self.calls == 1:
                return EngineTurnResult(
                    tool_calls=[ToolCall(id="handoff", name="transfer_with_context", args={})]
                )
            return await super().generate_turn_async(agent, history)

    engine = HandoffEngine()
    runner = Runner(triage, session_store=store, engine=engine)
    await runner.run_async(
        "session",
        "First message",
        {"channel": "whatsapp", "wamid": "first", "customer_tier": "standard"},
    )

    response = await runner.run_async(
        "session", "Second message", {"wamid": "second", "request_id": "request-2"}
    )

    expected = {
        "channel": "whatsapp",
        "wamid": "second",
        "customer_tier": "VIP",
        "invoice_id": "INV-101",
        "request_id": "request-2",
    }
    assert engine.observed[-1] == expected
    assert response.active_agent_name == "Billing"
    restored_store = (
        SQLiteSessionStore(tmp_path / "sessions.db")
        if isinstance(store, SQLiteSessionStore)
        else store
    )
    restored = restored_store.get("session")
    assert restored is not None
    assert restored.metadata == expected
    assert restored.active_agent.name == "Billing"


@pytest.mark.asyncio
async def test_second_turn_merges_metadata_before_engine_runs(
    runner: Runner, store: BaseSessionStore, tmp_path: Path
) -> None:
    await runner.run_async("session", "First message", {"channel": "whatsapp", "wamid": "first"})
    await runner.run_async("session", "Second message", {"wamid": "second", "message_count": 2})

    expected = {"channel": "whatsapp", "wamid": "second", "message_count": 2}
    assert isinstance(runner.engine, MetadataEngine)
    assert runner.engine.observed[-1] == expected
    session = store.get("session")
    assert session is not None
    assert session.metadata == expected
    if isinstance(store, SQLiteSessionStore):
        reloaded = SQLiteSessionStore(tmp_path / "sessions.db").get("session")
        assert reloaded is not None
        assert reloaded.metadata == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {}])
async def test_empty_metadata_leaves_existing_values_unchanged(
    runner: Runner, store: BaseSessionStore, metadata: dict[str, Any] | None
) -> None:
    original = {"channel": "whatsapp", "wamid": "first"}
    await runner.run_async("session", "First message", original.copy())
    await runner.run_async("session", "Second message", metadata)

    assert isinstance(runner.engine, MetadataEngine)
    assert runner.engine.observed[-1] == original
    session = store.get("session")
    assert session is not None
    assert session.metadata == original


@pytest.mark.asyncio
async def test_failed_turn_restores_previous_metadata(
    runner: Runner, store: BaseSessionStore
) -> None:
    original = {"channel": "whatsapp", "wamid": "first"}
    await runner.run_async("session", "First message", original.copy())
    assert isinstance(runner.engine, MetadataEngine)
    runner.engine.error = RuntimeError("Engine failed")

    with pytest.raises(RuntimeError, match="Engine failed"):
        await runner.run_async("session", "Failed message", {"wamid": "second", "temporary": True})

    assert runner.engine.observed[-1] == {**original, "wamid": "second", "temporary": True}
    restored = store.get("session")
    assert restored is not None
    assert restored.metadata == original
    assert restored.active_agent.name == "Assistant"
    assert [message.content for message in restored.history] == ["First message", "Reply"]


@pytest.mark.asyncio
async def test_cancelled_turn_restores_previous_metadata(
    runner: Runner, store: BaseSessionStore
) -> None:
    original = {"channel": "whatsapp", "wamid": "first"}
    await runner.run_async("session", "First message", original.copy())
    assert isinstance(runner.engine, MetadataEngine)
    runner.engine.block = True
    task = asyncio.create_task(
        runner.run_async("session", "Cancelled message", {"wamid": "second", "temporary": True})
    )
    await runner.engine.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.engine.observed[-1] == {**original, "wamid": "second", "temporary": True}
    restored = store.get("session")
    assert restored is not None
    assert restored.metadata == original
    assert restored.active_agent.name == "Assistant"
    assert [message.content for message in restored.history] == ["First message", "Reply"]


@pytest.mark.asyncio
async def test_metadata_merge_waits_for_session_lock() -> None:
    store = SessionStore()
    engine = MetadataEngine(store)
    runner = Runner(Agent(name="Assistant", instructions="Help."), store, engine)
    await runner.run_async("session", "First message", {"wamid": "first"})
    session = store.get("session")
    assert session is not None
    queued = asyncio.Event()

    async def second_turn() -> None:
        queued.set()
        await runner.run_async("session", "Second message", {"wamid": "second"})

    async with session.lock:
        task = asyncio.create_task(second_turn())
        await queued.wait()
        assert session.metadata == {"wamid": "first"}
        assert len(engine.observed) == 1
    await task
    assert session.metadata == {"wamid": "second"}
