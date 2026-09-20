"""Tests for chatflow-agent persistent SQLiteSessionStore."""

from pathlib import Path

import pytest

from chatflow_agent import (
    BaseSessionContext,
    BaseSessionStore,
    Runner,
    SQLiteSessionContext,
    SQLiteSessionStore,
)
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import EngineTurnResult, GeminiEngine
from chatflow_agent.memory import SQLiteSessionStore as SQLiteStoreFromMemory
from chatflow_agent.types import Message, Role, ToolCall


class MockEngine(GeminiEngine):
    """Mock engine that simulates turn evaluations deterministically."""

    def __init__(self, turns_sequence: list[EngineTurnResult]) -> None:
        super().__init__()
        self.turns = list(turns_sequence)

    async def generate_turn_async(
        self,
        agent: Agent,
        history: list[Message],
    ) -> EngineTurnResult:
        if not self.turns:
            return EngineTurnResult(text="No more turns queued.")
        return self.turns.pop(0)


@pytest.fixture
def test_agents() -> tuple[Agent, Agent]:
    billing = Agent(name="Billing", instructions="You handle invoices and billing inquiries.")
    triage = Agent(name="Triage", instructions="You route users to specialists.", handoffs=[billing])
    return triage, billing


def test_sqlite_convenience_imports() -> None:
    assert SQLiteSessionStore is SQLiteStoreFromMemory
    assert issubclass(SQLiteSessionStore, BaseSessionStore)
    assert issubclass(SQLiteSessionContext, BaseSessionContext)


def test_sqlite_context_initialization(tmp_path: Path, test_agents: tuple[Agent, Agent]) -> None:
    triage, _ = test_agents
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path=db_path, default_max_turns=5)

    ctx = store.get_or_create(session_id="user_123", default_agent=triage, metadata={"channel": "whatsapp"})

    assert ctx.session_id == "user_123"
    assert ctx.active_agent.name == "Triage"
    assert len(ctx.history) == 0
    assert ctx.metadata == {"channel": "whatsapp"}
    assert "user_123" in store.list_sessions()


def test_sqlite_messages_persistence_and_reload(tmp_path: Path, test_agents: tuple[Agent, Agent]) -> None:
    triage, _ = test_agents
    db_path = tmp_path / "sessions.db"

    # Store instance 1: create session and add messages
    store1 = SQLiteSessionStore(db_path=db_path)
    session1 = store1.get_or_create("user_456", default_agent=triage)
    session1.add_message(Message(role=Role.USER, content="Hello from WhatsApp"))
    session1.add_message(Message(role=Role.MODEL, content="Welcome, how can I help?"))

    assert len(session1.history) == 2

    # Store instance 2 (simulating application restart): re-read from sqlite db
    store2 = SQLiteSessionStore(db_path=db_path)
    assert "user_456" in store2.list_sessions()

    session2 = store2.get_or_create("user_456", default_agent=triage)
    assert len(session2.history) == 2
    assert session2.history[0].role == Role.USER
    assert session2.history[0].content == "Hello from WhatsApp"
    assert session2.history[1].role == Role.MODEL
    assert session2.history[1].content == "Welcome, how can I help?"


def test_sqlite_sliding_window_pruning(tmp_path: Path, test_agents: tuple[Agent, Agent]) -> None:
    triage, _ = test_agents
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path=db_path, default_max_turns=2)  # max 2 turns = max 4 messages

    ctx = store.get_or_create("user_prune", default_agent=triage)
    for i in range(6):
        role = Role.USER if i % 2 == 0 else Role.MODEL
        ctx.add_message(Message(role=role, content=f"Message {i}"))

    # In-memory history is pruned to 4
    assert len(ctx.history) == 4
    assert ctx.history[0].content == "Message 2"
    assert ctx.history[-1].content == "Message 5"

    # Reload from disk in a fresh store: verify SQLite table is also pruned
    reloaded_store = SQLiteSessionStore(db_path=db_path, default_max_turns=2)
    reloaded_ctx = reloaded_store.get_or_create("user_prune", default_agent=triage)
    assert len(reloaded_ctx.history) == 4
    assert reloaded_ctx.history[0].content == "Message 2"
    assert reloaded_ctx.history[-1].content == "Message 5"


def test_sqlite_active_agent_switch_persistence(tmp_path: Path, test_agents: tuple[Agent, Agent]) -> None:
    triage, billing = test_agents
    db_path = tmp_path / "sessions.db"

    # Store 1: switch active agent to Billing
    store1 = SQLiteSessionStore(db_path=db_path)
    ctx1 = store1.get_or_create("user_handoff", default_agent=triage)
    assert ctx1.active_agent.name == "Triage"

    ctx1.set_active_agent(billing)
    assert ctx1.active_agent.name == "Billing"

    # Store 2: configure resolver and reload
    store2 = SQLiteSessionStore(db_path=db_path)
    agents_map = {"Triage": triage, "Billing": billing}
    store2.set_agent_resolver(lambda name: agents_map.get(name))

    ctx2 = store2.get_or_create("user_handoff", default_agent=triage)
    assert ctx2.active_agent.name == "Billing"


def test_sqlite_delete_and_clear_history(tmp_path: Path, test_agents: tuple[Agent, Agent]) -> None:
    triage, _ = test_agents
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path=db_path)

    ctx = store.get_or_create("user_del", default_agent=triage)
    ctx.add_message(Message(role=Role.USER, content="To be deleted"))
    assert len(ctx.history) == 1

    # Clear history only
    ctx.clear_history()
    assert len(ctx.history) == 0

    reloaded = store.get("user_del")
    assert reloaded is not None
    assert len(reloaded.history) == 0

    # Delete entire session
    assert store.delete("user_del") is True
    assert store.get("user_del") is None
    assert "user_del" not in store.list_sessions()
    assert store.delete("non_existent") is False


@pytest.mark.asyncio
async def test_sqlite_runner_survives_reboot(tmp_path: Path, test_agents: tuple[Agent, Agent]) -> None:
    triage, billing = test_agents
    db_path = tmp_path / "runner_sessions.db"

    # Server Instance 1: Runner runs turn with handoff to Billing
    mock_engine1 = MockEngine([
        EngineTurnResult(
            tool_calls=[
                ToolCall(
                    id="call_h1",
                    name="transfer_to_billing",
                    args={"reason": "Customer invoice query"},
                )
            ]
        ),
        EngineTurnResult(text="Hello, I am Billing. What invoice do you need?"),
    ])
    store1 = SQLiteSessionStore(db_path=db_path)
    runner1 = Runner(starting_agent=triage, session_store=store1, engine=mock_engine1)

    res1 = await runner1.run_async(session_id="customer_99", user_message="I need an invoice")
    assert res1.active_agent_name == "Billing"
    assert res1.content == "Hello, I am Billing. What invoice do you need?"

    # Server Instance 2 (Simulated reboot with fresh Runner pointing to same DB)
    mock_engine2 = MockEngine([
        EngineTurnResult(text="Here is invoice #12345 for $500."),
    ])
    store2 = SQLiteSessionStore(db_path=db_path)
    runner2 = Runner(starting_agent=triage, session_store=store2, engine=mock_engine2)

    res2 = await runner2.run_async(session_id="customer_99", user_message="Invoice for May")
    # Verified: Runner 2 immediately routed to Billing without defaulting back to Triage!
    assert res2.active_agent_name == "Billing"
    assert res2.content == "Here is invoice #12345 for $500."

    session = runner2.get_session("customer_99")
    assert session is not None
    assert session.active_agent.name == "Billing"
