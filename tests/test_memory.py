"""Tests for chatflow-agent session context and memory storage."""

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import SessionContext, SessionStore
from chatflow_agent.types import Message, Role


@pytest.fixture
def test_agents() -> tuple[Agent, Agent]:
    agent1 = Agent(name="Triage", instructions="Triage prompt")
    agent2 = Agent(name="Billing", instructions="Billing prompt")
    return agent1, agent2


def test_session_context_initialization(test_agents: tuple[Agent, Agent]) -> None:
    agent1, _ = test_agents
    ctx = SessionContext(session_id="chat_101", initial_agent=agent1, max_turns=5)

    assert ctx.session_id == "chat_101"
    assert ctx.active_agent.name == "Triage"
    assert len(ctx.history) == 0
    assert ctx.max_turns == 5


def test_session_context_add_and_clear_messages(test_agents: tuple[Agent, Agent]) -> None:
    agent1, _ = test_agents
    ctx = SessionContext(session_id="chat_101", initial_agent=agent1)

    msg1 = Message(role=Role.USER, content="Hello")
    msg2 = Message(role=Role.MODEL, content="Hi there!")
    ctx.add_message(msg1)
    ctx.add_message(msg2)

    assert len(ctx.history) == 2
    assert ctx.history[0].content == "Hello"
    assert ctx.history[1].content == "Hi there!"

    ctx.clear_history()
    assert len(ctx.history) == 0


def test_session_context_sliding_window_pruning(test_agents: tuple[Agent, Agent]) -> None:
    agent1, _ = test_agents
    # Max 2 turns = max 4 messages
    ctx = SessionContext(session_id="chat_102", initial_agent=agent1, max_turns=2)

    # Add 6 messages
    for i in range(6):
        role = Role.USER if i % 2 == 0 else Role.MODEL
        ctx.add_message(Message(role=role, content=f"Message {i}"))

    # Should retain exactly 4 messages: messages 2, 3, 4, 5
    assert len(ctx.history) == 4
    assert ctx.history[0].content == "Message 2"
    assert ctx.history[-1].content == "Message 5"


def test_session_context_switch_active_agent(test_agents: tuple[Agent, Agent]) -> None:
    agent1, agent2 = test_agents
    ctx = SessionContext(session_id="chat_103", initial_agent=agent1)

    assert ctx.active_agent.name == "Triage"
    ctx.set_active_agent(agent2)
    assert ctx.active_agent.name == "Billing"


def test_session_store_lifecycle(test_agents: tuple[Agent, Agent]) -> None:
    agent1, _ = test_agents
    store = SessionStore(default_max_turns=10)

    # 1. Create new session
    session_a = store.get_or_create("user_alpha", default_agent=agent1)
    assert session_a.session_id == "user_alpha"
    assert session_a.active_agent.name == "Triage"

    # 2. Add turn
    session_a.add_message(Message(role=Role.USER, content="Ping"))

    # 3. Retrieve existing session
    session_again = store.get_or_create("user_alpha", default_agent=agent1)
    assert len(session_again.history) == 1
    assert session_again.history[0].content == "Ping"

    # 4. List and check sessions
    assert "user_alpha" in store.list_sessions()
    assert store.get("user_alpha") is not None
    assert store.get("non_existent") is None

    # 5. Delete session
    assert store.delete("user_alpha") is True
    assert store.get("user_alpha") is None
    assert "user_alpha" not in store.list_sessions()
