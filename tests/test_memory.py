"""Tests for chatflow-agent session context and memory storage."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import SessionContext, SessionStore, SQLiteSessionStore
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


def test_session_store_copies_metadata_for_each_session(test_agents: tuple[Agent, Agent]) -> None:
    agent1, _ = test_agents
    original_metadata = {"channel": "web", "preferences": {"labels": ["support"]}}
    expected_metadata = {"channel": "web", "preferences": {"labels": ["support"]}}
    store = SessionStore()

    first = store.get_or_create("session_a", default_agent=agent1, metadata=original_metadata)
    second = store.get_or_create("session_b", default_agent=agent1, metadata=original_metadata)

    first.update_metadata({"private": "session_a"})
    first.metadata["preferences"]["labels"].append("private")

    assert first.metadata is not second.metadata
    assert first.metadata["preferences"] is not original_metadata["preferences"]
    assert second.metadata == expected_metadata
    assert original_metadata == expected_metadata


@pytest.mark.parametrize("limit", [0, -1, -10])
def test_invalid_max_turns(limit: int) -> None:
    agent = Agent(name="A", instructions="test")
    with pytest.raises(ValueError, match="max_turns"):
        SessionContext("s1", agent, max_turns=limit)


@pytest.mark.parametrize("limit", [0, -1, -10])
def test_session_store_rejects_invalid_default_max_turns(limit: int) -> None:
    with pytest.raises(ValueError, match="default_max_turns"):
        SessionStore(default_max_turns=limit)


def test_sqlite_session_store_get_uses_single_read_only_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_agents: tuple[Agent, Agent]
) -> None:
    agent1, agent2 = test_agents
    db_path = tmp_path / "single_get.db"
    with SQLiteSessionStore(db_path, default_max_turns=7) as store:
        store._agent_resolver = lambda name: {"Triage": agent1, "Billing": agent2}.get(name)
        ctx = store.get_or_create("s1", default_agent=agent1, metadata={"tier": "gold"})
        ctx.add_message(Message(role=Role.USER, content="first"))
        ctx.add_message(Message(role=Role.MODEL, content="second"))
        ctx.set_active_agent(agent2, context_updates={"routed": True})

        real_connect = sqlite3.connect
        connect_calls = 0
        executed_sql: list[str] = []

        def counting_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            nonlocal connect_calls
            connect_calls += 1
            conn = real_connect(*args, **kwargs)
            conn.set_trace_callback(lambda sql: executed_sql.append(sql.strip().upper()))
            return conn

        def forbid_get_or_create(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("get() must not delegate to get_or_create()")

        monkeypatch.setattr("chatflow_agent.core.memory.sqlite3.connect", counting_connect)
        monkeypatch.setattr(store, "get_or_create", forbid_get_or_create)

        loaded = store.get("s1")
        assert loaded is not None
        assert connect_calls == 1
        assert not any(
            sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in executed_sql
        )
        assert loaded.session_id == "s1"
        assert loaded.active_agent is agent2
        assert loaded.max_turns == 7
        assert loaded.metadata == {"tier": "gold", "routed": True}
        assert [m.content for m in loaded.history] == ["first", "second"]

        connect_calls = 0
        executed_sql.clear()
        assert store.get("missing") is None
        assert connect_calls == 1
        assert not any(
            sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in executed_sql
        )


def test_sqlite_session_store_get_fallback_agent_and_never_recreates_deleted_session(
    tmp_path: Path, test_agents: tuple[Agent, Agent]
) -> None:
    agent1, _ = test_agents
    db_path = tmp_path / "deleted_get.db"
    with SQLiteSessionStore(db_path) as store:
        store.get_or_create("s1", default_agent=agent1, metadata={"k": "v"})
        # Without an _agent_resolver, get() constructs a fallback Agent with the stored name
        loaded = store.get("s1")
        assert loaded is not None
        assert loaded.active_agent.name == "Triage"
        assert loaded.metadata == {"k": "v"}

        # Insert an orphaned message row and verify get() still returns None without recreating s1
        with store._connection() as conn:
            conn.execute("DELETE FROM chatflow_sessions WHERE session_id = ?", ("s1",))
            conn.execute(
                """
                INSERT INTO chatflow_messages (session_id, role, content, message_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "s1",
                    Role.USER.value,
                    "orphan",
                    Message(role=Role.USER, content="orphan").model_dump_json(),
                ),
            )

        assert store.get("s1") is None
        assert store.list_sessions() == []


def test_sqlite_session_store_get_reads_consistent_snapshot_across_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_agents: tuple[Agent, Agent]
) -> None:
    """get() holds a single read snapshot across chatflow_sessions and chatflow_messages."""
    agent1, agent2 = test_agents
    db_path = tmp_path / "snapshot_get.db"
    with SQLiteSessionStore(db_path) as store:
        store._agent_resolver = lambda name: {"Triage": agent1, "Billing": agent2}.get(name)
        ctx = store.get_or_create("s1", default_agent=agent1, metadata={"version": 1})
        ctx.add_message(Message(role=Role.USER, content="v1-msg"))

        real_connect = sqlite3.connect
        interleaved_mode = "update"

        def hooking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            conn = real_connect(*args, **kwargs)

            def on_sql(sql: str) -> None:
                normalized = " ".join(sql.split()).upper()
                if "SELECT MESSAGE_JSON FROM CHATFLOW_MESSAGES" in normalized:
                    # Fired after chatflow_sessions was read, right before chatflow_messages is read.
                    with real_connect(str(db_path)) as writer:
                        if interleaved_mode == "update":
                            writer.execute(
                                """
                                UPDATE chatflow_sessions
                                SET active_agent_name = 'Billing', metadata_json = '{"version": 2}'
                                WHERE session_id = 's1'
                                """
                            )
                            writer.execute(
                                """
                                INSERT INTO chatflow_messages (session_id, role, content, message_json)
                                VALUES (?, ?, ?, ?)
                                """,
                                (
                                    "s1",
                                    Role.MODEL.value,
                                    "v2-msg",
                                    Message(role=Role.MODEL, content="v2-msg").model_dump_json(),
                                ),
                            )
                        elif interleaved_mode == "delete":
                            writer.execute(
                                "DELETE FROM chatflow_messages WHERE session_id = 's1'"
                            )
                            writer.execute(
                                "DELETE FROM chatflow_sessions WHERE session_id = 's1'"
                            )
                        writer.commit()

            conn.set_trace_callback(on_sql)
            return conn

        monkeypatch.setattr("chatflow_agent.core.memory.sqlite3.connect", hooking_connect)

        # 1. Interleaved external update between reading chatflow_sessions and chatflow_messages:
        # get() must return the complete v1 snapshot, not v1 metadata mixed with v2 messages.
        snap_v1 = store.get("s1")
        assert snap_v1 is not None
        assert snap_v1.active_agent is agent1
        assert snap_v1.metadata == {"version": 1}
        assert [m.content for m in snap_v1.history] == ["v1-msg"]

        # 2. Interleaved external deletion between reading chatflow_sessions and chatflow_messages:
        # get() must return the complete v2 snapshot rather than a torn session with empty history.
        interleaved_mode = "delete"
        snap_v2 = store.get("s1")
        assert snap_v2 is not None
        assert snap_v2.active_agent is agent2
        assert snap_v2.metadata == {"version": 2}
        assert [m.content for m in snap_v2.history] == ["v1-msg", "v2-msg"]

        # 3. Subsequent get() sees the committed deletion.
        interleaved_mode = "none"
        assert store.get("s1") is None


