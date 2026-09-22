"""Tests for conversation memory integrity, turn-based pruning, and tool result pairing."""

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import SessionContext, SQLiteSessionStore
from chatflow_agent.types import Message, Role, ToolCall, ToolResult


def test_turn_based_pruning_preserves_tool_pairs_and_starts_with_user() -> None:
    """Pruning must discard complete turns and ensure the retained window begins with a USER message."""
    agent = Agent(name="TestAgent", instructions="Test")
    # Set max_turns = 2 complete turns
    session = SessionContext(session_id="test_pruning", initial_agent=agent, max_turns=2)

    # Add System prompt
    session.add_message(Message(role=Role.SYSTEM, content="System prompt instructions"))

    # Turn 1: User -> Model (calls tools 1 & 2) -> Tool 1 -> Tool 2 -> Model
    session.add_message(Message(role=Role.USER, content="User Question 1"))
    session.add_message(
        Message(
            role=Role.MODEL,
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="tool_a", args={}),
                ToolCall(id="call_2", name="tool_b", args={}),
            ],
        )
    )
    session.add_message(
        Message(
            role=Role.TOOL,
            tool_results=[
                ToolResult(tool_call_id="call_1", name="tool_a", content="Result A"),
                ToolResult(tool_call_id="call_2", name="tool_b", content="Result B"),
            ],
        )
    )
    session.add_message(Message(role=Role.MODEL, content="Answer 1"))

    # Turn 2: User -> Model (calls tool 3) -> Tool 3 -> Model
    session.add_message(Message(role=Role.USER, content="User Question 2"))
    session.add_message(
        Message(
            role=Role.MODEL,
            content="",
            tool_calls=[ToolCall(id="call_3", name="tool_c", args={})],
        )
    )
    session.add_message(
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(tool_call_id="call_3", name="tool_c", content="Result C")],
        )
    )
    session.add_message(Message(role=Role.MODEL, content="Answer 2"))

    # Turn 3: User -> Model answer
    session.add_message(Message(role=Role.USER, content="User Question 3"))
    session.add_message(Message(role=Role.MODEL, content="Answer 3"))

    # Assertions on retained history
    roles = [m.role for m in session.history]

    # 1. System prompt is preserved
    assert roles[0] == Role.SYSTEM
    assert session.history[0].content == "System prompt instructions"

    # 2. The retained conversation must start with Role.USER (Turn 2), never an orphaned Role.TOOL!
    assert roles[1] == Role.USER
    assert session.history[1].content == "User Question 2"

    # 3. Turn 2 tools are intact
    assert roles[2] == Role.MODEL
    assert roles[3] == Role.TOOL
    assert session.history[3].tool_results[0].content == "Result C"
    assert roles[4] == Role.MODEL
    assert session.history[4].content == "Answer 2"

    # 4. Turn 3 is present
    assert roles[5] == Role.USER
    assert roles[6] == Role.MODEL
    assert session.history[6].content == "Answer 3"


def test_sqlite_turn_based_pruning_integrity(tmp_path) -> None:
    """SQLiteSessionStore must prune old turns without leaving orphaned tools when reloaded from disk."""
    db_file = tmp_path / "sqlite_prune.db"
    store = SQLiteSessionStore(db_path=str(db_file), default_max_turns=2)
    agent = Agent(name="Bot", instructions="Test")

    ctx = store.get_or_create("user_sqlite_prune", default_agent=agent)
    ctx.add_message(Message(role=Role.SYSTEM, content="System Prompt"))

    # Turn 1
    ctx.add_message(Message(role=Role.USER, content="Turn 1 Question"))
    ctx.add_message(
        Message(
            role=Role.MODEL,
            content="",
            tool_calls=[ToolCall(id="c1", name="search", args={})],
        )
    )
    ctx.add_message(
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(tool_call_id="c1", name="search", content="search result 1")],
        )
    )
    ctx.add_message(Message(role=Role.MODEL, content="Turn 1 Final Answer"))

    # Turn 2
    ctx.add_message(Message(role=Role.USER, content="Turn 2 Question"))
    ctx.add_message(
        Message(
            role=Role.MODEL,
            content="",
            tool_calls=[ToolCall(id="c2", name="lookup", args={})],
        )
    )
    ctx.add_message(
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(tool_call_id="c2", name="lookup", content="lookup result 2")],
        )
    )
    ctx.add_message(Message(role=Role.MODEL, content="Turn 2 Final Answer"))

    # Turn 3
    ctx.add_message(Message(role=Role.USER, content="Turn 3 Question"))
    ctx.add_message(Message(role=Role.MODEL, content="Turn 3 Final Answer"))

    # Now reload completely from SQLite
    new_store = SQLiteSessionStore(db_path=str(db_file), default_max_turns=2)
    reloaded_ctx = new_store.get_or_create("user_sqlite_prune", default_agent=agent)

    reloaded_roles = [m.role for m in reloaded_ctx.history]
    assert reloaded_roles[0] == Role.SYSTEM
    assert reloaded_roles[1] == Role.USER
    assert reloaded_ctx.history[1].content == "Turn 2 Question"
    # Ensure no orphaned TOOL messages
    assert Role.TOOL not in [reloaded_roles[1]]
