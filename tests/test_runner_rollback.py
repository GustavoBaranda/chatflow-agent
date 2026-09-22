"""Tests for Runner history rollback on turn failure or cancellation."""

from pathlib import Path

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.memory import SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Role, ToolCall


class FailingEngine(GeminiEngine):
    """Engine that succeeds on first call then fails with ProviderError."""

    def __init__(self, succeed_times: int = 1) -> None:
        super().__init__()
        self.succeed_times = succeed_times
        self.call_count = 0

    async def generate_turn_async(self, agent, history):
        self.call_count += 1
        if self.call_count <= self.succeed_times:
            return EngineTurnResult(text="Successful turn response")
        raise ProviderError("Simulated LLM Provider 500 / 429 error")


class ToolThenFailEngine(GeminiEngine):
    """Engine that requests a tool on turn 1, then fails on turn 2."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate_turn_async(self, agent, history):
        self.calls += 1
        if self.calls == 1:
            return EngineTurnResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="dummy_tool", args={})],
            )
        raise ProviderError("Engine failed after tool execution!")


@pytest.mark.asyncio
async def test_runner_rolls_back_in_memory_history_on_failure() -> None:
    """If a turn raises an exception, the partial user message must be rolled back."""
    engine = FailingEngine(succeed_times=1)
    agent = Agent(name="Bot", instructions="Test")
    runner = Runner(starting_agent=agent, engine=engine)

    # Turn 1: succeeds
    res1 = await runner.run_async(session_id="user_rollback_1", user_message="Hello 1")
    assert res1.content == "Successful turn response"

    session = runner.get_session("user_rollback_1")
    assert session is not None
    assert len(session.history) == 2  # USER + MODEL

    # Turn 2: fails with ProviderError
    with pytest.raises(ProviderError):
        await runner.run_async(session_id="user_rollback_1", user_message="Hello 2 (fails)")

    # History must be rolled back to exactly length 2 (Turn 1 only)
    assert len(session.history) == 2
    assert session.history[-1].content == "Successful turn response"
    assert session.history[-1].role == Role.MODEL


@pytest.mark.asyncio
async def test_runner_rolls_back_sqlite_history_on_failure(tmp_path: Path) -> None:
    """SQLite session store must roll back messages from failed turn to disk."""
    db_file = tmp_path / "rollback_test.db"
    store = SQLiteSessionStore(db_path=db_file)
    engine = FailingEngine(succeed_times=1)
    agent = Agent(name="Bot", instructions="Test")
    runner = Runner(starting_agent=agent, session_store=store, engine=engine)

    # Turn 1 succeeds
    await runner.run_async(session_id="user_sql_rb", user_message="Hello Turn 1")

    # Turn 2 fails
    with pytest.raises(ProviderError):
        await runner.run_async(session_id="user_sql_rb", user_message="Hello Turn 2 (will fail)")

    # Fresh reload from disk: verify only Turn 1 exists
    store2 = SQLiteSessionStore(db_path=db_file)
    reloaded = store2.get_or_create("user_sql_rb", default_agent=agent)
    assert len(reloaded.history) == 2
    assert reloaded.history[0].role == Role.USER
    assert reloaded.history[0].content == "Hello Turn 1"
    assert reloaded.history[1].role == Role.MODEL
    assert reloaded.history[1].content == "Successful turn response"


@pytest.mark.asyncio
async def test_runner_rolls_back_partial_tool_sequence_on_failure() -> None:
    """Partial tool calls executed before turn failure must be rolled back to avoid orphaned tools."""
    engine = ToolThenFailEngine()
    agent = Agent(name="ToolBot", instructions="Test")

    @agent.tool
    def dummy_tool() -> str:
        return "tool execution ok"

    runner = Runner(starting_agent=agent, engine=engine)
    session = runner.session_store.get_or_create("user_tool_rb", default_agent=agent)

    # Turn fails after tool executes
    with pytest.raises(ProviderError):
        await runner.run_async(session_id="user_tool_rb", user_message="Trigger tool then fail")

    # History must be completely empty (rolled back to 0 messages)
    assert len(session.history) == 0
