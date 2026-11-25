"""Boolean flags are not numeric timeout or history-limit settings."""

from pathlib import Path

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.core.memory import (
    SessionContext,
    SessionStore,
    SQLiteSessionContext,
    SQLiteSessionStore,
)
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Message


class CountingEngine(BaseEngine):
    def __init__(self) -> None:
        self.calls = 0

    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        self.calls += 1
        return EngineTurnResult(text="ok")


@pytest.mark.parametrize("value", [True, False])
def test_contexts_reject_boolean_turn_limits(value: bool, tmp_path: Path) -> None:
    agent = Agent(name="test", instructions="test")
    with pytest.raises(ValueError, match="max_turns"):
        SessionContext("test", agent, max_turns=value)
    with pytest.raises(ValueError, match="max_turns"):
        SQLiteSessionContext("test", agent, tmp_path / "context.db", max_turns=value)


@pytest.mark.parametrize("value", [True, False])
def test_stores_reject_boolean_turn_limits(value: bool, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="default_max_turns"):
        SessionStore(default_max_turns=value)
    path = tmp_path / "invalid.db"
    with pytest.raises(ValueError, match="default_max_turns"):
        with SQLiteSessionStore(path, default_max_turns=value):
            pass
    assert not path.exists()


@pytest.mark.parametrize("value", [True, False])
def test_whatsapp_rejects_boolean_queue_timeout(value: bool) -> None:
    with pytest.raises(ValueError, match="queue_timeout_seconds"):
        WhatsAppChannel(
            verify_token="fake-verify-token",
            app_secret="fake-app-secret",
            queue_timeout_seconds=value,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_runner_rejects_boolean_timeout_without_calling_engine(value: bool) -> None:
    engine = CountingEngine()
    async with Runner(
        starting_agent=Agent(name="test", instructions="test"), engine=engine
    ) as runner:
        with pytest.raises(ValueError, match="turn_timeout"):
            await runner.run_async(session_id="test", user_message="hi", turn_timeout=value)
    assert engine.calls == 0


@pytest.mark.parametrize("limit", [1, 40])
def test_positive_integer_turn_limits_remain_valid(limit: int, tmp_path: Path) -> None:
    agent = Agent(name="test", instructions="test")
    assert SessionContext("test", agent, max_turns=limit).max_turns == limit
    assert (
        SQLiteSessionContext("test", agent, tmp_path / "context.db", max_turns=limit).max_turns
        == limit
    )
    assert SessionStore(default_max_turns=limit).default_max_turns == limit
    with SQLiteSessionStore(tmp_path / "store.db", default_max_turns=limit) as store:
        assert store.default_max_turns == limit


@pytest.mark.parametrize("seconds", [1, 1.5])
def test_positive_numeric_queue_timeouts_remain_valid(seconds: float) -> None:
    channel = WhatsAppChannel(
        verify_token="fake-verify-token",
        app_secret="fake-app-secret",
        queue_timeout_seconds=seconds,
    )
    assert channel.queue_timeout_seconds == seconds


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [None, 1, 1.5])
async def test_numeric_and_disabled_turn_timeouts_remain_valid(seconds: float | None) -> None:
    engine = CountingEngine()
    async with Runner(
        starting_agent=Agent(name="test", instructions="test"), engine=engine
    ) as runner:
        response = await runner.run_async(
            session_id="test", user_message="hi", turn_timeout=seconds
        )
    assert response.content == "ok"
    assert engine.calls == 1
