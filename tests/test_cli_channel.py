"""Tests for BaseChannel and CLIChannel."""

import pytest

from chatflow_agent.channels.base import BaseChannel, ChannelError
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner


class DummyChannel(BaseChannel):
    """Minimal concrete channel for base tests."""

    def run(self, **kwargs) -> None:
        pass


class MockEngine(GeminiEngine):
    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply

    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text=self.reply)


def test_base_channel_detached_error() -> None:
    channel = DummyChannel()
    with pytest.raises(ChannelError) as exc_info:
        channel.dispatch(session_id="s1", message="hello")
    assert "No Runner is attached" in str(exc_info.value)


@pytest.mark.asyncio
async def test_base_channel_detached_error_async() -> None:
    channel = DummyChannel()
    with pytest.raises(ChannelError) as exc_info:
        await channel.dispatch_async(session_id="s1", message="hello")
    assert "No Runner is attached" in str(exc_info.value)


def test_base_channel_attach_and_dispatch() -> None:
    agent = Agent(name="Greeter", instructions="Greet.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Welcome!"))

    channel = DummyChannel()
    ret = channel.attach(runner)
    assert ret is channel  # Fluent interface

    resp = channel.dispatch(session_id="test_session", message="Hi")
    assert resp.content == "Welcome!"
    assert resp.active_agent_name == "Greeter"


def test_cli_channel_send_message() -> None:
    agent = Agent(name="CLI Assistant", instructions="Assist user via CLI.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."))

    cli = CLIChannel(session_id="custom_cli_user")
    cli.attach(runner)

    resp = cli.send_message("Run diagnostic")
    assert resp.content == "Command processed."
    assert resp.active_agent_name == "CLI Assistant"


# --- /reset command (#85) ----------------------------------------------------


def _drive_cli_loop(cli: CLIChannel, inputs: list[str], monkeypatch) -> None:
    """Run CLIChannel.run() with scripted terminal input."""
    iterator = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(iterator))
    cli.run()


def test_cli_reset_when_no_session_exists_prints_and_skips_dispatch(
    monkeypatch, capsys
) -> None:
    """/reset with no prior session is a safe no-op that never reaches the agent."""
    agent = Agent(name="CLI Assistant", instructions="Assist user via CLI.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."))

    cli = CLIChannel(session_id="no_session_yet")
    cli.attach(runner)

    def _fail_dispatch(self, session_id, message):
        raise AssertionError(f"dispatch() must not be called for {message!r}")

    monkeypatch.setattr(type(cli), "dispatch", _fail_dispatch)

    _drive_cli_loop(cli, ["/reset", "exit"], monkeypatch)

    assert runner.get_session("no_session_yet") is None
    assert "Conversation memory has been reset." in capsys.readouterr().out


def test_cli_reset_clears_active_session_history(monkeypatch, capsys) -> None:
    """/reset clears the in-memory SessionStore history for the current session."""
    agent = Agent(name="CLI Assistant", instructions="Assist user via CLI.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."))

    cli = CLIChannel(session_id="cli_user")
    cli.attach(runner)
    cli.send_message("remember this")

    session = runner.get_session("cli_user")
    assert session is not None
    assert len(session.history) > 0

    _drive_cli_loop(cli, ["/reset", "exit"], monkeypatch)

    assert session.history == []
    assert "Conversation memory has been reset." in capsys.readouterr().out


def test_cli_reset_with_sqlite_session_clears_persisted_history(
    monkeypatch, capsys, tmp_path
) -> None:
    """/reset deletes the persisted history so a fresh store reloads nothing."""
    from chatflow_agent.core.memory import SQLiteSessionStore

    agent = Agent(name="CLI Assistant", instructions="Assist user via CLI.")
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path=str(db_path))
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."), session_store=store)

    cli = CLIChannel(session_id="sqlite_user")
    cli.attach(runner)
    cli.send_message("persist this")

    _drive_cli_loop(cli, ["/reset", "exit"], monkeypatch)
    assert "Conversation memory has been reset." in capsys.readouterr().out

    store.close()

    reloaded = SQLiteSessionStore(db_path=str(db_path))
    try:
        session = reloaded.get("sqlite_user")
        assert session is not None
        assert session.history == []
    finally:
        reloaded.close()


def test_cli_reset_only_matches_exact_command(monkeypatch, capsys) -> None:
    """Messages merely containing 'reset' are still dispatched to the agent."""
    agent = Agent(name="CLI Assistant", instructions="Assist user via CLI.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."))

    cli = CLIChannel(session_id="cli_user")
    cli.attach(runner)

    _drive_cli_loop(cli, ["please reset my password", "exit"], monkeypatch)

    out = capsys.readouterr().out
    assert "Conversation memory has been reset." not in out
    assert "[CLI Assistant]: Command processed." in out
