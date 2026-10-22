"""Tests for BaseChannel and CLIChannel."""

import pytest
from chatflow_agent.channels.base import BaseChannel, ChannelError
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import AgentResponse, Handoff


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


def test_cli_run_exit_commands(monkeypatch, capsys) -> None:
    """Test clean exit on exit/quit/salir."""
    agent = Agent(name="TestAgent", instructions="Test.")
    runner = Runner(starting_agent=agent, engine=MockEngine("ok"))
    cli = CLIChannel()
    cli.attach(runner)

    for cmd in ["exit", "quit", "salir"]:
        inputs = iter([cmd])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run()
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out
        assert "Session ended by user." not in captured.out


def test_cli_run_eof_and_keyboard_interrupt(monkeypatch, capsys) -> None:
    """Test clean exit on EOFError and KeyboardInterrupt."""
    agent = Agent(name="TestAgent", instructions="Test.")
    runner = Runner(starting_agent=agent, engine=MockEngine("ok"))
    cli = CLIChannel()
    cli.attach(runner)

    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    cli.run()
    captured = capsys.readouterr()
    assert "Session ended by user." in captured.out

    def raise_kb(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_kb)
    cli.run()
    captured = capsys.readouterr()
    assert "Session ended by user." in captured.out


def test_cli_run_skip_empty_input(monkeypatch, capsys) -> None:
    """Test that empty inputs are skipped."""
    agent = Agent(name="TestAgent", instructions="Test.")
    runner = Runner(starting_agent=agent, engine=MockEngine("ok"))
    cli = CLIChannel()
    cli.attach(runner)

    inputs = iter(["", "   ", "hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # Mock dispatch to count calls
    call_count = 0
    original_dispatch = cli.dispatch

    def mock_dispatch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_dispatch(*args, **kwargs)

    cli.dispatch = mock_dispatch  # type: ignore
    cli.run()
    assert call_count == 1  # only "hello" dispatched


def test_cli_run_handoff_display(monkeypatch, capsys) -> None:
    """Test handoff indicator printing."""
    agent = Agent(name="TestAgent", instructions="Test.")
    runner = Runner(starting_agent=agent, engine=MockEngine("ok"))
    cli = CLIChannel()
    cli.attach(runner)

    handoff = Handoff(target_agent_name="Specialist", reason="Need expert")
    mock_response = AgentResponse(
        content="Handled by specialist",
        active_agent_name="Specialist",
        handoff=handoff,
    )

    def mock_dispatch(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(cli, "dispatch", mock_dispatch)

    inputs = iter(["transfer me", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    cli.run()
    captured = capsys.readouterr()
    assert "[Transferred to Specialist]" in captured.out
    assert "Reason: Need expert" in captured.out
    assert "[Specialist]: Handled by specialist" in captured.out


def test_cli_run_exception_handling(monkeypatch, capsys) -> None:
    """Test that exceptions in dispatch are caught and printed to stderr, loop continues."""
    agent = Agent(name="TestAgent", instructions="Test.")
    runner = Runner(starting_agent=agent, engine=MockEngine("ok"))
    cli = CLIChannel()
    cli.attach(runner)

    call_count = 0

    def mock_dispatch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated error")
        return AgentResponse(content="Recovered", active_agent_name="TestAgent")

    monkeypatch.setattr(cli, "dispatch", mock_dispatch)

    inputs = iter(["bad", "good", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    cli.run()
    captured = capsys.readouterr()
    assert "Error during processing: Simulated error" in captured.err
    assert "[TestAgent]: Recovered" in captured.out
    assert call_count == 2
