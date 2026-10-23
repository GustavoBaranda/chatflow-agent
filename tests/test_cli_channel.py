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
    resp = channel.dispatch(session_id="s1", message="hi")
    assert resp.content == "Welcome!"
    assert resp.active_agent_name == "Greeter"


def test_cli_channel_send_message() -> None:
    agent = Agent(name="CLI Assistant", instructions="Help.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."))

    cli = CLIChannel(session_id="custom_cli_user")
    cli.attach(runner)

    resp = cli.send_message("Run diagnostic")
    assert resp.content == "Command processed."
    assert resp.active_agent_name == "CLI Assistant"


@pytest.mark.parametrize("cmd", ["exit", "quit", "salir"])
def test_cli_run_exit_commands(monkeypatch, capsys, cmd) -> None:
    """Test clean exit on exit/quit/salir."""
    agent = Agent(name="TestAgent", instructions="Test.")
    runner = Runner(starting_agent=agent, engine=MockEngine("ok"))
    cli = CLIChannel()
    cli.attach(runner)

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

    def raise_keyboard(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_keyboard)
    cli.run()
    captured = capsys.readouterr()
    assert "Session ended by user." in captured.out
