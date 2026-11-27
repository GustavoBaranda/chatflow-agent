"""Regression for issue #240: /reset must use Runner.reset_session()."""
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner


class _Engine(GeminiEngine):
    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text="ok")


def test_reset_uses_public_runner_api(monkeypatch, capsys, tmp_path):
    stores = [SessionStore(), SQLiteSessionStore(db_path=str(tmp_path / "s.db"))]
    for store in stores:
        runner = Runner(Agent(name="A", instructions="x"), session_store=store, engine=_Engine())
        cli = CLIChannel(session_id="s1")
        cli.attach(runner)
        runner.run("s1", "seed")
        calls, dispatched = [], []
        monkeypatch.setattr(runner, "reset_session", lambda sid, real=runner.reset_session, calls=calls: (calls.append(sid), real(sid))[1])
        monkeypatch.setattr(cli, "dispatch", lambda sid, msg, dispatched=dispatched: dispatched.append(msg))
        monkeypatch.setattr("builtins.input", lambda prompt="", it=iter(["/reset", "exit"]): next(it))
        cli.run()
        assert calls == ["s1"] and dispatched == [] and "Conversation memory has been reset." in capsys.readouterr().out
        assert (session := runner.get_session("s1")) is not None and session.history == []
    stores[1].close()
