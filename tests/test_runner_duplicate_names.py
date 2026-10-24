"""Regression tests for duplicate agent-name handoff resolution (#123)."""

from pathlib import Path

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult
from chatflow_agent.core.engines import BaseEngine
from chatflow_agent.core.memory import SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import ToolCall
from tests.test_runner import MockEngine


def refund() -> str:
    """Issue a refund."""
    return "refunded"


def reboot() -> str:
    """Reboot a server."""
    return "rebooted"


class NameRouter(BaseEngine):
    """Deterministic engine that follows a name-to-handoff map, then reports tools."""

    def __init__(self, steps: dict[str, str]) -> None:
        self._steps = steps

    async def generate_turn_async(self, agent: Agent, history: list) -> EngineTurnResult:
        step = self._steps.get(agent.name)
        if step and not any(
            message.tool_calls and message.tool_calls[0].name == step for message in history
        ):
            return EngineTurnResult(tool_calls=[ToolCall(id=step, name=step, args={})])
        return EngineTurnResult(
            text=f"{agent.get_instructions()} with tools {[tool.name for tool in agent.tools]}"
        )


def test_runner_rejects_duplicate_names_at_construction() -> None:
    # Same-name peers must sit on different parents: Agent.add_handoff() collapses
    # duplicates on a single agent, so Root(handoffs=[left, right]) would keep only one.
    left_support = Agent(name="Support", instructions="LEFT")
    right_support = Agent(name="Support", instructions="RIGHT")
    left = Agent(name="Left desk", instructions="left", handoffs=[left_support])
    right = Agent(name="Right desk", instructions="right", handoffs=[right_support])
    root = Agent(name="Root", instructions="root", handoffs=[left, right])
    with pytest.raises(ValueError, match="Duplicate agent name 'Support'"):
        Runner(starting_agent=root, engine=MockEngine([]))


def test_runner_allows_shared_instance_in_dag() -> None:
    shared = Agent(name="Support", instructions="SHARED", tools=[refund])
    left = Agent(name="Left desk", instructions="left", handoffs=[shared])
    right = Agent(name="Right desk", instructions="right", handoffs=[shared])
    root = Agent(name="Root", instructions="root", handoffs=[left, right])
    runner = Runner(starting_agent=root, engine=MockEngine([]))
    assert runner._agents_registry["Support"] is shared


def test_register_agent_rejects_conflicting_name() -> None:
    root = Agent(name="Root", instructions="root")
    runner = Runner(starting_agent=root, engine=MockEngine([]))
    first = Agent(name="Support", instructions="FIRST")
    second = Agent(name="Support", instructions="SECOND")
    runner.register_agent(first)
    with pytest.raises(ValueError, match="Duplicate agent name 'Support'"):
        runner.register_agent(second)
    runner.register_agent(first)  # same instance is a no-op


def test_issue_123_reproducer_raises_at_construction() -> None:
    """The issue reproducer tree is a configuration error after the fix."""
    billing_support = Agent(name="Support", instructions="BILLING support", tools=[refund])
    tech_support = Agent(name="Support", instructions="TECH support", tools=[reboot])
    billing_desk = Agent(name="Billing desk", instructions="billing", handoffs=[billing_support])
    tech_desk = Agent(name="Tech desk", instructions="tech", handoffs=[tech_support])
    root = Agent(name="Root", instructions="root", handoffs=[billing_desk, tech_desk])
    with pytest.raises(ValueError, match="Duplicate agent name 'Support'"):
        Runner(
            starting_agent=root,
            engine=NameRouter(
                {"Root": "transfer_to_tech_desk", "Tech desk": "transfer_to_support"}
            ),
        )


@pytest.mark.asyncio
async def test_late_add_handoff_replacement_updates_registry() -> None:
    old_support = Agent(name="Support", instructions="OLD support", tools=[refund])
    new_support = Agent(name="Support", instructions="NEW support", tools=[reboot])
    desk = Agent(name="Desk", instructions="desk", handoffs=[old_support])
    root = Agent(name="Root", instructions="root", handoffs=[desk])
    engine = NameRouter({"Root": "transfer_to_desk", "Desk": "transfer_to_support"})
    runner = Runner(starting_agent=root, engine=engine)
    assert runner._agents_registry["Support"] is old_support

    desk.add_handoff(new_support)
    reply = await runner.run_async("s", "need tech")
    assert runner._agents_registry["Support"] is new_support
    assert reply.content == "NEW support with tools ['reboot']"
    assert reply.active_agent_name == "Support"


@pytest.mark.asyncio
async def test_late_replacement_conflicts_when_old_still_reachable() -> None:
    old_support = Agent(name="Support", instructions="OLD", tools=[refund])
    new_support = Agent(name="Support", instructions="NEW", tools=[reboot])
    left = Agent(name="Left", instructions="left", handoffs=[old_support])
    right = Agent(name="Right", instructions="right", handoffs=[old_support])
    root = Agent(name="Root", instructions="root", handoffs=[left, right])
    runner = Runner(starting_agent=root, engine=MockEngine([]))

    right.add_handoff(new_support)
    with pytest.raises(ValueError, match="Duplicate agent name 'Support'"):
        runner._resolve_handoff_target(right, "Support")


@pytest.mark.asyncio
async def test_sqlite_session_resolves_replaced_active_agent(tmp_path: Path) -> None:
    """Existing SQLite session uses the new peer after late replacement (real turns)."""
    old_support = Agent(name="Support", instructions="OLD support", tools=[refund])
    new_support = Agent(name="Support", instructions="NEW support", tools=[reboot])
    desk = Agent(name="Desk", instructions="desk", handoffs=[old_support])
    root = Agent(name="Root", instructions="root", handoffs=[desk])
    db_path = tmp_path / "dup.db"
    store = SQLiteSessionStore(str(db_path))
    engine = NameRouter({"Root": "transfer_to_desk", "Desk": "transfer_to_support"})
    runner = Runner(starting_agent=root, session_store=store, engine=engine)

    first = await runner.run_async("s1", "go")
    assert first.active_agent_name == "Support"
    assert first.content == "OLD support with tools ['refund']"
    assert runner.get_session("s1").active_agent is old_support

    desk.add_handoff(new_support)
    # Another real turn performs the late-replacement handoff (not a private helper).
    other = await runner.run_async("s2", "go")
    assert other.content == "NEW support with tools ['reboot']"
    assert runner._agents_registry["Support"] is new_support

    # Existing SQLite session re-resolves active agent by name on the next turn.
    second = await runner.run_async("s1", "still broken")
    assert second.active_agent_name == "Support"
    assert second.content == "NEW support with tools ['reboot']"
    reloaded = store.get_or_create("s1", default_agent=root)
    assert reloaded.active_agent is new_support


def test_register_agent_tree_is_atomic_on_deep_duplicate() -> None:
    root = Agent(name="Root", instructions="root")
    runner = Runner(starting_agent=root, engine=MockEngine([]))
    # Pre-register a conflicting leaf name.
    runner.register_agent(Agent(name="Leaf", instructions="existing"))
    before = dict(runner._agents_registry)

    branch = Agent(name="Branch", instructions="branch")
    mid = Agent(name="Mid", instructions="mid", handoffs=[branch])
    conflict = Agent(name="Leaf", instructions="conflict")
    branch.add_handoff(conflict)

    with pytest.raises(ValueError, match="Duplicate agent name 'Leaf'"):
        runner.register_agent(mid)

    assert runner._agents_registry == before
    assert "Branch" not in runner._agents_registry
    assert "Mid" not in runner._agents_registry


@pytest.mark.asyncio
async def test_late_replacement_validation_failure_leaves_registry_unchanged() -> None:
    old_support = Agent(name="Support", instructions="OLD", tools=[refund])
    desk = Agent(name="Desk", instructions="desk", handoffs=[old_support])
    root = Agent(name="Root", instructions="root", handoffs=[desk])
    runner = Runner(starting_agent=root, engine=MockEngine([]))
    before = dict(runner._agents_registry)

    # New Support brings a child that collides with an existing agent name.
    conflict_child = Agent(name="Desk", instructions="should conflict")
    new_support = Agent(
        name="Support", instructions="NEW", tools=[reboot], handoffs=[conflict_child]
    )
    desk.add_handoff(new_support)

    with pytest.raises(ValueError, match="Duplicate agent name 'Desk'"):
        runner._resolve_handoff_target(desk, "Support")

    assert runner._agents_registry == before
    assert runner._agents_registry["Support"] is old_support


def test_late_replacement_rejects_manually_registered_agent() -> None:
    root = Agent(name="Root", instructions="root")
    runner = Runner(starting_agent=root, engine=MockEngine([]))
    manual = Agent(name="Support", instructions="MANUAL", tools=[refund])
    runner.register_agent(manual)
    assert "Support" in runner._manual_agent_names

    desk = Agent(name="Desk", instructions="desk")
    root.add_handoff(desk)
    runner.register_agent(desk)
    replacement = Agent(name="Support", instructions="LATE", tools=[reboot])
    desk.add_handoff(replacement)

    with pytest.raises(ValueError, match="manually registered agent 'Support'"):
        runner._resolve_handoff_target(desk, "Support")
    assert runner._agents_registry["Support"] is manual
