"""Tests for chatflow-agent Runner loop, tool calling, and multi-agent handoffs."""

import pytest
from typing import Any, List, Optional

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Handoff, Message, Role, ToolCall


class MockEngine(GeminiEngine):
    """Mock engine that simulates Gemini turn evaluations deterministically."""

    def __init__(self, turns_sequence: List[EngineTurnResult]) -> None:
        super().__init__()
        self.turns = list(turns_sequence)
        self.call_history: List[dict] = []

    async def generate_turn_async(
        self,
        agent: Agent,
        history: List[Message],
    ) -> EngineTurnResult:
        self.call_history.append({"agent": agent.name, "history_len": len(history)})
        if not self.turns:
            return EngineTurnResult(text="No more turns queued.")
        return self.turns.pop(0)


@pytest.mark.asyncio
async def test_runner_direct_text_response() -> None:
    agent = Agent(name="Greeter", instructions="Greet the user.")
    mock_engine = MockEngine([EngineTurnResult(text="Hello! How can I help you today?")])

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = await runner.run_async(session_id="user_1", user_message="Hi")

    assert response.content == "Hello! How can I help you today?"
    assert response.active_agent_name == "Greeter"
    assert response.handoff is None
    assert len(response.tool_calls_executed) == 0

    # Verify session history has 2 messages: user and model
    session = runner.get_session("user_1")
    assert session is not None
    assert len(session.history) == 2
    assert session.history[0].role == Role.USER
    assert session.history[1].role == Role.MODEL


@pytest.mark.asyncio
async def test_runner_with_tool_execution() -> None:
    agent = Agent(name="Calculator", instructions="Compute math.")

    @agent.tool
    def multiply(a: int, b: int) -> int:
        return a * b

    # Turn 1: Model requests tool call
    # Turn 2: Model receives tool result and produces final answer
    mock_engine = MockEngine([
        EngineTurnResult(
            tool_calls=[ToolCall(id="call_1", name="multiply", args={"a": 6, "b": 7})]
        ),
        EngineTurnResult(text="6 multiplied by 7 is 42."),
    ])

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = await runner.run_async(session_id="user_2", user_message="What is 6 * 7?")

    assert response.content == "6 multiplied by 7 is 42."
    assert "multiply" in response.tool_calls_executed
    assert response.active_agent_name == "Calculator"


@pytest.mark.asyncio
async def test_runner_multiagent_handoff_flow() -> None:
    # 1. Define specialist agent
    billing_agent = Agent(
        name="Billing Agent",
        instructions="Handle invoices and refunds.",
    )

    @billing_agent.tool
    def get_invoice(invoice_id: str) -> dict:
        return {"id": invoice_id, "amount": 100}

    # 2. Define triage agent with handoff
    triage_agent = Agent(
        name="Triage Agent",
        instructions="Route users to correct agent.",
        handoffs=[billing_agent],
    )

    # Turn 1: Triage decides to hand off to Billing Agent
    # Turn 2: Billing agent receives handoff result and calls get_invoice
    # Turn 3: Billing agent provides final answer
    mock_engine = MockEngine([
        EngineTurnResult(
            tool_calls=[
                ToolCall(
                    id="call_h1",
                    name="transfer_to_billing_agent",
                    args={"reason": "Customer invoice query"},
                )
            ]
        ),
        EngineTurnResult(
            tool_calls=[
                ToolCall(id="call_inv1", name="get_invoice", args={"invoice_id": "INV-10"})
            ]
        ),
        EngineTurnResult(text="Here is your invoice INV-10 for $100."),
    ])

    runner = Runner(starting_agent=triage_agent, engine=mock_engine)
    response = await runner.run_async(
        session_id="user_3", user_message="I need help with invoice INV-10"
    )

    # Assert active agent switched to Billing Agent!
    assert response.active_agent_name == "Billing Agent"
    assert response.content == "Here is your invoice INV-10 for $100."
    assert response.handoff is not None
    assert response.handoff.target_agent_name == "Billing Agent"
    assert response.handoff.reason == "Customer invoice query"

    # Assert session retains Billing Agent for future turns
    session = runner.get_session("user_3")
    assert session is not None
    assert session.active_agent.name == "Billing Agent"


def test_runner_sync_wrapper() -> None:
    agent = Agent(name="Echo", instructions="Echo input.")
    mock_engine = MockEngine([EngineTurnResult(text="Echo reply.")])

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = runner.run(session_id="user_4", user_message="Testing sync")

    assert response.content == "Echo reply."
    assert response.active_agent_name == "Echo"
