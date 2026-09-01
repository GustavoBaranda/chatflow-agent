"""Tests for chatflow-agent Agent class and synthetic handoff generation."""

import pytest
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.tools import Tool
from chatflow_agent.types import Handoff


def test_agent_initialization() -> None:
    agent = Agent(
        name="Support Agent",
        instructions="Help users with general queries.",
        model="gemini-2.5-flash",
    )
    assert agent.name == "Support Agent"
    assert agent.get_instructions() == "Help users with general queries."
    assert agent.model == "gemini-2.5-flash"
    assert len(agent.tools) == 0
    assert len(agent.handoffs) == 0


def test_agent_dynamic_instructions() -> None:
    counter = 1

    def dynamic_prompt() -> str:
        return f"Current shift #{counter}"

    agent = Agent(name="Shift Agent", instructions=dynamic_prompt)
    assert agent.get_instructions() == "Current shift #1"


def test_agent_tool_decorator_and_add_tool() -> None:
    agent = Agent(name="Finance Agent", instructions="Finance operations.")

    @agent.tool
    def get_balance(account_id: str) -> dict:
        """Fetch account balance."""
        return {"account": account_id, "balance": 1000}

    @agent.tool(name="calc_tax", description="Compute taxes")
    def calculate_tax(amount: float) -> float:
        return amount * 0.21

    assert len(agent.tools) == 2
    tool_names = [t.name for t in agent.tools]
    assert "get_balance" in tool_names
    assert "calc_tax" in tool_names


def test_agent_self_handoff_prevention() -> None:
    agent = Agent(name="Triage", instructions="Frontline triage.")
    with pytest.raises(ValueError) as exc_info:
        agent.add_handoff(agent)
    assert "cannot register a handoff to itself" in str(exc_info.value)


def test_agent_synthetic_handoff_tools() -> None:
    billing_agent = Agent(
        name="Billing Agent",
        instructions="Process payments, invoices, and card renewals.",
    )
    tech_agent = Agent(
        name="Technical Support",
        instructions="Troubleshoot servers and outages.",
    )

    triage_agent = Agent(
        name="Triage Agent",
        instructions="Frontline intake.",
        handoffs=[billing_agent],
    )
    triage_agent.add_handoff(tech_agent)

    # 1. Check handoffs count
    assert len(triage_agent.handoffs) == 2

    # 2. Check get_all_tools generates synthetic tools
    all_tools = triage_agent.get_all_tools()
    tool_names = [t.name for t in all_tools]

    assert "transfer_to_billing_agent" in tool_names
    assert "transfer_to_technical_support" in tool_names

    # 3. Test executing synthetic handoff tool returns Handoff
    billing_tool = next(t for t in all_tools if t.name == "transfer_to_billing_agent")
    result = billing_tool.execute(reason="Customer wants refund")

    assert isinstance(result, Handoff)
    assert result.target_agent_name == "Billing Agent"
    assert result.reason == "Customer wants refund"
