"""Tests for chatflow-agent Agent class and synthetic handoff generation."""

import re

import pytest

from chatflow_agent.core.agent import Agent, _sanitize_tool_name
from chatflow_agent.types import Handoff

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


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


def test_agent_user_tool_takes_precedence_over_synthetic_handoff() -> None:
    billing_agent = Agent(name="Billing", instructions="Handle invoices.")
    tech_agent = Agent(name="Technical Support", instructions="Troubleshoot issues.")
    agent = Agent(name="Triage", instructions="Route users.", handoffs=[billing_agent, tech_agent])

    @agent.tool
    def transfer_to_billing(account_id: str) -> str:
        return f"Custom billing route for {account_id}"

    for _ in range(2):
        all_tools = agent.get_all_tools()
        names = [t.name for t in all_tools]
        assert names[0] == "transfer_to_billing"
        assert all_tools[0] is transfer_to_billing
        assert all_tools[0].execute(account_id="demo-1") == "Custom billing route for demo-1"
        assert all_tools[0].parameters_schema["required"] == ["account_id"]
        # User tool keeps the preferred name; handoffs are disambiguated and preserved.
        assert "transfer_to_technical_support" in names
        assert len(all_tools) == 3
        assert len(set(names)) == 3

    assert agent.tools == [transfer_to_billing]
    assert agent.handoffs == [billing_agent, tech_agent]


def test_agent_sanitized_handoff_name_collision_disambiguates_peers() -> None:
    billing_agent = Agent(name="Billing Team", instructions="Handle invoices.")
    other_billing_agent = Agent(name="billing-team", instructions="Handle subscriptions.")
    agent = Agent(
        name="Triage",
        instructions="Route users.",
        handoffs=[billing_agent, other_billing_agent],
    )

    all_tools = agent.get_all_tools()
    tool_names = [t.name for t in all_tools]

    assert len(all_tools) == 2
    assert len(set(tool_names)) == 2
    assert tool_names[0] == "transfer_to_billing_team"
    assert tool_names[1].startswith("transfer_to_billing_team")
    assert all(_TOOL_NAME_RE.fullmatch(n) and len(n) <= 60 for n in tool_names)

    result = all_tools[0].execute(reason="Invoice query")
    assert isinstance(result, Handoff)
    assert result.target_agent_name == "Billing Team"
    assert result.reason == "Invoice query"

    other = all_tools[1].execute(reason="Subscription query")
    assert isinstance(other, Handoff)
    assert other.target_agent_name == "billing-team"


def test_sanitize_accented_latin_transliteration() -> None:
    assert _sanitize_tool_name("Atención al Cliente") == "transfer_to_atencion_al_cliente"
    assert _sanitize_tool_name("Soporte Técnico") == "transfer_to_soporte_tecnico"


def test_sanitize_ascii_names_unchanged() -> None:
    assert _sanitize_tool_name("Billing Agent") == "transfer_to_billing_agent"
    assert _sanitize_tool_name("Technical Support") == "transfer_to_technical_support"


@pytest.mark.parametrize(
    "name",
    ["客户服务", "Техподдержка", "خدمة العملاء", "🤖", "销售专员"],
)
def test_sanitize_non_latin_produces_deterministic_valid_name(name: str) -> None:
    first = _sanitize_tool_name(name)
    second = _sanitize_tool_name(name)
    assert first == second
    assert first.startswith("transfer_to_")
    assert first != "transfer_to_"
    assert _TOOL_NAME_RE.fullmatch(first)
    assert len(first) <= 60


def test_unicode_handoffs_are_all_preserved_without_collision() -> None:
    triage = Agent("triage", "frontline")
    support = Agent("客户服务", "support instructions")
    sales = Agent("销售专员", "sales instructions")
    triage.add_handoff(support)
    triage.add_handoff(sales)

    tools = triage.get_all_tools()
    names = [t.name for t in tools]

    assert len(tools) == 2
    assert len(set(names)) == 2
    assert "transfer_to_" not in names
    assert all(_TOOL_NAME_RE.fullmatch(n) and len(n) <= 60 for n in names)

    targets = {t.execute(reason="route").target_agent_name for t in tools}
    assert targets == {"客户服务", "销售专员"}


def test_user_tool_name_collision_still_preserves_handoff() -> None:
    billing_agent = Agent(name="Billing", instructions="Handle invoices.")
    tech_agent = Agent(name="Technical Support", instructions="Troubleshoot issues.")
    agent = Agent(name="Triage", instructions="Route users.", handoffs=[billing_agent, tech_agent])

    @agent.tool
    def transfer_to_billing(account_id: str) -> str:
        return f"Custom billing route for {account_id}"

    all_tools = agent.get_all_tools()
    names = [t.name for t in all_tools]

    assert names[0] == "transfer_to_billing"
    assert all_tools[0] is transfer_to_billing
    assert "transfer_to_technical_support" in names
    # Billing handoff must still appear under a disambiguated name.
    handoff_names = [n for n in names if n != "transfer_to_billing"]
    assert len(handoff_names) == 2
    assert len(set(names)) == 3


def test_sanitize_tool_name_respects_length_limit() -> None:
    long_name = "A" * 80
    sanitized = _sanitize_tool_name(long_name)
    assert len(sanitized) <= 60
    assert _TOOL_NAME_RE.fullmatch(sanitized)


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_agent_rejects_empty_or_whitespace_name(name: str) -> None:
    with pytest.raises(ValueError, match="Agent name must be a non-empty string"):
        Agent(name=name, instructions="Help user")
