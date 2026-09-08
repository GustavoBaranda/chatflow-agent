"""Tests for chatflow-agent types and exceptions."""

from datetime import datetime, timezone

from chatflow_agent.exceptions import (
    AgentHandoffError,
    ChatFlowError,
    DependencyError,
    ProviderError,
    ToolExecutionError,
)
from chatflow_agent.types import (
    AgentResponse,
    Handoff,
    Message,
    Role,
    ToolCall,
    ToolResult,
)


def test_role_enum_values() -> None:
    assert Role.USER.value == "user"
    assert Role.MODEL.value == "model"
    assert Role.SYSTEM.value == "system"
    assert Role.TOOL.value == "tool"


def test_tool_call_serialization() -> None:
    call = ToolCall(id="call_123", name="check_balance", args={"account_id": "acc_99"})
    data = call.model_dump()
    assert data["id"] == "call_123"
    assert data["name"] == "check_balance"
    assert data["args"]["account_id"] == "acc_99"


def test_tool_result_defaults() -> None:
    res = ToolResult(tool_call_id="call_123", name="check_balance", content={"balance": 500})
    assert res.is_error is False
    assert res.content == {"balance": 500}


def test_handoff_creation() -> None:
    handoff = Handoff(
        target_agent_name="Billing Agent",
        reason="User requested refund",
        context_updates={"priority": "high"},
    )
    assert handoff.target_agent_name == "Billing Agent"
    assert handoff.reason == "User requested refund"
    assert handoff.context_updates["priority"] == "high"


def test_message_defaults_and_timestamp() -> None:
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"
    assert msg.tool_calls is None
    assert isinstance(msg.timestamp, datetime)
    assert msg.timestamp.tzinfo == timezone.utc


def test_agent_response_with_handoff() -> None:
    handoff = Handoff(target_agent_name="Tech Support")
    resp = AgentResponse(
        content="Transferring you to tech support.",
        active_agent_name="Triage Agent",
        tool_calls_executed=["lookup_ticket"],
        handoff=handoff,
    )
    assert resp.active_agent_name == "Triage Agent"
    assert resp.handoff is not None
    assert resp.handoff.target_agent_name == "Tech Support"
    assert resp.tool_calls_executed == ["lookup_ticket"]


def test_dependency_error_formatting() -> None:
    err = DependencyError(feature="WhatsAppChannel", extra_package="whatsapp")
    assert isinstance(err, ChatFlowError)
    assert "pip install \"chatflow-agent[whatsapp]\"" in str(err)
    assert err.feature == "WhatsAppChannel"
    assert err.extra_package == "whatsapp"


def test_tool_execution_error() -> None:
    original = ValueError("Division by zero")
    err = ToolExecutionError(tool_name="calculate_tax", original_error=original)
    assert isinstance(err, ChatFlowError)
    assert "calculate_tax" in str(err)
    assert "Division by zero" in str(err)


def test_agent_handoff_error() -> None:
    err = AgentHandoffError("Target agent 'Sales' not found")
    assert isinstance(err, ChatFlowError)
    assert "Target agent 'Sales' not found" in str(err)


def test_provider_error() -> None:
    err = ProviderError("Gemini quota exceeded")
    assert isinstance(err, ChatFlowError)
    assert "Gemini quota exceeded" in str(err)
