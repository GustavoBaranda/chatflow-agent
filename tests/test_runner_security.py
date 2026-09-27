"""Security tests for Runner error sanitization (SEC-02)."""

import logging
import re
import uuid

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Message, Role, ToolCall, ToolResult


class MockEngine(GeminiEngine):
    """Mock engine that simulates Gemini turn evaluations deterministically."""

    def __init__(self, turns_sequence: list[EngineTurnResult]) -> None:
        super().__init__()
        self.turns = list(turns_sequence)
        self.call_history: list[dict] = []

    async def generate_turn_async(
        self,
        agent: Agent,
        history: list[Message],
    ) -> EngineTurnResult:
        self.call_history.append({"agent": agent.name, "history": list(history)})
        if not self.turns:
            return EngineTurnResult(text="No more turns queued.")
        return self.turns.pop(0)


SENSITIVE_DB_URL = (
    "postgresql://superadmin:P@ssw0rd99!_SecretKey@internal-db-cluster.corp.local:5432/finance_prod"
)
SENSITIVE_INTERNAL_PATH = "/var/secrets/aws_credentials.json"


@pytest.mark.asyncio
async def test_tool_error_sanitization_masks_sensitive_data_from_llm() -> None:
    """Verify that tool execution errors do NOT leak internal sensitive details to the LLM."""
    agent = Agent(name="SecureAgent", instructions="Help with tasks.")

    @agent.tool
    def query_finance_db(query: str) -> str:
        # Simulate an internal DB error leaking credentials and paths
        raise RuntimeError(
            f"Connection refused to {SENSITIVE_DB_URL} while reading {SENSITIVE_INTERNAL_PATH}"
        )

    # Turn 1: Model requests query_finance_db
    # Turn 2: Model receives tool result (which must be sanitized) and responds to user
    mock_engine = MockEngine(
        [
            EngineTurnResult(
                tool_calls=[
                    ToolCall(
                        id="call_sec_1",
                        name="query_finance_db",
                        args={"query": "SELECT * FROM balances;"},
                    )
                ]
            ),
            EngineTurnResult(text="I encountered an issue accessing the database."),
        ]
    )

    runner = Runner(starting_agent=agent, engine=mock_engine)
    response = await runner.run_async(session_id="sec_test_user_1", user_message="Check finances")

    assert response.content == "I encountered an issue accessing the database."

    # Inspect the history sent to the model in Turn 2
    session = runner.get_session("sec_test_user_1")
    assert session is not None

    tool_messages = [m for m in session.history if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_results is not None
    assert len(tool_messages[0].tool_results) == 1

    tool_result: ToolResult = tool_messages[0].tool_results[0]
    assert tool_result.is_error is True

    # 1. Ensure NO sensitive details leak into tool_result.content
    assert SENSITIVE_DB_URL not in str(tool_result.content)
    assert "P@ssw0rd99!" not in str(tool_result.content)
    assert SENSITIVE_INTERNAL_PATH not in str(tool_result.content)
    assert "Connection refused" not in str(tool_result.content)

    # 2. Ensure NO sensitive details leak into any message in the entire session history
    for msg in session.history:
        if msg.content:
            assert "P@ssw0rd99!" not in str(msg.content)
            assert SENSITIVE_DB_URL not in str(msg.content)
            assert SENSITIVE_INTERNAL_PATH not in str(msg.content)

    # 3. Ensure generic message with trace_id format is present
    match = re.search(r"\[trace_id:\s*([a-f0-9\-]+)\]", str(tool_result.content))
    assert match is not None, f"Expected [trace_id: <uuid>] in content: {tool_result.content}"
    extracted_trace_id = match.group(1)
    # Validate that it is a valid UUID
    parsed_uuid = uuid.UUID(extracted_trace_id)
    assert str(parsed_uuid) == extracted_trace_id


@pytest.mark.asyncio
async def test_tool_error_logged_to_server_with_trace_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that full technical details and trace_id are logged on the server."""
    agent = Agent(name="LoggingAgent", instructions="Help with tasks.")

    @agent.tool
    def failing_api_call(endpoint: str) -> str:
        raise ConnectionError(f"Failed to connect to internal endpoint: {SENSITIVE_DB_URL}")

    mock_engine = MockEngine(
        [
            EngineTurnResult(
                tool_calls=[
                    ToolCall(
                        id="call_sec_2",
                        name="failing_api_call",
                        args={"endpoint": "/private/data"},
                    )
                ]
            ),
            EngineTurnResult(text="Service is currently unavailable."),
        ]
    )

    runner = Runner(starting_agent=agent, engine=mock_engine)

    with caplog.at_level(logging.ERROR, logger="chatflow_agent.runner"):
        await runner.run_async(session_id="sec_test_user_2", user_message="Call private API")

    # Find the error log record from chatflow_agent.runner
    runner_error_records = [
        r
        for r in caplog.records
        if r.name == "chatflow_agent.runner" and r.levelno == logging.ERROR
    ]
    assert len(runner_error_records) >= 1

    log_record = runner_error_records[0]

    # Verify that the server log contains the full technical error and sensitive string
    assert SENSITIVE_DB_URL in log_record.message
    assert "failing_api_call" in log_record.message
    assert "LoggingAgent" in log_record.message
    assert log_record.exc_info is not None

    # Verify that trace_id in log matches the one exposed in the tool result
    session = runner.get_session("sec_test_user_2")
    assert session is not None
    assert session.history is not None
    tool_messages = [m for m in session.history if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_results is not None
    tool_result: ToolResult = tool_messages[0].tool_results[0]

    match = re.search(r"\[trace_id:\s*([a-f0-9\-]+)\]", str(tool_result.content))
    assert match is not None
    trace_id_in_result = match.group(1)

    assert trace_id_in_result in log_record.message
