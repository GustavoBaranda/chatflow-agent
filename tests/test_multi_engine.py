"""Tests for Multi-Provider engines (OpenAI, Grok, Gemma/Ollama, Claude) and heterogeneous swarms."""

import json
from typing import Any

import httpx
import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import (
    AnthropicEngine,
    EngineTurnResult,
    GeminiEngine,
    OpenAIEngine,
    resolve_engine,
)
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Message, Role, ToolCall


class MockTransport(httpx.AsyncBaseTransport):
    """Custom in-memory transport to simulate LLM HTTP API responses."""

    def __init__(self, response_factory: Any) -> None:
        self.response_factory = response_factory
        self.requests_received: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_received.append(request)
        status_code, data = self.response_factory(request)
        return httpx.Response(
            status_code=status_code,
            content=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            request=request,
        )


@pytest.mark.asyncio
async def test_openai_compatible_engine_text_response() -> None:
    def fake_response(req: httpx.Request) -> tuple[int, dict]:
        body = json.loads(req.read().decode("utf-8"))
        assert body["model"] == "gpt-4o-mini"
        assert len(body["messages"]) >= 1
        return 200, {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from OpenAI-compatible model!",
                    }
                }
            ]
        }

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)

    engine = OpenAIEngine(
        api_key="sk-test-key",
        provider="openai",
        http_client=client,
    )
    agent = Agent(name="Assistant", model="gpt-4o-mini", instructions="Be concise.")
    result = await engine.generate_turn_async(
        agent=agent,
        history=[Message(role=Role.USER, content="Hello")],
    )

    assert result.text == "Hello from OpenAI-compatible model!"
    assert not result.has_tool_calls
    assert len(transport.requests_received) == 1
    req = transport.requests_received[0]
    assert req.url == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_gemma_ollama_engine_tool_calling() -> None:
    def fake_response(req: httpx.Request) -> tuple[int, dict]:
        body = json.loads(req.read().decode("utf-8"))
        assert body["model"] == "gemma2:9b"
        assert "tools" in body
        return 200, {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_add_1",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": json.dumps({"x": 10, "y": 20}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)

    # Local Gemma via Ollama preset
    engine = OpenAIEngine(
        provider="ollama",
        http_client=client,
    )
    agent = Agent(name="LocalGemma", model="gemma2:9b", instructions="Math helper.")

    @agent.tool
    def add(x: int, y: int) -> int:
        return x + y

    result = await engine.generate_turn_async(
        agent=agent,
        history=[Message(role=Role.USER, content="Add 10 and 20")],
    )

    assert result.has_tool_calls
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].args == {"x": 10, "y": 20}
    assert engine.base_url == "http://localhost:11434/v1"


@pytest.mark.asyncio
async def test_grok_xai_preset() -> None:
    engine = OpenAIEngine(provider="grok", api_key="xai-secret-123")
    assert engine.base_url == "https://api.x.ai/v1"
    assert engine.api_key == "xai-secret-123"


@pytest.mark.asyncio
async def test_anthropic_claude_engine() -> None:
    def fake_claude_response(req: httpx.Request) -> tuple[int, dict]:
        assert req.headers["x-api-key"] == "ant-test-key"
        assert req.headers["anthropic-version"] == "2023-06-01"
        return 200, {
            "content": [
                {"type": "text", "text": "Claude response analyzing your query."},
                {
                    "type": "tool_use",
                    "id": "tool_u1",
                    "name": "audit_record",
                    "input": {"record_id": 99},
                },
            ]
        }

    transport = MockTransport(fake_claude_response)
    client = httpx.AsyncClient(transport=transport)

    engine = AnthropicEngine(api_key="ant-test-key", http_client=client)
    agent = Agent(name="Auditor", model="claude-3-5-sonnet", instructions="Audit stuff.")

    @agent.tool
    def audit_record(record_id: int) -> str:
        return f"Audited {record_id}"

    result = await engine.generate_turn_async(
        agent=agent,
        history=[Message(role=Role.USER, content="Audit record 99")],
    )

    assert result.text == "Claude response analyzing your query."
    assert result.has_tool_calls
    assert result.tool_calls[0].name == "audit_record"
    assert result.tool_calls[0].args == {"record_id": 99}


def test_resolve_engine_heuristic() -> None:
    # 1. Gemma / Ollama
    agent_gemma = Agent(name="Gemma", provider="ollama", model="gemma2:9b", instructions="...")
    assert isinstance(resolve_engine(agent_gemma), OpenAIEngine)

    # 2. Grok
    agent_grok = Agent(name="Grok", provider="grok", model="grok-2", instructions="...")
    engine_grok = resolve_engine(agent_grok)
    assert isinstance(engine_grok, OpenAIEngine)
    assert engine_grok.base_url == "https://api.x.ai/v1"

    # 3. Claude
    agent_claude = Agent(name="Claude", provider="anthropic", model="claude-3-5-sonnet", instructions="...")
    assert isinstance(resolve_engine(agent_claude), AnthropicEngine)

    # 4. Gemini
    agent_gemini = Agent(name="Gemini", provider="gemini", model="gemini-2.5-flash", instructions="...")
    assert isinstance(resolve_engine(agent_gemini), GeminiEngine)


@pytest.mark.asyncio
async def test_heterogeneous_swarm_handoff() -> None:
    """Test multi-agent handoff where Agent A (Gemini) transfers to Agent B (Gemma local)."""
    # Specialist Agent: Gemma local
    gemma_agent = Agent(
        name="Local Specialist",
        provider="ollama",
        model="gemma2:9b",
        instructions="Specialized local processing.",
    )

    # Mock engine specifically attached to gemma_agent
    class MockGemmaEngine(OpenAIEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            return EngineTurnResult(text="Local Gemma specialist processed your query!")

    gemma_agent.engine = MockGemmaEngine(provider="ollama")

    # Frontline Agent: Triage (default engine mock)
    triage_agent = Agent(
        name="Triage",
        instructions="Route query.",
        handoffs=[gemma_agent],
    )

    class MockTriageEngine(GeminiEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            return EngineTurnResult(
                tool_calls=[
                    ToolCall(
                        id="h1",
                        name="transfer_to_local_specialist",
                        args={"reason": "Routing to local AI"},
                    )
                ]
            )

    runner = Runner(starting_agent=triage_agent, engine=MockTriageEngine())
    response = await runner.run_async(session_id="swarm_test", user_message="Process offline")

    # Verify handoff succeeded across different AI engines!
    assert response.active_agent_name == "Local Specialist"
    assert response.content == "Local Gemma specialist processed your query!"
    assert response.handoff is not None
    assert response.handoff.target_agent_name == "Local Specialist"
