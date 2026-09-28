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
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall


class MockTransport(httpx.AsyncBaseTransport):
    """Custom in-memory transport to simulate LLM HTTP API responses."""

    def __init__(self, response_factory: Any) -> None:
        self.response_factory = response_factory
        self.requests_received: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_received.append(request)
        res = self.response_factory(request)
        if len(res) == 3:
            status_code, data, custom_headers = res
            headers = {"Content-Type": "application/json", **custom_headers}
        else:
            status_code, data = res
            headers = {"Content-Type": "application/json"}
        return httpx.Response(
            status_code=status_code,
            content=json.dumps(data).encode("utf-8"),
            headers=headers,
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


def test_custom_base_url_rejects_fallback_to_global_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure custom/unofficial base_url raises ValueError instead of leaking OPENAI_API_KEY (SEC-07)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-prod-key-12345")

    # 1. Custom remote endpoint without api_key must raise ValueError
    with pytest.raises(ValueError, match="A custom or unofficial base_url was provided"):
        OpenAIEngine(base_url="https://untrusted-proxy.example.com/v1")

    # 2. Custom remote endpoint with explicit api_key succeeds
    engine_with_key = OpenAIEngine(
        base_url="https://untrusted-proxy.example.com/v1",
        api_key="custom-proxy-key",
    )
    assert engine_with_key.api_key == "custom-proxy-key"

    # 3. Custom remote endpoint with explicit empty api_key (unauthenticated) succeeds
    engine_unauth = OpenAIEngine(
        base_url="https://untrusted-proxy.example.com/v1",
        api_key="",
    )
    assert engine_unauth.api_key == ""

    # 4. Localhost base_url does NOT leak global OPENAI_API_KEY
    engine_local = OpenAIEngine(base_url="http://localhost:8000/v1")
    assert engine_local.api_key != "sk-secret-prod-key-12345"
    assert engine_local.api_key == "ollama"

    # 5. Official OpenAI base_url STILL falls back to OPENAI_API_KEY as expected
    engine_official = OpenAIEngine()
    assert engine_official.api_key == "sk-secret-prod-key-12345"


@pytest.mark.asyncio
async def test_anthropic_retry_on_529_overloaded() -> None:
    """Test AnthropicEngine retries on HTTP 529 Overloaded and succeeds."""
    attempts = 0

    def fake_response(req: httpx.Request) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return 529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        return 200, {
            "content": [{"type": "text", "text": "Recovered from 529"}]
        }

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)
    engine = AnthropicEngine(api_key="ant-test-key", http_client=client)
    engine.base_delay = 0.001
    engine.max_delay = 0.01

    agent = Agent(name="Claude", model="claude-3-5-sonnet", instructions="Helpful Claude.")
    result = await engine.generate_turn_async(
        agent=agent,
        history=[Message(role=Role.USER, content="Hello")],
    )

    assert result.text == "Recovered from 529"
    assert attempts == 2
    assert len(transport.requests_received) == 2


@pytest.mark.asyncio
async def test_anthropic_retry_with_retry_after_header() -> None:
    """Test AnthropicEngine honors Retry-After header on 429."""
    attempts = 0

    def fake_response(req: httpx.Request) -> tuple[int, dict, dict[str, str]] | tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return 429, {"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}}, {"Retry-After": "0.01"}
        return 200, {
            "content": [{"type": "text", "text": "Recovered after rate limit"}]
        }

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)
    engine = AnthropicEngine(api_key="ant-test-key", http_client=client)
    engine.base_delay = 0.001
    engine.max_delay = 0.05

    agent = Agent(name="Claude", model="claude-3-5-sonnet", instructions="Helpful Claude.")
    result = await engine.generate_turn_async(
        agent=agent,
        history=[Message(role=Role.USER, content="Hello")],
    )

    assert result.text == "Recovered after rate limit"
    assert attempts == 2
    assert len(transport.requests_received) == 2


@pytest.mark.asyncio
async def test_anthropic_retry_on_connect_error() -> None:
    """Test AnthropicEngine retries on httpx.ConnectError."""
    attempts = 0

    def fake_response(req: httpx.Request) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("Connection refused", request=req)
        return 200, {
            "content": [{"type": "text", "text": "Connected on retry"}]
        }

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)
    engine = AnthropicEngine(api_key="ant-test-key", http_client=client)
    engine.base_delay = 0.001
    engine.max_delay = 0.01

    agent = Agent(name="Claude", model="claude-3-5-sonnet", instructions="Helpful Claude.")
    result = await engine.generate_turn_async(
        agent=agent,
        history=[Message(role=Role.USER, content="Hello")],
    )

    assert result.text == "Connected on retry"
    assert attempts == 2
    assert len(transport.requests_received) == 2


@pytest.mark.asyncio
async def test_anthropic_max_retries_exhausted_raises_provider_error() -> None:
    """Test AnthropicEngine raises ProviderError after max retries are exhausted."""
    attempts = 0

    def fake_response(req: httpx.Request) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        return 529, {"type": "error", "error": {"type": "overloaded_error", "message": "Anthropic is overloaded"}}

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)
    engine = AnthropicEngine(api_key="ant-test-key", http_client=client)
    engine.max_retries = 2
    engine.base_delay = 0.001
    engine.max_delay = 0.01

    agent = Agent(name="Claude", model="claude-3-5-sonnet", instructions="Helpful Claude.")
    with pytest.raises(ProviderError, match="Anthropic API error \\(529\\)"):
        await engine.generate_turn_async(
            agent=agent,
            history=[Message(role=Role.USER, content="Hello")],
        )

    assert attempts == 3  # initial + 2 retries
    assert len(transport.requests_received) == 3


@pytest.mark.asyncio
async def test_anthropic_non_retryable_error_does_not_retry() -> None:
    """Test non-transient 4xx errors (e.g. 401 Unauthorized) fail immediately."""
    attempts = 0

    def fake_response(req: httpx.Request) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        return 401, {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API Key"}}

    transport = MockTransport(fake_response)
    client = httpx.AsyncClient(transport=transport)
    engine = AnthropicEngine(api_key="ant-test-key", http_client=client)
    engine.max_retries = 3
    engine.base_delay = 0.001

    agent = Agent(name="Claude", model="claude-3-5-sonnet", instructions="Helpful Claude.")
    with pytest.raises(ProviderError, match="Anthropic API error \\(401\\)"):
        await engine.generate_turn_async(
            agent=agent,
            history=[Message(role=Role.USER, content="Hello")],
        )

    assert attempts == 1  # No retries on 401
    assert len(transport.requests_received) == 1


