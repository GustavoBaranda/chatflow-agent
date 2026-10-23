"""Provider response failures stay catchable without retrying completed generations."""

import json
from typing import Any

import httpx
import pytest

from chatflow_agent import Agent, AnthropicEngine, OpenAIEngine, ProviderError, Runner
from chatflow_agent.types import Message, Role

EngineType = type[OpenAIEngine] | type[AnthropicEngine]
ENGINES = [pytest.param(OpenAIEngine, id="openai"), pytest.param(AnthropicEngine, id="anthropic")]
RETRYABLE_STATUSES = [429, 500, 502, 503, 504]
TEST_API_KEY = "test-provider-response-key"


def _engine(engine_type: EngineType, client: httpx.AsyncClient) -> OpenAIEngine | AnthropicEngine:
    engine = engine_type(api_key=TEST_API_KEY, http_client=client)
    engine.base_delay = 0
    engine.max_delay = 0
    return engine


def _openai_tool_response(arguments: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "record", "arguments": arguments},
                        }
                    ]
                }
            }
        ],
    }


def _anthropic_tool_response(arguments: Any) -> dict[str, Any]:
    return {"content": [{"type": "tool_use", "id": "call_1", "name": "record", "input": arguments}]}


def _text_response(engine_type: EngineType, text: str) -> dict[str, Any]:
    if engine_type is OpenAIEngine:
        return {"choices": [{"message": {"content": text}}]}
    return {"content": [{"type": "text", "text": text}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_type", ENGINES)
@pytest.mark.parametrize(
    "body", ["<html>login</html>", "[1]", "null"], ids=["html", "array", "null"]
)
async def test_malformed_response_raises_provider_error_without_retry(
    engine_type: EngineType,
    body: str,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        engine = _engine(engine_type, client)
        with pytest.raises(ProviderError) as error:
            await engine.generate_turn_async(
                Agent(name="Assistant", instructions="Be helpful."),
                [Message(role=Role.USER, content="Hi")],
            )

    assert error.value.__cause__ is not None
    assert not isinstance(error.value.__cause__, ProviderError)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_type", "payload"),
    [
        pytest.param(OpenAIEngine, {"choices": [None]}, id="openai-choice-null"),
        pytest.param(OpenAIEngine, {"choices": [1]}, id="openai-choice-scalar"),
        pytest.param(OpenAIEngine, {"choices": {"wrong": {}}}, id="openai-choices-object"),
        pytest.param(OpenAIEngine, {"choices": [{"message": None}]}, id="openai-message-null"),
        pytest.param(
            OpenAIEngine,
            {"choices": [{"message": {"content": ["invalid"]}}]},
            id="openai-text-list",
        ),
        pytest.param(
            OpenAIEngine, {"choices": [{"message": {"tool_calls": [None]}}]}, id="openai-tool-null"
        ),
        pytest.param(
            OpenAIEngine,
            {"choices": [{"message": {"tool_calls": [{"function": None}]}}]},
            id="openai-function-null",
        ),
        pytest.param(OpenAIEngine, _openai_tool_response("[1,2]"), id="openai-args-array"),
        pytest.param(OpenAIEngine, _openai_tool_response("null"), id="openai-args-null"),
        pytest.param(AnthropicEngine, {"content": None}, id="anthropic-content-null"),
        pytest.param(AnthropicEngine, {"content": [None]}, id="anthropic-block-null"),
        pytest.param(
            AnthropicEngine, {"content": [{"type": "text", "text": None}]}, id="anthropic-text-null"
        ),
        pytest.param(
            AnthropicEngine,
            {"content": [{"type": "text", "text": ["invalid"]}]},
            id="anthropic-text-list",
        ),
        pytest.param(AnthropicEngine, _anthropic_tool_response([1, 2]), id="anthropic-input-array"),
        pytest.param(AnthropicEngine, _anthropic_tool_response(None), id="anthropic-input-null"),
    ],
)
async def test_malformed_response_fields_raise_chained_provider_error(
    engine_type: EngineType,
    payload: dict[str, Any],
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        engine = _engine(engine_type, client)
        with pytest.raises(ProviderError) as error:
            await engine.generate_turn_async(
                Agent(name="Assistant", instructions="Be helpful."),
                [Message(role=Role.USER, content="Hi")],
            )

    assert error.value.__cause__ is not None
    assert not isinstance(error.value.__cause__, ProviderError)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_type", ENGINES)
async def test_empty_response_remains_valid(engine_type: EngineType) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    ) as client:
        result = await _engine(engine_type, client).generate_turn_async(
            Agent(name="Assistant", instructions="Be helpful."), []
        )

    assert result.text == ("" if engine_type is OpenAIEngine else None)
    assert result.tool_calls == []
    assert result.raw_response == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param('{"value": 7}', {"value": 7}, id="json-object"),
        pytest.param({"value": 7}, {"value": 7}, id="object"),
        pytest.param("{}", {}, id="empty-object"),
        pytest.param("  ", {}, id="blank"),
        pytest.param("{invalid", {"raw_input": "{invalid"}, id="invalid-json"),
    ],
)
async def test_openai_supported_tool_arguments_are_preserved(
    arguments: Any,
    expected: dict[str, Any],
) -> None:
    payload = _openai_tool_response(arguments)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        result = await _engine(OpenAIEngine, client).generate_turn_async(
            Agent(name="Assistant", instructions="Be helpful."), []
        )

    assert result.text is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "record"
    assert result.tool_calls[0].args == expected
    assert result.raw_response == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{}, {"value": 7}], ids=["empty-object", "object"])
async def test_anthropic_text_and_tool_arguments_are_preserved(arguments: dict[str, Any]) -> None:
    payload = _anthropic_tool_response(arguments)
    payload["content"][:0] = [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        result = await _engine(AnthropicEngine, client).generate_turn_async(
            Agent(name="Assistant", instructions="Be helpful."), []
        )

    assert result.text == "Hello world"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "record"
    assert result.tool_calls[0].args == arguments
    assert result.raw_response == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_type", "status"),
    [
        pytest.param(engine_type, status, id=f"{engine_type.__name__}-{status}")
        for engine_type in (OpenAIEngine, AnthropicEngine)
        for status in (400, 401, *RETRYABLE_STATUSES)
    ]
    + [pytest.param(AnthropicEngine, 529, id="anthropic-529")],
)
async def test_http_error_body_is_bounded_and_retry_count_is_preserved(
    engine_type: EngineType,
    status: int,
) -> None:
    body = "Provider failure: " + "x" * 20000 + "END-OF-BODY"
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            text=body,
            headers={"Retry-After": "0", "x-response-secret": "response-header-value"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), headers={"x-request-secret": "request-header-value"}
    ) as client:
        engine = _engine(engine_type, client)
        with pytest.raises(ProviderError) as error:
            await engine.generate_turn_async(
                Agent(name="Assistant", instructions="Be helpful."), []
            )

    message = str(error.value)
    assert str(status) in message
    assert body[:1000] in message
    assert body[:1001] not in message
    assert "END-OF-BODY" not in message
    assert len(message) <= 1100
    assert TEST_API_KEY not in message
    assert "authorization" not in message.lower()
    assert "x-api-key" not in message.lower()
    assert "x-request-secret" not in message
    assert "request-header-value" not in message
    assert "response-header-value" not in message
    expected_requests = engine.max_retries + 1 if status in (*RETRYABLE_STATUSES, 529) else 1
    assert len(requests) == expected_requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_type", "status"),
    [
        pytest.param(engine_type, status, id=f"{engine_type.__name__}-{status}")
        for engine_type in (OpenAIEngine, AnthropicEngine)
        for status in RETRYABLE_STATUSES
    ]
    + [pytest.param(AnthropicEngine, 529, id="anthropic-529")],
)
async def test_retryable_http_error_can_still_recover(engine_type: EngineType, status: int) -> None:
    requests: list[httpx.Request] = []
    payload = _text_response(engine_type, "Recovered")

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, text="Temporary failure", headers={"Retry-After": "0"})
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _engine(engine_type, client).generate_turn_async(
            Agent(name="Assistant", instructions="Be helpful."), []
        )

    assert len(requests) == 2
    assert result.text == "Recovered"
    assert result.tool_calls == []
    assert result.raw_response == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_type", ENGINES)
async def test_runner_catches_malformed_response_and_restores_history(
    engine_type: EngineType,
) -> None:
    requests: list[httpx.Request] = []
    executed: list[int] = []
    agent = Agent(name="Assistant", instructions="Be helpful.")

    @agent.tool
    def record(value: int) -> str:
        executed.append(value)
        return str(value)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=_text_response(engine_type, "First reply"))
        # A valid call before the malformed one must never be partly executed.
        if engine_type is OpenAIEngine:
            payload = _openai_tool_response('{"value": 7}')
            payload["choices"][0]["message"]["tool_calls"].append(None)
        else:
            payload = _anthropic_tool_response({"value": 7})
            payload["content"].append(None)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        runner = Runner(starting_agent=agent, engine=_engine(engine_type, client))
        response = await runner.run_async("session", "Hello")
        assert response.content == "First reply"
        session = runner.get_session("session")
        assert session is not None
        previous_history = [message.model_dump() for message in session.history]
        with pytest.raises(ProviderError) as error:
            await runner.run_async("session", "Record 7")

    assert error.value.__cause__ is not None
    assert len(requests) == 2
    assert executed == []
    assert [message.model_dump() for message in session.history] == previous_history


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_type", ENGINES)
async def test_runner_can_execute_valid_provider_tool_call(engine_type: EngineType) -> None:
    requests: list[httpx.Request] = []
    executed: list[int] = []
    agent = Agent(name="Assistant", instructions="Be helpful.")

    @agent.tool
    def record(value: int) -> str:
        executed.append(value)
        return f"Recorded {value}"

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            payload = (
                _openai_tool_response('{"value": 7}')
                if engine_type is OpenAIEngine
                else _anthropic_tool_response({"value": 7})
            )
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json=_text_response(engine_type, "Done"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        runner = Runner(starting_agent=agent, engine=_engine(engine_type, client))
        response = await runner.run_async("session", "Record 7")

    assert response.content == "Done"
    assert response.tool_calls_executed == ["record"]
    assert executed == [7]
    assert len(requests) == 2
    # The second actual HTTP request must carry the tool result back to the provider.
    sent = json.loads(requests[1].content)
    if engine_type is OpenAIEngine:
        assert sent["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "record",
            "content": "Recorded 7",
        }
    else:
        assert sent["messages"][-1] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "Recorded 7",
                    "is_error": False,
                }
            ],
        }
