"""Malformed provider responses become ProviderError, and error bodies are bounded.

Regression for #122: a 2xx response with an HTML body, a JSON array, a non-object
choice or non-object tool arguments escaped OpenAIEngine/AnthropicEngine as
JSONDecodeError, AttributeError, TypeError or ValidationError; a 4xx body was
copied whole into the ProviderError message.
"""

import httpx
import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import AnthropicEngine, OpenAIEngine
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role

TOOL_CALL_WITH_ARRAY_ARGS = {
    "choices": [
        {
            "message": {
                "tool_calls": [
                    {"id": "c", "function": {"name": "f", "arguments": "[1,2]"}}
                ]
            }
        }
    ]
}


class ScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


def _client(response: httpx.Response) -> tuple[httpx.AsyncClient, ScriptedTransport]:
    transport = ScriptedTransport(response)
    return httpx.AsyncClient(transport=transport), transport


async def _generate(engine: OpenAIEngine | AnthropicEngine) -> None:
    agent = Agent(name="A", instructions="x")
    await engine.generate_turn_async(
        agent=agent, history=[Message(role=Role.USER, content="hi")]
    )


CASES = [
    ("openai", httpx.Response(200, text="<html>login</html>")),
    ("openai", httpx.Response(200, json=[1])),
    ("openai", httpx.Response(200, json={"choices": ["not-an-object"]})),
    (
        "openai",
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c", "function": {"name": None, "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            },
        ),
    ),
    (
        "openai",
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"type": "text"}]}}]},
        ),
    ),
    ("anthropic", httpx.Response(200, text="<html>")),
    ("anthropic", httpx.Response(200, json={"content": None})),
    (
        "anthropic",
        httpx.Response(
            200, json={"content": [{"type": "tool_use", "input": [1, 2]}]}
        ),
    ),
    (
        "anthropic",
        httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "id": None, "name": None, "input": {}}
                ]
            },
        ),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,response", CASES)
async def test_malformed_response_raises_provider_error(kind: str, response: httpx.Response) -> None:
    client, _ = _client(response)
    engine = (
        OpenAIEngine(api_key="test", http_client=client)
        if kind == "openai"
        else AnthropicEngine(api_key="test", http_client=client)
    )

    with pytest.raises(ProviderError):
        await _generate(engine)


@pytest.mark.asyncio
async def test_a_2xx_shape_error_is_not_retried() -> None:
    client, transport = _client(
        httpx.Response(200, json={"choices": ["not-an-object"]})
    )
    engine = OpenAIEngine(api_key="test", http_client=client)

    with pytest.raises(ProviderError):
        await _generate(engine)

    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_error_body_is_truncated_and_has_no_request_headers() -> None:
    body = "x" * 20000
    client, _ = _client(httpx.Response(400, text=body))
    engine = OpenAIEngine(api_key="secret-key", http_client=client)

    with pytest.raises(ProviderError) as excinfo:
        await _generate(engine)

    message = str(excinfo.value)
    assert "(400)" in message
    assert len(message) <= 1100, len(message)
    assert "x" * 1000 in message
    assert "secret-key" not in message
    assert "Authorization" not in message


@pytest.mark.asyncio
async def test_non_object_tool_arguments_are_kept_as_raw_input() -> None:
    client, _ = _client(httpx.Response(200, json=TOOL_CALL_WITH_ARRAY_ARGS))
    engine = OpenAIEngine(api_key="test", http_client=client)

    agent = Agent(name="A", instructions="x")
    result = await engine.generate_turn_async(
        agent=agent, history=[Message(role=Role.USER, content="hi")]
    )

    assert result.tool_calls[0].args == {"raw_input": "[1,2]"}
