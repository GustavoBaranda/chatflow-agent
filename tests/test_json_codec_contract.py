"""Consistent JSON encoding for tool results and metadata across engines/stores (#167)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import AnthropicEngine, BaseEngine, EngineTurnResult, OpenAIEngine
from chatflow_agent.core.json_codec import encode_tool_content, to_jsonable
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall, ToolResult

WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class _Probe(BaseModel):
    label: str


class _Odd:
    def __str__(self) -> str:
        return "odd-object"


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


@pytest.mark.parametrize(
    "value, expected",
    [
        (WHEN, "2026-01-02T03:04:05+00:00"),
        (_Probe(label="x"), {"label": "x"}),
        (_Odd(), "odd-object"),
        ({"when": WHEN}, {"when": "2026-01-02T03:04:05+00:00"}),
    ],
)
def test_to_jsonable_coerces(value, expected):
    assert to_jsonable(value) == expected


def test_encode_tool_content_matches_across_shapes():
    payload = {"when": WHEN, "probe": _Probe(label="ok")}
    encoded = encode_tool_content(payload)
    assert json.loads(encoded) == {
        "when": "2026-01-02T03:04:05+00:00",
        "probe": {"label": "ok"},
    }


@pytest.mark.parametrize("engine_name", ["openai", "anthropic"])
@pytest.mark.asyncio
async def test_engines_encode_datetime_tool_results(engine_name: str):
    history = [
        Message(role=Role.USER, content="hi"),
        Message(
            role=Role.MODEL,
            content=None,
            tool_calls=[ToolCall(id="c1", name="when", args={})],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[
                ToolResult(
                    tool_call_id="c1",
                    name="when",
                    content={"when": WHEN},
                )
            ],
        ),
    ]
    if engine_name == "openai":
        response = httpx.Response(
            200, json={"choices": [{"message": {"content": "done"}}]}
        )
        client, transport = _client(response)
        engine = OpenAIEngine(api_key="k", http_client=client)
        agent = Agent(name="A", instructions="x", model="gpt-4o-mini")
        try:
            await engine.generate_turn_async(agent=agent, history=history)
            body = json.loads(transport.requests[-1].content.decode())
            tool_msg = next(m for m in body["messages"] if m.get("role") == "tool")
            assert json.loads(tool_msg["content"]) == {
                "when": "2026-01-02T03:04:05+00:00"
            }
        finally:
            await client.aclose()
    else:
        response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
            },
        )
        client, transport = _client(response)
        engine = AnthropicEngine(api_key="k", http_client=client)
        agent = Agent(name="A", instructions="x", model="claude-3-5-haiku-latest")
        try:
            await engine.generate_turn_async(agent=agent, history=history)
            body = json.loads(transport.requests[-1].content.decode())
            tool_block = None
            for msg in body["messages"]:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_block = block
                        break
                if tool_block is not None:
                    break
            assert tool_block is not None
            assert json.loads(tool_block["content"]) == {
                "when": "2026-01-02T03:04:05+00:00"
            }
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_openai_non_string_tool_call_id_is_provider_error():
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": 123,
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        },
    )
    client, _transport = _client(response)
    engine = OpenAIEngine(api_key="k", http_client=client)
    agent = Agent(name="A", instructions="x", model="gpt-4o-mini")
    try:
        with pytest.raises(ProviderError, match="tool call's id is not a string"):
            await engine.generate_turn_async(
                agent=agent, history=[Message(role=Role.USER, content="hi")]
            )
    finally:
        await client.aclose()


class _EchoEngine(BaseEngine):
    async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
        return EngineTurnResult(text="ok")


@pytest.mark.parametrize("store_kind", ["memory", "sqlite_file", "sqlite_memory"])
@pytest.mark.asyncio
async def test_metadata_datetime_works_across_stores(store_kind: str, tmp_path: Path):
    agent = Agent(name="A", instructions="x")
    if store_kind == "memory":
        store: SessionStore | SQLiteSessionStore = SessionStore()
    elif store_kind == "sqlite_memory":
        store = SQLiteSessionStore(":memory:")
    else:
        store = SQLiteSessionStore(tmp_path / "s.db")

    runner = Runner(starting_agent=agent, session_store=store, engine=_EchoEngine())
    try:
        result = await runner.run_async("s1", "hi", metadata={"at": WHEN})
        assert result.content == "ok"
        session = store.get("s1")
        assert session is not None
        assert session.metadata["at"] in (WHEN, "2026-01-02T03:04:05+00:00")
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite_file", "sqlite_memory"])
@pytest.mark.asyncio
async def test_tool_returning_datetime_and_object_across_stores(
    store_kind: str, tmp_path: Path
):
    agent = Agent(name="A", instructions="x", model="gpt-4o-mini")

    @agent.tool
    def stamp() -> dict:
        return {"when": WHEN, "odd": _Odd()}

    payloads = [
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "stamp", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]

    class _QueueTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self._payloads = iter(payloads)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(self._payloads))

    client = httpx.AsyncClient(transport=_QueueTransport())
    engine = OpenAIEngine(api_key="k", http_client=client)

    if store_kind == "memory":
        store: SessionStore | SQLiteSessionStore = SessionStore()
    elif store_kind == "sqlite_memory":
        store = SQLiteSessionStore(":memory:")
    else:
        store = SQLiteSessionStore(tmp_path / "t.db")

    runner = Runner(starting_agent=agent, session_store=store, engine=engine)
    try:
        result = await runner.run_async("s1", "hi")
        assert result.content == "done"
        session = store.get("s1")
        assert session is not None
        tool_msgs = [m for m in session.history if m.role == Role.TOOL]
        assert tool_msgs
        content = tool_msgs[0].tool_results[0].content
        assert content["when"] == "2026-01-02T03:04:05+00:00"
        assert content["odd"] == "odd-object"
    finally:
        await client.aclose()
        close = getattr(store, "close", None)
        if close is not None:
            close()
