"""Regression tests for collision-resistant fallback tool-call IDs in OpenAIEngine (#213)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import AnthropicEngine, GeminiEngine, OpenAIEngine
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role

SYNTHETIC_ID = re.compile(r"call_[0-9a-f]{24}")
ANTHROPIC_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]+")
OPENAI_MAX_ID_LEN = 40


def lookup(city: str = "BA") -> str:
    """Mock weather lookup."""
    return f"sunny in {city}"


def get_time(tz: str = "UTC") -> str:
    """Mock clock lookup."""
    return f"12:00 {tz}"


def _openai_tool_resp(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}


def _openai_text_resp(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _assert_cross_provider_valid(call_id: str) -> None:
    assert SYNTHETIC_ID.fullmatch(call_id), f"unexpected synthetic ID format: {call_id!r}"
    assert len(call_id) <= OPENAI_MAX_ID_LEN
    assert ANTHROPIC_ID_PATTERN.fullmatch(call_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_tc",
    [
        {"type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": None, "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "   \t\n ", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
    ],
    ids=["missing", "none", "empty", "whitespace"],
)
async def test_missing_none_empty_or_whitespace_id_generates_valid_unique_fallback(
    raw_tc: dict[str, Any],
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_tool_resp([raw_tc]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = OpenAIEngine(api_key="sk-test", http_client=client)
        agent = Agent(name="Bot", instructions="Test", tools=[lookup], engine=engine)
        r1 = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])
        r2 = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])

    id1 = r1.tool_calls[0].id
    id2 = r2.tool_calls[0].id
    _assert_cross_provider_valid(id1)
    _assert_cross_provider_valid(id2)
    assert id1 != id2


@pytest.mark.asyncio
async def test_valid_provider_supplied_ids_are_preserved_byte_for_byte() -> None:
    provider_ids = ["call_AbC123xyz0123456789abcde", "functions.lookup:0", "custom-id_42"]
    calls = [
        {"id": pid, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        for pid in provider_ids
    ]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=_openai_tool_resp(calls)))
    ) as client:
        engine = OpenAIEngine(api_key="sk-test", http_client=client)
        agent = Agent(name="Bot", instructions="Test", tools=[lookup], engine=engine)
        res = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])

    assert [c.id for c in res.tool_calls] == provider_ids


@pytest.mark.asyncio
async def test_non_string_tool_call_id_still_raises_provider_error() -> None:
    bad_call = {"id": 123, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(200, json=_openai_tool_resp([bad_call]))
        )
    ) as client:
        engine = OpenAIEngine(api_key="sk-test", http_client=client)
        agent = Agent(name="Bot", instructions="Test", tools=[lookup], engine=engine)
        with pytest.raises(ProviderError, match="id is not a string"):
            await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])


@pytest.mark.asyncio
async def test_two_consecutive_turns_without_ids_produce_distinct_matching_pairs() -> None:
    recorded_bodies: list[dict[str, Any]] = []
    replies = [
        _openai_tool_resp([{"type": "function", "function": {"name": "lookup", "arguments": '{"city":"BA"}'}}]),
        _openai_text_resp("Turn 1 done"),
        _openai_tool_resp([{"id": None, "type": "function", "function": {"name": "lookup", "arguments": '{"city":"NYC"}'}}]),
        _openai_text_resp("Turn 2 done"),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        recorded_bodies.append(json.loads(req.content))
        return httpx.Response(200, json=replies.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = OpenAIEngine(api_key="sk-test", http_client=client)
        agent = Agent(name="Bot", instructions="Test", tools=[lookup], engine=engine)
        store = SessionStore()
        runner = Runner(agent, session_store=store, engine=engine)

        r1 = await runner.run_async("s1", "Weather BA?")
        r2 = await runner.run_async("s1", "Weather NYC?")

    assert r1.content == "Turn 1 done"
    assert r2.content == "Turn 2 done"

    session = store.get("s1")
    assert session is not None
    tc_ids = [c.id for m in session.history if m.tool_calls for c in m.tool_calls]
    tr_ids = [r.tool_call_id for m in session.history if m.tool_results for r in m.tool_results]
    assert len(tc_ids) == 2
    assert len(set(tc_ids)) == 2
    assert tc_ids == tr_ids
    for cid in tc_ids:
        _assert_cross_provider_valid(cid)

    # Last outbound OpenAI request carries both distinct IDs without regenerating them
    last_calls = [c["id"] for m in recorded_bodies[-1]["messages"] for c in m.get("tool_calls", [])]
    last_results = [m["tool_call_id"] for m in recorded_bodies[-1]["messages"] if m["role"] == "tool"]
    assert last_calls == tc_ids
    assert last_results == tr_ids


@pytest.mark.asyncio
async def test_multiple_missing_id_calls_within_same_turn_are_distinct() -> None:
    replies = [
        _openai_tool_resp(
            [
                {"type": "function", "function": {"name": "lookup", "arguments": '{"city":"BA"}'}},
                {"id": "", "type": "function", "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'}},
                {"id": None, "type": "function", "function": {"name": "lookup", "arguments": '{"city":"MAD"}'}},
            ]
        ),
        _openai_text_resp("All three done"),
    ]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=replies.pop(0)))
    ) as client:
        engine = OpenAIEngine(api_key="sk-test", http_client=client)
        agent = Agent(name="Bot", instructions="Test", tools=[lookup, get_time], engine=engine)
        store = SessionStore()
        runner = Runner(agent, session_store=store, engine=engine)
        res = await runner.run_async("s_multi", "Check all")

    assert res.content == "All three done"
    session = store.get("s_multi")
    assert session is not None
    tc_ids = [c.id for m in session.history if m.tool_calls for c in m.tool_calls]
    tr_ids = [r.tool_call_id for m in session.history if m.tool_results for r in m.tool_results]
    assert len(tc_ids) == 3
    assert len(set(tc_ids)) == 3
    assert tc_ids == tr_ids
    for cid in tc_ids:
        _assert_cross_provider_valid(cid)


@pytest.mark.asyncio
async def test_sqlite_persistence_and_reload_preserve_unique_ids_across_restarts(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"

    def make_replies(label: str) -> list[dict[str, Any]]:
        return [
            _openai_tool_resp([{"type": "function", "function": {"name": "lookup", "arguments": f'{{"city":"{label}"}}'}}]),
            _openai_text_resp(f"Done {label}"),
        ]

    # First store lifecycle: Turn 1
    replies_1 = make_replies("T1")
    with SQLiteSessionStore(db_path) as store1:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=replies_1.pop(0)))
        ) as client1:
            eng1 = OpenAIEngine(api_key="sk-test", http_client=client1)
            agent1 = Agent(name="Bot", instructions="Test", tools=[lookup], engine=eng1)
            await Runner(agent1, session_store=store1, engine=eng1).run_async("s_sql", "First")
        sess1 = store1.get("s_sql")
        assert sess1 is not None
        first_id = [c.id for m in sess1.history if m.tool_calls for c in m.tool_calls][0]

    # Second store lifecycle (simulates process restart): Turn 2
    replies_2 = make_replies("T2")
    with SQLiteSessionStore(db_path) as store2:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=replies_2.pop(0)))
        ) as client2:
            eng2 = OpenAIEngine(api_key="sk-test", http_client=client2)
            agent2 = Agent(name="Bot", instructions="Test", tools=[lookup], engine=eng2)
            await Runner(agent2, session_store=store2, engine=eng2).run_async("s_sql", "Second")
        reloaded = store2.get("s_sql")
        assert reloaded is not None
        tc_ids = [c.id for m in reloaded.history if m.tool_calls for c in m.tool_calls]
        tr_ids = [r.tool_call_id for m in reloaded.history if m.tool_results for r in m.tool_results]

    assert len(tc_ids) == 2
    assert tc_ids[0] == first_id  # Persisted ID was not regenerated on reload
    assert tc_ids[0] != tc_ids[1]  # Post-restart turn did not collide with persisted ID
    assert tc_ids == tr_ids
    for cid in tc_ids:
        _assert_cross_provider_valid(cid)


@pytest.mark.asyncio
async def test_handoff_from_openai_missing_ids_to_anthropic_and_gemini_preserves_unique_valid_ids() -> None:
    openai_replies = [
        _openai_tool_resp([{"type": "function", "function": {"name": "lookup", "arguments": '{"city":"BA"}'}}]),
        _openai_text_resp("OA T1"),
        _openai_tool_resp([{"id": "", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"NYC"}'}}]),
        _openai_tool_resp([{"id": None, "type": "function", "function": {"name": "transfer_to_claude", "arguments": "{}"}}]),
    ]
    anthropic_bodies: list[dict[str, Any]] = []
    anthropic_replies = [
        {"content": [{"type": "tool_use", "id": "toolu_handoff_01", "name": "transfer_to_gemini", "input": {}}]},
    ]
    gemini_bodies: list[dict[str, Any]] = []

    def an_handler(req: httpx.Request) -> httpx.Response:
        anthropic_bodies.append(json.loads(req.content))
        return httpx.Response(200, json=anthropic_replies.pop(0))

    def gm_handler(req: httpx.Request) -> httpx.Response:
        gemini_bodies.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "Final Gemini"}]}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=openai_replies.pop(0)))
    ) as oa_client, httpx.AsyncClient(
        transport=httpx.MockTransport(an_handler)
    ) as an_client, httpx.AsyncClient(
        transport=httpx.MockTransport(gm_handler)
    ) as gm_client:
        oa_eng = OpenAIEngine(api_key="sk-test", http_client=oa_client)
        an_eng = AnthropicEngine(api_key="ant-test", http_client=an_client)
        gm_eng = GeminiEngine(api_key="gm-test", http_client=gm_client)

        gemini_agent = Agent(name="Gemini", instructions="Finish.", engine=gm_eng)
        claude_agent = Agent(name="Claude", instructions="Forward.", handoffs=[gemini_agent], engine=an_eng)
        oa_agent = Agent(name="OpenAI", instructions="Start.", tools=[lookup], handoffs=[claude_agent], engine=oa_eng)

        store = SessionStore()
        runner = Runner(oa_agent, session_store=store, engine=oa_eng)

        r1 = await runner.run_async("s_handoff", "Turn 1")
        assert r1.content == "OA T1"
        r2 = await runner.run_async("s_handoff", "Turn 2")
        assert r2.content == "Final Gemini"

    # Anthropic received 3 unique, valid tool_use/tool_result IDs (2 lookups + 1 handoff)
    an_blocks = [
        b for m in anthropic_bodies[0]["messages"] if isinstance(m["content"], list) for b in m["content"]
    ]
    an_use_ids = [b["id"] for b in an_blocks if b["type"] == "tool_use"]
    an_res_ids = [b["tool_use_id"] for b in an_blocks if b["type"] == "tool_result"]
    assert len(an_use_ids) == 3
    assert len(set(an_use_ids)) == 3
    assert an_use_ids == an_res_ids
    for cid in an_use_ids:
        _assert_cross_provider_valid(cid)

    # Gemini received 4 unique IDs (3 OpenAI synthetic + 1 Anthropic provider ID preserved)
    gm_parts = [p for c in gemini_bodies[0]["contents"] for p in c["parts"]]
    gm_call_ids = [p["functionCall"].get("id") for p in gm_parts if "functionCall" in p]
    gm_resp_ids = [p["functionResponse"].get("id") for p in gm_parts if "functionResponse" in p]
    assert len(gm_call_ids) == 4
    assert len(set(gm_call_ids)) == 4
    assert gm_call_ids == gm_resp_ids
    assert gm_call_ids[:3] == an_use_ids
    assert gm_call_ids[3] == "toolu_handoff_01"


@pytest.mark.asyncio
async def test_concurrent_executions_generate_unique_fallback_ids() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_tool_resp(
                [
                    {"type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                    {"id": "", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = OpenAIEngine(api_key="sk-test", http_client=client)
        agent = Agent(name="Bot", instructions="Test", tools=[lookup, get_time], engine=engine)
        results = await asyncio.gather(
            *(
                engine.generate_turn_async(agent, [Message(role=Role.USER, content=f"msg-{i}")])
                for i in range(25)
            )
        )

    all_ids = [c.id for res in results for c in res.tool_calls]
    assert len(all_ids) == 50
    assert len(set(all_ids)) == 50
    for cid in all_ids:
        _assert_cross_provider_valid(cid)
