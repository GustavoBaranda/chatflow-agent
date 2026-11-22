"""Regression tests for GeminiEngine synthetic fallback tool-call IDs (#231).

Verifies that missing, null, empty, or whitespace-only ``functionCall.id`` values receive
collision-resistant stateless Gemini-scoped IDs (``cf_gemini_<24 hex chars>``), that
provider-supplied IDs are preserved byte-for-byte, that ``_with_provider_id()`` omits both
current and legacy Gemini synthetic IDs when sending history back to Gemini while preserving
OpenAI/Anthropic IDs, and that multi-turn, SQLite-reloaded, concurrent, and cross-provider
handoff payloads remain free of duplicate IDs.
"""

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
from chatflow_agent.core.engines.anthropic import _TOOL_USE_ID
from chatflow_agent.core.engines.openai_compatible import _OPENAI_MAX_TOOL_CALL_ID_LENGTH
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall, ToolResult

GEMINI_SYNTHETIC_ID = re.compile(r"cf_gemini_[0-9a-f]{24}")
OPENAI_SYNTHETIC_ID = re.compile(r"call_[0-9a-f]{24}")


def _gemini_parts_response(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": parts}}]}


def _gemini_call(name: str, call_id: Any = ..., args: dict[str, Any] | None = None) -> dict[str, Any]:
    fc: dict[str, Any] = {"name": name, "args": args or {}}
    if call_id is not ...:
        fc["id"] = call_id
    return _gemini_parts_response([{"functionCall": fc}])


def _gemini_text(text: str) -> dict[str, Any]:
    return _gemini_parts_response([{"text": text}])


def _openai_call(name: str, call_id: Any = ..., args: str = "{}") -> dict[str, Any]:
    tc: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": args}}
    if call_id is not ...:
        tc["id"] = call_id
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [tc]}}]}


def _openai_text(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _anthropic_call(name: str, call_id: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": args or {}}]}


def _anthropic_text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _make_mock_client(
    replies: list[dict[str, Any]],
    recorded_bodies: list[dict[str, Any]] | None = None,
) -> httpx.AsyncClient:
    queue = list(replies)

    def _handler(request: httpx.Request) -> httpx.Response:
        if recorded_bodies is not None:
            recorded_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def _gemini_payload_ids(body: dict[str, Any]) -> tuple[list[str | None], list[str | None]]:
    parts = [p for c in body["contents"] for p in c["parts"]]
    calls = [p["functionCall"].get("id") for p in parts if "functionCall" in p]
    results = [p["functionResponse"].get("id") for p in parts if "functionResponse" in p]
    return calls, results


def _anthropic_payload_ids(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    blocks = [b for m in body["messages"] if isinstance(m["content"], list) for b in m["content"]]
    calls = [b["id"] for b in blocks if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
    return calls, results


def _openai_payload_ids(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    calls = [c["id"] for m in body["messages"] for c in m.get("tool_calls", [])]
    results = [m["tool_call_id"] for m in body["messages"] if m["role"] == "tool"]
    return calls, results


# --- 1, 2, 3, 13: Input cases, provider ID preservation, non-string rejection, format/length ---


@pytest.mark.parametrize("raw_id", [..., None, "", "   ", "\t\n"])
async def test_missing_null_empty_or_whitespace_id_gets_unique_gemini_synthetic_id(raw_id: Any) -> None:
    agent = Agent(name="Bot", instructions="Test")
    client = _make_mock_client([_gemini_call("lookup", raw_id), _gemini_call("lookup", raw_id)])
    engine = GeminiEngine(api_key="test-key", http_client=client)
    try:
        r1 = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])
        r2 = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="again")])
        id1 = r1.tool_calls[0].id
        id2 = r2.tool_calls[0].id
        assert GEMINI_SYNTHETIC_ID.fullmatch(id1)
        assert GEMINI_SYNTHETIC_ID.fullmatch(id2)
        assert id1 != id2
        assert len(id1) == 34
        assert len(id1) <= _OPENAI_MAX_TOOL_CALL_ID_LENGTH
        assert _TOOL_USE_ID.fullmatch(id1)
        assert not id1.startswith("call_")
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "provider_id",
    [
        "fc-7Gq2",
        "fc:lookup/1",
        " fc-outer-spaces ",
        "call_AbC123xyz0123456789abcde",
        "toolu_01A09q90qw90lq917835lq9",
    ],
)
async def test_non_empty_provider_id_is_preserved_byte_for_byte_even_if_repeated(provider_id: str) -> None:
    agent = Agent(name="Bot", instructions="Test")
    client = _make_mock_client([_gemini_call("lookup", provider_id), _gemini_call("lookup", provider_id)])
    engine = GeminiEngine(api_key="test-key", http_client=client)
    try:
        r1 = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="one")])
        r2 = await engine.generate_turn_async(agent, [Message(role=Role.USER, content="two")])
        assert r1.tool_calls[0].id == provider_id
        assert r2.tool_calls[0].id == provider_id
    finally:
        await client.aclose()


@pytest.mark.parametrize("bad_id", [123, True, ["fc-1"], {"id": "fc-1"}])
async def test_non_string_id_raises_provider_error(bad_id: Any) -> None:
    agent = Agent(name="Bot", instructions="Test")
    client = _make_mock_client([_gemini_call("lookup", bad_id)])
    engine = GeminiEngine(api_key="test-key", http_client=client)
    try:
        with pytest.raises(ProviderError, match="functionCall.id"):
            await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])
    finally:
        await client.aclose()


# --- 4, 5, 6, 15: Consecutive turns, multi-call turn, ToolCall/ToolResult pairing, Gemini omission ---


async def test_consecutive_gemini_turns_and_multiple_calls_produce_unique_ids_and_omit_on_resend() -> None:
    def lookup(city: str = "Paris") -> str:
        """Look up weather."""
        return f"sunny in {city}"

    bodies: list[dict[str, Any]] = []
    client = _make_mock_client(
        [
            # Turn 1: two calls without IDs in one response
            _gemini_parts_response(
                [
                    {"functionCall": {"name": "lookup", "args": {"city": "Paris"}}},
                    {"functionCall": {"name": "lookup", "args": {"city": "Rome"}, "id": ""}},
                ]
            ),
            _gemini_text("Both checked."),
            # Turn 2: same tool again at part index 0 without ID
            _gemini_call("lookup", ..., {"city": "Madrid"}),
            _gemini_text("Madrid checked."),
        ],
        recorded_bodies=bodies,
    )
    engine = GeminiEngine(api_key="test-key", http_client=client)
    agent = Agent(name="Bot", instructions="Check weather.", tools=[lookup], engine=engine)
    store = SessionStore()
    runner = Runner(agent, session_store=store)
    try:
        r1 = await runner.run_async("s1", "Paris and Rome?")
        r2 = await runner.run_async("s1", "Madrid too?")
        assert r1.content == "Both checked."
        assert r2.content == "Madrid checked."

        ctx = store.get_or_create("s1", agent)
        call_ids = [c.id for m in ctx.history if m.role == Role.MODEL for c in (m.tool_calls or [])]
        result_ids = [r.tool_call_id for m in ctx.history if m.role == Role.TOOL for r in (m.tool_results or [])]
        assert len(call_ids) == 3
        assert len(set(call_ids)) == 3
        assert call_ids == result_ids
        assert all(GEMINI_SYNTHETIC_ID.fullmatch(cid) for cid in call_ids)

        # In the final request sent back to Gemini, all 3 synthetic IDs must be omitted.
        sent_calls, sent_results = _gemini_payload_ids(bodies[-1])
        assert sent_calls == [None, None, None]
        assert sent_results == [None, None, None]
    finally:
        await client.aclose()


# --- 7: Multi-turn Gemini -> Anthropic handoff (real payload validation) ---


async def test_multiturn_gemini_to_anthropic_handoff_sends_unique_tool_use_ids() -> None:
    def lookup() -> str:
        """Look up data."""
        return "ok"

    gm_bodies: list[dict[str, Any]] = []
    an_bodies: list[dict[str, Any]] = []
    gm_client = _make_mock_client(
        [
            _gemini_call("lookup"),
            _gemini_text("First done."),
            _gemini_call("lookup"),
            _gemini_call("transfer_to_claude"),
        ],
        recorded_bodies=gm_bodies,
    )
    an_client = _make_mock_client([_anthropic_text("Handled by Claude.")], recorded_bodies=an_bodies)
    claude = Agent(
        name="Claude",
        instructions="Finish.",
        engine=AnthropicEngine(api_key="ant-test", http_client=an_client),
    )
    triage = Agent(
        name="Triage",
        instructions="Route.",
        tools=[lookup],
        handoffs=[claude],
        engine=GeminiEngine(api_key="gm-test", http_client=gm_client),
    )
    runner = Runner(triage, session_store=SessionStore())
    try:
        assert (await runner.run_async("s-ga", "Turn 1")).content == "First done."
        assert (await runner.run_async("s-ga", "Turn 2")).content == "Handled by Claude."

        an_calls, an_results = _anthropic_payload_ids(an_bodies[0])
        assert len(an_calls) == 3
        assert len(set(an_calls)) == 3
        assert an_calls == an_results
        assert all(GEMINI_SYNTHETIC_ID.fullmatch(cid) for cid in an_calls)
    finally:
        await gm_client.aclose()
        await an_client.aclose()


# --- 8: Multi-turn Gemini -> OpenAI handoff (including long/dotted tool names) ---


async def test_multiturn_gemini_to_openai_handoff_sends_unique_valid_ids() -> None:
    def very_long_dotted_style_lookup_tool_name_for_testing() -> str:
        """Long tool name exceeding 40 chars if embedded in an ID."""
        return "data"

    gm_bodies: list[dict[str, Any]] = []
    oa_bodies: list[dict[str, Any]] = []
    tool_name = very_long_dotted_style_lookup_tool_name_for_testing.__name__
    gm_client = _make_mock_client(
        [
            _gemini_call(tool_name),
            _gemini_text("First done."),
            _gemini_call(tool_name),
            _gemini_call("transfer_to_gpt"),
        ],
        recorded_bodies=gm_bodies,
    )
    oa_client = _make_mock_client([_openai_text("Handled by GPT.")], recorded_bodies=oa_bodies)
    gpt = Agent(
        name="GPT",
        instructions="Finish.",
        engine=OpenAIEngine(api_key="sk-test", http_client=oa_client),
    )
    triage = Agent(
        name="Triage",
        instructions="Route.",
        tools=[very_long_dotted_style_lookup_tool_name_for_testing],
        handoffs=[gpt],
        engine=GeminiEngine(api_key="gm-test", http_client=gm_client),
    )
    runner = Runner(triage, session_store=SessionStore())
    try:
        assert (await runner.run_async("s-go", "Turn 1")).content == "First done."
        assert (await runner.run_async("s-go", "Turn 2")).content == "Handled by GPT."

        oa_calls, oa_results = _openai_payload_ids(oa_bodies[0])
        assert len(oa_calls) == 3
        assert len(set(oa_calls)) == 3
        assert oa_calls == oa_results
        assert all(GEMINI_SYNTHETIC_ID.fullmatch(cid) for cid in oa_calls)
        assert all(len(cid) <= _OPENAI_MAX_TOOL_CALL_ID_LENGTH for cid in oa_calls)
    finally:
        await gm_client.aclose()
        await oa_client.aclose()


# --- 9: Round-trip Gemini -> Anthropic -> Gemini ---


async def test_roundtrip_gemini_anthropic_gemini_omits_gemini_synthetic_and_keeps_anthropic_id() -> None:
    def lookup() -> str:
        """Look up."""
        return "sunny"

    gm_bodies: list[dict[str, Any]] = []
    an_bodies: list[dict[str, Any]] = []
    gm_client = _make_mock_client(
        [
            _gemini_call("lookup"),
            _gemini_call("transfer_to_claude"),
            _gemini_text("Back on Gemini."),
        ],
        recorded_bodies=gm_bodies,
    )
    an_client = _make_mock_client(
        [_anthropic_call("transfer_to_triage", "toolu_back_to_gemini_01")],
        recorded_bodies=an_bodies,
    )
    claude = Agent(
        name="Claude",
        instructions="Hand back.",
        engine=AnthropicEngine(api_key="ant-test", http_client=an_client),
    )
    triage = Agent(
        name="Triage",
        instructions="Route.",
        tools=[lookup],
        handoffs=[claude],
        engine=GeminiEngine(api_key="gm-test", http_client=gm_client),
    )
    claude.add_handoff(triage)
    runner = Runner(triage, session_store=SessionStore())
    try:
        response = await runner.run_async("s-rt", "Go")
        assert response.content == "Back on Gemini."

        # Anthropic received both Gemini synthetic IDs verbatim and distinct.
        an_calls, an_results = _anthropic_payload_ids(an_bodies[0])
        assert len(an_calls) == 2
        assert len(set(an_calls)) == 2
        assert an_calls == an_results
        assert all(GEMINI_SYNTHETIC_ID.fullmatch(cid) for cid in an_calls)

        # Final Gemini request omits its own two synthetic IDs but preserves Anthropic's real ID.
        gm_calls, gm_results = _gemini_payload_ids(gm_bodies[-1])
        assert gm_calls == [None, None, "toolu_back_to_gemini_01"]
        assert gm_results == [None, None, "toolu_back_to_gemini_01"]
    finally:
        await gm_client.aclose()
        await an_client.aclose()


# --- 10: SQLite persistence and reload across simulated process restart ---


async def test_sqlite_persistence_and_reload_avoids_collisions_across_restarts(tmp_path: Path) -> None:
    def lookup() -> str:
        """Look up."""
        return "value"

    db_path = tmp_path / "gemini_fallback.db"

    # Process 1
    client1 = _make_mock_client([_gemini_call("lookup"), _gemini_text("Turn 1 done.")])
    try:
        with SQLiteSessionStore(db_path) as store1:
            agent1 = Agent(
                name="Bot",
                instructions="Use lookup.",
                tools=[lookup],
                engine=GeminiEngine(api_key="gm-test", http_client=client1),
            )
            r1 = await Runner(agent1, session_store=store1).run_async("persist-1", "First")
            assert r1.content == "Turn 1 done."
    finally:
        await client1.aclose()

    # Process 2 (fresh store + fresh engine)
    bodies2: list[dict[str, Any]] = []
    client2 = _make_mock_client(
        [_gemini_call("lookup"), _gemini_text("Turn 2 done.")],
        recorded_bodies=bodies2,
    )
    try:
        with SQLiteSessionStore(db_path) as store2:
            agent2 = Agent(
                name="Bot",
                instructions="Use lookup.",
                tools=[lookup],
                engine=GeminiEngine(api_key="gm-test", http_client=client2),
            )
            r2 = await Runner(agent2, session_store=store2).run_async("persist-1", "Second")
            assert r2.content == "Turn 2 done."

            ctx = store2.get_or_create("persist-1", agent2)
            call_ids = [c.id for m in ctx.history if m.role == Role.MODEL for c in (m.tool_calls or [])]
            result_ids = [r.tool_call_id for m in ctx.history if m.role == Role.TOOL for r in (m.tool_results or [])]
            assert len(call_ids) == 2
            assert len(set(call_ids)) == 2
            assert call_ids == result_ids
            assert all(GEMINI_SYNTHETIC_ID.fullmatch(cid) for cid in call_ids)

            # Both persisted synthetic IDs are omitted when resending to Gemini.
            sent_calls, sent_results = _gemini_payload_ids(bodies2[-1])
            assert sent_calls == [None, None]
            assert sent_results == [None, None]
    finally:
        await client2.aclose()


# --- 11: Backward compatibility with legacy SQLite sessions (call_{name}_{part_idx}) ---


async def test_legacy_sqlite_session_with_old_gemini_synthetic_ids_remains_compatible(tmp_path: Path) -> None:
    def lookup() -> str:
        """Look up."""
        return "fresh"

    db_path = tmp_path / "legacy_gemini.db"
    seed_agent = Agent(name="Bot", instructions="Use lookup.", tools=[lookup])
    with SQLiteSessionStore(db_path) as seed_store:
        ctx = seed_store.get_or_create("legacy-s1", seed_agent)
        ctx.add_message(Message(role=Role.USER, content="Old turn"))
        ctx.add_message(
            Message(
                role=Role.MODEL,
                tool_calls=[ToolCall(id="call_lookup_0", name="lookup", args={})],
            )
        )
        ctx.add_message(
            Message(
                role=Role.TOOL,
                tool_results=[ToolResult(tool_call_id="call_lookup_0", name="lookup", content="old")],
            )
        )
        ctx.add_message(Message(role=Role.MODEL, content="Old answer"))

    bodies: list[dict[str, Any]] = []
    client = _make_mock_client([_gemini_call("lookup"), _gemini_text("New answer")], recorded_bodies=bodies)
    try:
        with SQLiteSessionStore(db_path) as reopened_store:
            agent = Agent(
                name="Bot",
                instructions="Use lookup.",
                tools=[lookup],
                engine=GeminiEngine(api_key="gm-test", http_client=client),
            )
            res = await Runner(agent, session_store=reopened_store).run_async("legacy-s1", "New turn")
            assert res.content == "New answer"

            ctx = reopened_store.get_or_create("legacy-s1", agent)
            call_ids = [c.id for m in ctx.history if m.role == Role.MODEL for c in (m.tool_calls or [])]
            assert call_ids[0] == "call_lookup_0"
            assert GEMINI_SYNTHETIC_ID.fullmatch(call_ids[1])
            assert call_ids[0] != call_ids[1]

            # Both the legacy call_lookup_0 and the new cf_gemini_* ID are omitted for Gemini.
            sent_calls, sent_results = _gemini_payload_ids(bodies[-1])
            assert sent_calls == [None, None]
            assert sent_results == [None, None]
    finally:
        await client.aclose()


# --- 12: Concurrency and uniqueness ---


async def test_concurrent_gemini_calls_generate_unique_synthetic_ids() -> None:
    batch_size = 25
    replies = [
        _gemini_parts_response(
            [
                {"functionCall": {"name": "lookup", "args": {}}},
                {"functionCall": {"name": "lookup", "args": {}, "id": "  "}},
            ]
        )
        for _ in range(batch_size)
    ]
    client = _make_mock_client(replies)
    engine = GeminiEngine(api_key="gm-test", http_client=client)
    agent = Agent(name="Bot", instructions="Concurrent.")
    try:
        results = await asyncio.gather(
            *(engine.generate_turn_async(agent, [Message(role=Role.USER, content=f"msg-{i}")]) for i in range(batch_size))
        )
        ids = [tc.id for r in results for tc in r.tool_calls]
        assert len(ids) == batch_size * 2
        assert len(set(ids)) == batch_size * 2
        assert all(GEMINI_SYNTHETIC_ID.fullmatch(cid) for cid in ids)
    finally:
        await client.aclose()


# --- 14: OpenAI -> Gemini handoff preserves OpenAI synthetic IDs ---


async def test_openai_to_gemini_handoff_preserves_openai_synthetic_ids() -> None:
    def lookup() -> str:
        """Look up."""
        return "ok"

    oa_bodies: list[dict[str, Any]] = []
    gm_bodies: list[dict[str, Any]] = []
    oa_client = _make_mock_client(
        [
            _openai_call("lookup", None),
            _openai_call("transfer_to_gemini", ""),
        ],
        recorded_bodies=oa_bodies,
    )
    gm_client = _make_mock_client([_gemini_text("Done on Gemini.")], recorded_bodies=gm_bodies)
    gemini_agent = Agent(
        name="Gemini",
        instructions="Finish.",
        engine=GeminiEngine(api_key="gm-test", http_client=gm_client),
    )
    triage = Agent(
        name="Triage",
        instructions="Route.",
        tools=[lookup],
        handoffs=[gemini_agent],
        engine=OpenAIEngine(api_key="sk-test", http_client=oa_client),
    )
    runner = Runner(triage, session_store=SessionStore())
    try:
        response = await runner.run_async("s-og", "Start")
        assert response.content == "Done on Gemini."

        gm_calls, gm_results = _gemini_payload_ids(gm_bodies[0])
        assert len(gm_calls) == 2
        assert len(set(gm_calls)) == 2
        assert gm_calls == gm_results
        assert all(isinstance(cid, str) and OPENAI_SYNTHETIC_ID.fullmatch(cid) for cid in gm_calls)
    finally:
        await oa_client.aclose()
        await gm_client.aclose()
