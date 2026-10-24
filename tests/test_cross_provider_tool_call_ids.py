"""Tool-call IDs across provider handoffs (httpx.MockTransport, no network).

Every engine sends back the IDs in the shared history, so a call and its result keep
matching. OpenAI and Anthropic get a stable replacement for an ID they would reject; the
history itself is never changed, so the provider that issued an ID still gets it back.
"""

import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import AnthropicEngine, BaseEngine, GeminiEngine, OpenAIEngine
from chatflow_agent.core.memory import SessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.types import Message, Role, ToolCall, ToolResult

# The replacement an engine sends for an ID its provider would reject.
REPLACEMENT = re.compile(r"call_[0-9a-f]{24}")

OPENAI_ID = "call_AbC123xyz0123456789abcde"
ANTHROPIC_ID = "toolu_01A09q90qw90lq917835lq9"
GEMINI_ID = "fc-7Gq2"


def tool_round_trip(call_id: str, name: str = "lookup") -> list[Message]:
    return [
        Message(role=Role.USER, content="Weather?"),
        Message(role=Role.MODEL, tool_calls=[ToolCall(id=call_id, name=name, args={})]),
        Message(role=Role.TOOL, tool_results=[ToolResult(tool_call_id=call_id, name=name, content="sunny")]),
    ]


def openai_ids(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    calls = [c["id"] for m in body["messages"] for c in m.get("tool_calls", [])]
    results = [m["tool_call_id"] for m in body["messages"] if m["role"] == "tool"]
    return calls, results


def anthropic_ids(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    blocks = [b for m in body["messages"] if isinstance(m["content"], list) for b in m["content"]]
    calls = [b["id"] for b in blocks if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
    return calls, results


def gemini_ids(body: dict[str, Any]) -> tuple[list[str | None], list[str | None]]:
    parts = [p for c in body["contents"] for p in c["parts"]]
    calls = [p["functionCall"].get("id") for p in parts if "functionCall" in p]
    results = [p["functionResponse"].get("id") for p in parts if "functionResponse" in p]
    return calls, results


def formatted_openai(history: list[Message]) -> dict[str, Any]:
    return {"messages": OpenAIEngine(api_key="sk-test")._format_history_to_messages(None, history)}


def formatted_anthropic(history: list[Message]) -> dict[str, Any]:
    return {"messages": AnthropicEngine(api_key="ant-test")._format_history(history)}


# --- What each destination keeps -----------------------------------------------------------


@pytest.mark.parametrize(
    ("call_id", "kept"),
    [
        (OPENAI_ID, True),
        (ANTHROPIC_ID, True),
        (GEMINI_ID, True),
        ("call_lookup_0", True),  # Gemini's synthetic ID
        ("functions.lookup:0", True),  # OpenAI sets no character rule
        ("x" * 40, True),
        ("x" * 41, False),  # longer than OpenAI accepts
        ("", False),
    ],
)
def test_openai_keeps_ids_it_accepts_and_replaces_the_others_in_calls_and_results(call_id: str, kept: bool) -> None:
    calls, results = openai_ids(formatted_openai(tool_round_trip(call_id)))
    assert calls == results
    if kept:
        assert calls == [call_id]
    else:
        assert REPLACEMENT.fullmatch(calls[0])


@pytest.mark.parametrize("provider", ["deepseek", "groq", "xai", "ollama"])
def test_other_openai_compatible_providers_keep_ids_since_no_limit_is_known_for_them(provider: str) -> None:
    engine = OpenAIEngine(api_key="sk-test", provider=provider)
    messages = engine._format_history_to_messages(None, tool_round_trip("x" * 64))
    assert openai_ids({"messages": messages}) == (["x" * 64], ["x" * 64])


@pytest.mark.parametrize(
    ("call_id", "kept"),
    [
        (OPENAI_ID, True),
        (ANTHROPIC_ID, True),
        (GEMINI_ID, True),
        ("call_lookup_0", True),
        ("x" * 64, True),  # Anthropic sets no length rule
        ("call_maps.lookup_0", False),  # Gemini's synthetic ID for a dotted tool name
        ("functions.lookup:0", False),
        ("fc:lookup/1", False),
        ("", False),
    ],
)
def test_anthropic_keeps_ids_it_accepts_and_replaces_the_others_in_calls_and_results(call_id: str, kept: bool) -> None:
    calls, results = anthropic_ids(formatted_anthropic(tool_round_trip(call_id)))
    assert calls == results
    if kept:
        assert calls == [call_id]
    else:
        assert REPLACEMENT.fullmatch(calls[0])


def test_gemini_keeps_ids_other_providers_issued() -> None:
    """Gemini documents no format for these IDs: they are sent as they are."""
    engine = GeminiEngine(api_key="test-key")
    agent = Agent(name="Bot", instructions="x")
    for call_id in [OPENAI_ID, ANTHROPIC_ID, "functions.lookup:0"]:
        assert gemini_ids(engine._build_request_body(agent, tool_round_trip(call_id))) == ([call_id], [call_id])


def test_replacements_are_stable_distinct_and_accepted_by_both_providers() -> None:
    first = formatted_anthropic(tool_round_trip("fc:a"))
    again = formatted_openai(tool_round_trip("fc:a" + "x" * 40))
    other = formatted_anthropic(tool_round_trip("fc:b"))
    replacement = anthropic_ids(first)[0][0]
    assert anthropic_ids(formatted_anthropic(tool_round_trip("fc:a")))[0][0] == replacement
    assert anthropic_ids(other)[0][0] != replacement
    assert openai_ids(again)[0][0] not in (replacement, anthropic_ids(other)[0][0])
    for value in (replacement, openai_ids(again)[0][0]):
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", value) and len(value) <= 40


def test_formatting_never_changes_the_history() -> None:
    history = tool_round_trip("fc:lookup/1" + "x" * 40)
    before = [m.model_dump() for m in history]
    formatted_openai(history)
    formatted_anthropic(history)
    assert [m.model_dump() for m in history] == before


# --- Handoffs through the Runner ---------------------------------------------------------------


def gemini_call(name: str, call_id: str | None) -> dict[str, Any]:
    call: dict[str, Any] = {"name": name, "args": {}}
    if call_id is not None:
        call["id"] = call_id
    return {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": call}]}}]}


def gemini_text(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def openai_call(name: str, call_id: str | None) -> dict[str, Any]:
    tool_call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [tool_call]}}]}


def openai_text(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def anthropic_call(name: str, call_id: str | None) -> dict[str, Any]:
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}]}


def anthropic_text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


class Provider:
    """One mocked provider: an engine whose requests are recorded and answered from a queue."""

    def __init__(
        self,
        build: Callable[[httpx.AsyncClient], BaseEngine],
        call: Callable[[str, str | None], dict[str, Any]],
        text: Callable[[str], dict[str, Any]],
        ids: Callable[[dict[str, Any]], tuple[list[Any], list[Any]]],
        handoff_id: str,
    ) -> None:
        self.replies: list[dict[str, Any]] = []
        self.bodies: list[dict[str, Any]] = []
        self.engine = build(httpx.AsyncClient(transport=httpx.MockTransport(self._handle)))
        self.call, self.text, self.ids, self.handoff_id = call, text, ids, handoff_id

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))


PROVIDERS: dict[str, Callable[[], Provider]] = {
    "gemini": lambda: Provider(
        lambda client: GeminiEngine(api_key="test-key", http_client=client),
        gemini_call, gemini_text, gemini_ids, "fc-handoff",
    ),
    "openai": lambda: Provider(
        lambda client: OpenAIEngine(api_key="sk-test", http_client=client),
        openai_call, openai_text, openai_ids, "call_handoff0001",
    ),
    "anthropic": lambda: Provider(
        lambda client: AnthropicEngine(api_key="ant-test", http_client=client),
        anthropic_call, anthropic_text, anthropic_ids, "toolu_handoff01",
    ),
}


def lookup() -> str:
    """Look the weather up."""
    return "sunny"


def agents(first: Provider, second: Provider) -> tuple[Agent, Agent]:
    specialist = Agent(name="Specialist", instructions="Answer.", engine=second.engine)
    triage = Agent(name="Triage", instructions="Route.", tools=[lookup], handoffs=[specialist], engine=first.engine)
    return triage, specialist


@pytest.mark.parametrize(
    ("source", "destination", "lookup_id", "expected"),
    [
        ("gemini", "anthropic", "fc:lookup/1", None),  # None: replaced
        ("gemini", "anthropic", GEMINI_ID, GEMINI_ID),
        ("gemini", "anthropic", None, "call_lookup_0"),  # no Gemini ID: the synthetic one
        ("gemini", "openai", "fc-" + "a" * 45, None),
        ("gemini", "openai", "fc:lookup/1", "fc:lookup/1"),
        ("openai", "anthropic", OPENAI_ID, OPENAI_ID),
        ("anthropic", "openai", ANTHROPIC_ID, ANTHROPIC_ID),
        ("openai", "gemini", OPENAI_ID, OPENAI_ID),
        ("anthropic", "gemini", ANTHROPIC_ID, ANTHROPIC_ID),
    ],
)
async def test_handoff_sends_the_destination_matching_ids_it_accepts(
    source: str, destination: str, lookup_id: str | None, expected: str | None
) -> None:
    first, second = PROVIDERS[source](), PROVIDERS[destination]()
    first.replies = [first.call("lookup", lookup_id), first.call("transfer_to_specialist", first.handoff_id)]
    second.replies = [second.text("Sunny today.")]
    triage, _ = agents(first, second)

    response = await Runner(triage, session_store=SessionStore()).run_async("s1", "Weather?")

    assert response.content == "Sunny today."
    calls, results = second.ids(second.bodies[0])
    assert calls == results, "every result must carry its call's ID"
    assert calls[1] == first.handoff_id
    if expected is not None:
        assert calls[0] == expected
    else:
        assert REPLACEMENT.fullmatch(calls[0])


async def test_gemini_gets_its_own_ids_back_after_a_handoff_to_anthropic_and_back() -> None:
    gemini, anthropic = PROVIDERS["gemini"](), PROVIDERS["anthropic"]()
    gemini.replies = [
        gemini_call("lookup", "fc:lookup/1"),
        gemini_call("transfer_to_specialist", "fc-handoff"),
        gemini_text("Back with Triage."),
    ]
    anthropic.replies = [anthropic_call("transfer_to_triage", "toolu_back01")]
    triage, specialist = agents(gemini, anthropic)
    specialist.add_handoff(triage)

    response = await Runner(triage, session_store=SessionStore()).run_async("s1", "Weather?")

    assert response.content == "Back with Triage."
    sent_to_anthropic = anthropic_ids(anthropic.bodies[0])
    assert REPLACEMENT.fullmatch(sent_to_anthropic[0][0]) and sent_to_anthropic[0][1] == "fc-handoff"
    # The last Gemini request has the original IDs, not Anthropic's replacement.
    assert gemini_ids(gemini.bodies[-1]) == (
        ["fc:lookup/1", "fc-handoff", "toolu_back01"],
        ["fc:lookup/1", "fc-handoff", "toolu_back01"],
    )
