"""Tests for GeminiEngine over the Gemini REST API (httpx.MockTransport, no network)."""

import asyncio
import dataclasses
import inspect
import json
import logging
import re
import traceback
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any
from urllib.parse import quote

import httpx
import pytest
from pydantic import BaseModel

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import GeminiEngine, resolve_engine
from chatflow_agent.core.engines import gemini as gemini_module
from chatflow_agent.core.memory import SessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall, ToolResult

API_KEY = "test-secret-key-123"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


def text_payload(*texts: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t} for t in texts]}}]}


def call_payload(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    parts = [{"functionCall": {"name": name, "args": args}} for name, args in calls]
    return {"candidates": [{"content": {"role": "model", "parts": parts}}]}


class Recorder:
    """An httpx.MockTransport that records requests and replies with a queue or a handler."""

    def __init__(self, *replies: Any, handler: Handler | None = None) -> None:
        self.replies = list(replies)
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.handler is not None:
            result = self.handler(request)
            return await result if inspect.isawaitable(result) else result
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def make_engine(recorder: Recorder, **kwargs: Any) -> GeminiEngine:
    kwargs.setdefault("base_delay", 0.001)
    return GeminiEngine(
        api_key=API_KEY, http_client=httpx.AsyncClient(transport=recorder.transport), **kwargs
    )


def bot(**kwargs: Any) -> Agent:
    kwargs.setdefault("instructions", "Be helpful.")
    return Agent(name="Bot", model="gemini-2.5-flash", **kwargs)


USER_HELLO = [Message(role=Role.USER, content="Hello")]


@pytest.fixture(autouse=True)
def default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)


@pytest.fixture
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def attempt_clients(monkeypatch: pytest.MonkeyPatch) -> Callable[[Recorder], list[httpx.AsyncClient]]:
    """Route engine-created clients through a recorder and keep them for inspection."""

    def install(recorder: Recorder) -> list[httpx.AsyncClient]:
        created: list[httpx.AsyncClient] = []

        def factory(timeout: float) -> httpx.AsyncClient:
            client = httpx.AsyncClient(transport=recorder.transport, timeout=timeout)
            created.append(client)
            return client

        monkeypatch.setattr(gemini_module, "_new_attempt_client", factory)
        return created

    return install


# --- Request: endpoint and authentication -------------------------------------------------


async def test_request_targets_generate_content_with_key_in_header() -> None:
    recorder = Recorder(text_payload("Hi"))
    await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)

    req = recorder.requests[0]
    assert req.method == "POST"
    assert str(req.url) == ENDPOINT
    assert req.headers["x-goog-api-key"] == API_KEY
    assert req.headers["content-type"] == "application/json"
    assert "key" not in req.url.params
    assert API_KEY not in str(req.url)


async def test_query_auth_sends_key_as_parameter_and_not_as_header() -> None:
    recorder = Recorder(text_payload("Hi"))
    engine = make_engine(recorder, api_key_in_query=True)
    await engine.generate_turn_async(bot(), USER_HELLO)

    req = recorder.requests[0]
    assert req.url.params["key"] == API_KEY
    assert req.url.copy_with(query=None) == httpx.URL(ENDPOINT)
    assert "x-goog-api-key" not in req.headers


async def test_models_prefix_in_model_name_is_not_doubled() -> None:
    recorder = Recorder(text_payload("Hi"))
    agent = Agent(name="Bot", model="models/gemini-2.5-flash", instructions="x")
    await make_engine(recorder).generate_turn_async(agent, USER_HELLO)
    assert str(recorder.requests[0].url) == ENDPOINT


async def test_tuned_models_use_the_tuned_models_collection() -> None:
    recorder = Recorder(text_payload("Hi"))
    agent = Agent(name="Bot", model="tunedModels/my-tuned-model-123", instructions="x")
    await make_engine(recorder).generate_turn_async(agent, USER_HELLO)
    assert str(recorder.requests[0].url) == (
        "https://generativelanguage.googleapis.com/v1beta/tunedModels/my-tuned-model-123:generateContent"
    )


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.5-flash\n",
        " gemini-2.5-flash",
        "gemini?alt=sse",
        "gemini#fragment",
        "gemini&x=1",
        "../../v1beta/files",
        "models/../files",
        "a/b",
        "",
        "models/",
        "tunedModels/",
        "gemini-2.5-flash:streamGenerateContent",
        "gémini",
    ],
)
async def test_invalid_model_names_raise_provider_error_without_a_request(model: str) -> None:
    recorder = Recorder(text_payload("Hi"))
    agent = Agent(name="Bot", model=model, instructions="x")
    with pytest.raises(ProviderError, match="Invalid Gemini model name"):
        await make_engine(recorder).generate_turn_async(agent, USER_HELLO)
    assert recorder.requests == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("https://gemini-proxy.example.com", "https://gemini-proxy.example.com/v1beta/models/gemini-2.5-flash:generateContent"),
        ("https://gemini-proxy.example.com/", "https://gemini-proxy.example.com/v1beta/models/gemini-2.5-flash:generateContent"),
        ("http://127.0.0.1:8080/gateway/", "http://127.0.0.1:8080/gateway/v1beta/models/gemini-2.5-flash:generateContent"),
    ],
)
async def test_google_gemini_base_url_replaces_the_host_root(
    monkeypatch: pytest.MonkeyPatch, override: str, expected: str
) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", override)
    recorder = Recorder(text_payload("Hi"))
    await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert str(recorder.requests[0].url) == expected
    assert recorder.requests[0].headers["x-goog-api-key"] == API_KEY


@pytest.mark.parametrize(
    "override",
    ["ftp://proxy.example.com", "proxy.example.com", "https://", "https://[::1", "https://proxy.example.com/?a=1", "https://proxy example.com"],
)
async def test_invalid_google_gemini_base_url_raises_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", override)
    recorder = Recorder(text_payload("Hi"))
    with pytest.raises(ProviderError, match="GOOGLE_GEMINI_BASE_URL is not a valid") as excinfo:
        await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert override not in str(excinfo.value)
    assert recorder.requests == []


async def test_url_construction_errors_are_wrapped_in_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_module, "_API_BASE_URL", "https://[::1/v1beta")
    with pytest.raises(ProviderError, match="request URL is invalid"):
        await make_engine(Recorder(text_payload("Hi"))).generate_turn_async(bot(), USER_HELLO)


# --- Request: body mapping -----------------------------------------------------------------


async def test_system_instruction_joins_agent_instructions_and_system_messages() -> None:
    recorder = Recorder(text_payload("Hi"))
    history = [
        Message(role=Role.SYSTEM, content="Rule one"),
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.SYSTEM, content="Rule two"),
    ]
    await make_engine(recorder).generate_turn_async(bot(instructions="Base"), history)

    body = recorder.bodies()[0]
    assert body["systemInstruction"] == {"parts": [{"text": "Base\n\nRule one\n\nRule two"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


async def test_system_instruction_is_omitted_without_instructions() -> None:
    recorder = Recorder(text_payload("Hi"))
    await make_engine(recorder).generate_turn_async(bot(instructions=""), USER_HELLO)
    assert "systemInstruction" not in recorder.bodies()[0]


async def test_history_maps_roles_function_calls_and_function_responses() -> None:
    recorder = Recorder(text_payload("Done"))
    history = [
        Message(role=Role.USER, content="Weather and time?"),
        Message(role=Role.USER, content=""),
        Message(
            role=Role.MODEL,
            content="Checking.",
            tool_calls=[
                ToolCall(id="call_weather_1", name="weather", args={"city": "Paris"}),
                ToolCall(id="call_clock_2", name="clock", args={}),
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[
                ToolResult(tool_call_id="call_weather_1", name="weather", content={"temp": 21}),
                ToolResult(tool_call_id="call_clock_2", name="clock", content="12:00"),
            ],
        ),
        Message(role=Role.MODEL, content=None),
        Message(role=Role.SYSTEM, content="Not part of contents"),
        Message(role=Role.MODEL, content="It is sunny."),
    ]
    await make_engine(recorder).generate_turn_async(bot(), history)

    assert recorder.bodies()[0]["contents"] == [
        {"role": "user", "parts": [{"text": "Weather and time?"}]},
        {
            "role": "model",
            "parts": [
                {"text": "Checking."},
                {"functionCall": {"name": "weather", "args": {"city": "Paris"}}},
                {"functionCall": {"name": "clock", "args": {}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "weather", "response": {"temp": 21}}},
                {"functionResponse": {"name": "clock", "response": {"output": "12:00"}}},
            ],
        },
        {"role": "model", "parts": [{"text": "It is sunny."}]},
    ]


class Invoice(BaseModel):
    number: int
    issued: datetime


@dataclasses.dataclass
class Shipment:
    carrier: str
    eta: date


def tool_result_history(content: Any) -> list[Message]:
    return [
        Message(role=Role.USER, content="Status?"),
        Message(role=Role.MODEL, tool_calls=[ToolCall(id="c", name="status", args={})]),
        Message(role=Role.TOOL, tool_results=[ToolResult(tool_call_id="c", name="status", content=content)]),
    ]


async def send_tool_result(content: Any) -> dict[str, Any]:
    """Send one tool result and return the functionResponse payload the engine produced."""
    recorder = Recorder(text_payload("Done"))
    await make_engine(recorder).generate_turn_async(bot(), tool_result_history(content))
    part = recorder.bodies()[0]["contents"][2]["parts"][0]
    assert part["functionResponse"]["name"] == "status"
    return part["functionResponse"]["response"]  # type: ignore[no-any-return]


WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (WHEN, {"output": "2026-01-02T03:04:05+00:00"}),
        (date(2026, 1, 2), {"output": "2026-01-02"}),
        (Invoice(number=7, issued=WHEN), {"output": {"number": 7, "issued": "2026-01-02T03:04:05Z"}}),
        (Shipment(carrier="DHL", eta=date(2026, 1, 3)), {"output": {"carrier": "DHL", "eta": "2026-01-03"}}),
        (
            {"invoice": Invoice(number=8, issued=WHEN), "checked_at": WHEN},
            {"invoice": {"number": 8, "issued": "2026-01-02T03:04:05Z"}, "checked_at": "2026-01-02T03:04:05+00:00"},
        ),
        ({"tags": {"a"}}, {"tags": "{'a'}"}),
    ],
)
async def test_tool_results_json_cannot_encode_are_sent_structurally(content: Any, expected: Any) -> None:
    assert await send_tool_result(content) == expected


def circular_result() -> dict[str, Any]:
    result: dict[str, Any] = {"note": "SECRET-TOOL-OUTPUT"}
    result["self"] = result
    return result


@pytest.mark.parametrize(
    "content",
    [
        circular_result(),
        {"rows": {("SECRET-TOOL-OUTPUT", 1): "x"}},
        {"rows": [{frozenset({"SECRET-TOOL-OUTPUT"}): 1}]},
    ],
    ids=["circular", "nested-tuple-key", "frozenset-key-in-list"],
)
async def test_tool_results_json_cannot_encode_raise_provider_error_without_a_request(content: Any) -> None:
    recorder = Recorder(text_payload("unused"))
    with pytest.raises(ProviderError, match="request body could not be encoded as JSON") as excinfo:
        await make_engine(recorder).generate_turn_async(bot(), tool_result_history(content))
    assert "SECRET-TOOL-OUTPUT" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
    assert recorder.requests == []


async def test_tools_include_function_and_handoff_declarations() -> None:
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return "shipped"

    specialist = Agent(name="Billing Specialist", instructions="Billing.")
    agent = bot(tools=[lookup_order], handoffs=[specialist])
    recorder = Recorder(text_payload("Hi"))
    await make_engine(recorder).generate_turn_async(agent, USER_HELLO)

    tools = recorder.bodies()[0]["tools"]
    assert tools == [{"functionDeclarations": [t.to_gemini_declaration() for t in agent.get_all_tools()]}]
    names = [d["name"] for d in tools[0]["functionDeclarations"]]
    assert names[0] == "lookup_order"
    assert names[1].startswith("transfer_to_")


async def test_tools_key_is_omitted_when_agent_has_no_tools() -> None:
    recorder = Recorder(text_payload("Hi"))
    await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert "tools" not in recorder.bodies()[0]


# --- Response parsing --------------------------------------------------------------------


async def test_text_parts_are_concatenated_and_raw_response_is_the_dict() -> None:
    payload = text_payload("Hello, ", "world!")
    recorder = Recorder(payload)
    result = await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)

    assert result.text == "Hello, world!"
    assert result.tool_calls == []
    assert result.raw_response == payload


async def test_single_and_multiple_function_calls_get_part_index_ids() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me check."},
                        {"functionCall": {"name": "weather", "args": {"city": "Paris"}}},
                        {"functionCall": {"name": "clock"}, "thoughtSignature": "opaque"},
                    ],
                }
            }
        ]
    }
    result = await make_engine(Recorder(payload)).generate_turn_async(bot(), USER_HELLO)

    assert result.text == "Let me check."
    assert result.tool_calls == [
        ToolCall(id="call_weather_1", name="weather", args={"city": "Paris"}),
        ToolCall(id="call_clock_2", name="clock", args={}),
    ]


async def test_function_call_ids_from_gemini_are_kept_and_missing_ones_are_synthetic() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "weather", "args": {"city": "Paris"}, "id": "fc-weather"}},
                        {"functionCall": {"name": "clock", "id": ""}},
                        {"functionCall": {"name": "clock"}},
                    ],
                }
            }
        ]
    }
    result = await make_engine(Recorder(payload)).generate_turn_async(bot(), USER_HELLO)

    assert result.tool_calls == [
        ToolCall(id="fc-weather", name="weather", args={"city": "Paris"}),
        ToolCall(id="call_clock_1", name="clock", args={}),
        ToolCall(id="call_clock_2", name="clock", args={}),
    ]


async def test_only_the_first_candidate_is_used() -> None:
    payload = {"candidates": [text_payload("first")["candidates"][0], text_payload("second")["candidates"][0]]}
    result = await make_engine(Recorder(payload)).generate_turn_async(bot(), USER_HELLO)
    assert result.text == "first"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"candidates": None},
        {"candidates": []},
        {"candidates": [{}]},
        {"candidates": [{"content": None}]},
        {"candidates": [{"content": {}}]},
        {"candidates": [{"content": {"parts": []}}]},
        {"candidates": [{"finishReason": "SAFETY", "safetyRatings": []}]},
        {"candidates": [{"content": {"parts": [{"text": ""}]}}]},
    ],
)
async def test_empty_or_filtered_responses_return_no_text_and_no_calls(payload: dict[str, Any]) -> None:
    result = await make_engine(Recorder(payload)).generate_turn_async(bot(), USER_HELLO)
    assert result.text is None
    assert result.tool_calls == []
    assert result.raw_response == payload


async def test_blocked_prompt_raises_provider_error() -> None:
    payload = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(ProviderError, match="blockReason: SAFETY"):
        await make_engine(Recorder(payload)).generate_turn_async(bot(), USER_HELLO)


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, content=b"<html>Bad gateway</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"candidates": {"0": {}}}),
        httpx.Response(200, json={"candidates": ["text"]}),
        httpx.Response(200, json={"candidates": [{"content": "text"}]}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": "text"}}]}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": ["text"]}}]}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": 5}]}}]}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": [{"functionCall": "x"}]}}]}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": [{"functionCall": {"name": 1}}]}}]}),
        httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", "args": [1]}}]}}]}
        ),
        httpx.Response(200, json={"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", "id": 7}}]}}]}),
        httpx.Response(200, json={"promptFeedback": "blocked"}),
    ],
)
async def test_malformed_response_bodies_raise_provider_error(reply: httpx.Response) -> None:
    with pytest.raises(ProviderError, match="not valid JSON|malformed"):
        await make_engine(Recorder(reply)).generate_turn_async(bot(), USER_HELLO)


# --- Errors and retries ------------------------------------------------------------------


async def test_429_honours_retry_after_then_succeeds(caplog: pytest.LogCaptureFixture) -> None:
    recorder = Recorder(
        httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}}, headers={"Retry-After": "0.01"}),
        text_payload("Recovered"),
    )
    # base_delay is large: if Retry-After were ignored the computed delay would be ~3s.
    engine = make_engine(recorder, base_delay=3.0)
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.engine.gemini"):
        result = await engine.generate_turn_async(bot(), USER_HELLO)

    assert result.text == "Recovered"
    assert len(recorder.requests) == 2
    assert "HTTP 429. Retrying in 0.01s" in caplog.text


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retriable_statuses_are_retried_until_exhausted(status: int) -> None:
    recorder = Recorder(handler=lambda req: httpx.Response(status, json={"error": "busy"}))
    engine = make_engine(recorder, max_retries=2)
    with pytest.raises(ProviderError, match=f"\\({status}\\)"):
        await engine.generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_non_retriable_client_errors_raise_without_retry(status: int) -> None:
    recorder = Recorder(httpx.Response(status, json={"error": {"message": "API key not valid"}}))
    with pytest.raises(ProviderError, match=f"\\({status}\\).*API key not valid"):
        await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("status", [301, 302, 307])
async def test_redirect_statuses_raise_without_retry_even_with_a_json_body(status: int) -> None:
    recorder = Recorder(
        httpx.Response(status, json=text_payload("from a redirect"), headers={"Location": "https://example.invalid/"})
    )
    with pytest.raises(ProviderError, match=f"Gemini API request failed \\({status}\\)"):
        await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("in_query", [False, True], ids=["header-key", "query-key"])
@pytest.mark.parametrize("path", ["injected-client", "per-attempt-client", "pooled-client"])
async def test_redirects_are_not_followed_even_by_clients_configured_to_follow_them(
    monkeypatch: pytest.MonkeyPatch, path: str, in_query: bool, status: int
) -> None:
    """httpx keeps x-goog-api-key on a cross-origin redirect, so following one would leak the key."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "evil.example":
            return httpx.Response(200, json=text_payload("from the redirect target"))
        return httpx.Response(status, headers={"Location": "https://evil.example/collect"})

    recorder = Recorder(handler=handler)
    created: list[str] = []

    def following_client(kind: str, **kwargs: Any) -> httpx.AsyncClient:
        created.append(kind)
        return httpx.AsyncClient(transport=recorder.transport, follow_redirects=True, **kwargs)

    monkeypatch.setattr(gemini_module, "_new_attempt_client", lambda timeout: following_client("per-attempt", timeout=timeout))
    monkeypatch.setattr(gemini_module, "_new_pooled_client", lambda: following_client("pooled"))
    injected = following_client("injected") if path == "injected-client" else None
    engine = GeminiEngine(api_key=API_KEY, api_key_in_query=in_query, http_client=injected)

    with pytest.raises(ProviderError, match=f"Gemini API request failed \\({status}\\)") as excinfo:
        if path == "pooled-client":
            runner = Runner(bot(), session_store=SessionStore(), engine=engine)
            async with runner:
                await runner.run_async("s1", "Hello")
        else:
            await engine.generate_turn_async(bot(), USER_HELLO)

    assert created == [path.removesuffix("-client")]
    assert [r.url.host for r in recorder.requests] == ["generativelanguage.googleapis.com"]
    assert API_KEY not in str(excinfo.value)
    if injected is not None:
        assert not injected.is_closed
        await injected.aclose()


async def test_long_error_bodies_are_truncated_in_provider_errors() -> None:
    recorder = Recorder(httpx.Response(400, text="x" * 20000))
    with pytest.raises(ProviderError, match="\\(400\\)") as excinfo:
        await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    message = str(excinfo.value)
    assert "x" * 1000 in message
    assert "x" * 1001 not in message
    assert message.endswith("… (truncated)")


BOUNDARY_KEY = "AIzaBoundaryKey0123456789"


@pytest.mark.parametrize(
    "offset",
    # Key ends exactly at the 1000-character cut, straddles it, or starts one character before it.
    [1000 - len(BOUNDARY_KEY), 1001 - len(BOUNDARY_KEY), 990, 999],
)
@pytest.mark.parametrize("status", [400, 503])
async def test_api_key_is_redacted_before_the_error_body_is_truncated(offset: int, status: int) -> None:
    recorder = Recorder(httpx.Response(status, text="x" * offset + BOUNDARY_KEY + "y" * 2000))
    engine = GeminiEngine(
        api_key=BOUNDARY_KEY, max_retries=0, http_client=httpx.AsyncClient(transport=recorder.transport)
    )
    with pytest.raises(ProviderError, match=f"\\({status}\\)") as excinfo:
        await engine.generate_turn_async(bot(), USER_HELLO)

    message = str(excinfo.value)
    excerpt = message.split(": ", 1)[1]
    assert len(excerpt) <= 1000 + len("… (truncated)")
    assert excerpt.startswith("x" * offset + "*")
    for length in range(4, len(BOUNDARY_KEY) + 1):
        assert BOUNDARY_KEY[:length] not in message, f"key fragment leaked: {BOUNDARY_KEY[:length]!r}"


@pytest.mark.parametrize("error_cls", [httpx.ConnectError, httpx.ConnectTimeout])
async def test_connection_failures_are_retried(error_cls: type[httpx.RequestError]) -> None:
    recorder = Recorder(error_cls("boom"), text_payload("Recovered"))
    result = await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert result.text == "Recovered"
    assert len(recorder.requests) == 2


@pytest.mark.parametrize("error_cls", [httpx.ReadTimeout, httpx.WriteTimeout])
async def test_read_and_write_timeouts_are_not_retried(
    error_cls: type[httpx.RequestError], caplog: pytest.LogCaptureFixture
) -> None:
    """The request may already have reached Gemini, so sending it again could repeat it."""
    recorder = Recorder(error_cls("timed out"), text_payload("unused"))
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.engine.gemini"):
        with pytest.raises(ProviderError, match=f"Gemini API request failed \\({error_cls.__name__}\\): timed out"):
            await make_engine(recorder, max_retries=3).generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 1
    assert "Retrying" not in caplog.text


async def test_connect_errors_raise_provider_error_when_retries_are_exhausted() -> None:
    recorder = Recorder(handler=lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(ProviderError, match="refused"):
        await make_engine(recorder, max_retries=1).generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 2


async def test_other_transport_errors_are_not_retried() -> None:
    recorder = Recorder(httpx.ReadError("connection reset"))
    with pytest.raises(ProviderError, match="connection reset"):
        await make_engine(recorder).generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 1


async def test_timeout_seconds_bounds_each_attempt_and_timeouts_are_not_retried() -> None:
    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json=text_payload("late"))

    recorder = Recorder(handler=hang)
    engine = make_engine(recorder, timeout_seconds=0.05, max_retries=1)
    with pytest.raises(ProviderError, match="Gemini API request failed \\(TimeoutError\\)"):
        await engine.generate_turn_async(bot(), USER_HELLO)
    assert len(recorder.requests) == 1


# httpcore trace events (as httpcore 1.0 names them) reported before an attempt runs out of time.
CONNECTING = ("connection.connect_tcp.started",)
TLS_HANDSHAKE = ("connection.connect_tcp.started", "connection.connect_tcp.complete", "connection.start_tls.started")
SENDING = ("connection.connect_tcp.started", "connection.connect_tcp.complete", "http11.send_request_headers.started")
ON_A_REUSED_CONNECTION = ("http11.send_request_headers.started", "http11.send_request_headers.complete")


@pytest.mark.parametrize(
    ("events", "retried"),
    [(CONNECTING, True), (TLS_HANDSHAKE, True), (SENDING, False), (ON_A_REUSED_CONNECTION, False)],
    ids=["connecting", "tls-handshake", "sending", "reused-connection"],
)
@pytest.mark.parametrize("path", ["injected-client", "per-attempt-client", "pooled-client"])
async def test_an_attempt_that_runs_out_of_time_is_retried_only_while_opening_a_connection(
    monkeypatch: pytest.MonkeyPatch, path: str, events: tuple[str, ...], retried: bool
) -> None:
    """A connection that hangs while it is being opened has sent nothing, so retrying is safe."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if len(recorder.requests) == 1:
            for event in events:
                await request.extensions["trace"](event, {})
            await asyncio.sleep(10)
        return httpx.Response(200, json=text_payload("Recovered"))

    recorder = Recorder(handler=handler)
    monkeypatch.setattr(
        gemini_module, "_new_attempt_client", lambda timeout: httpx.AsyncClient(transport=recorder.transport)
    )
    monkeypatch.setattr(gemini_module, "_new_pooled_client", lambda: httpx.AsyncClient(transport=recorder.transport))
    injected = httpx.AsyncClient(transport=recorder.transport) if path == "injected-client" else None
    engine = GeminiEngine(api_key=API_KEY, timeout_seconds=0.05, max_retries=1, base_delay=0.001, http_client=injected)

    async def turn() -> str | None:
        if path == "pooled-client":
            runner = Runner(bot(), session_store=SessionStore(), engine=engine)
            async with runner:
                return (await runner.run_async("s1", "Hello")).content
        return (await engine.generate_turn_async(bot(), USER_HELLO)).text

    if retried:
        assert await turn() == "Recovered"
        assert len(recorder.requests) == 2
    else:
        with pytest.raises(ProviderError, match="Gemini API request failed \\(TimeoutError\\)"):
            await turn()
        assert len(recorder.requests) == 1


@pytest.mark.parametrize(
    "reply",
    [
        httpx.ConnectError(f"cannot reach {ENDPOINT}?key={API_KEY}"),
        httpx.Response(401, text=f"API key {API_KEY} not valid"),
    ],
)
async def test_api_key_is_redacted_from_provider_errors(reply: Any) -> None:
    engine = make_engine(Recorder(reply), max_retries=0, api_key_in_query=True)
    with pytest.raises(ProviderError) as excinfo:
        await engine.generate_turn_async(bot(), USER_HELLO)
    assert API_KEY not in str(excinfo.value)
    assert "***" in str(excinfo.value)


# Contains characters that URL encoding changes, so the encoded form differs from the raw one.
URL_KEY = "AIza+test/key=123"


def key_forms(key: str) -> list[str]:
    return [key, quote(key, safe=""), key.encode("unicode_escape").decode()]


def assert_no_key(text: str, key: str = URL_KEY) -> None:
    for form in key_forms(key):
        assert form not in text, f"API key leaked as {form!r}: {text!r}"


async def test_api_key_is_redacted_from_retry_warnings_and_final_error(caplog: pytest.LogCaptureFixture) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url} with key {URL_KEY}")

    recorder = Recorder(handler=refuse)
    engine = GeminiEngine(
        api_key=URL_KEY,
        api_key_in_query=True,
        max_retries=2,
        base_delay=0.001,
        http_client=httpx.AsyncClient(transport=recorder.transport),
    )
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.engine.gemini"):
        with pytest.raises(ProviderError) as excinfo:
            await engine.generate_turn_async(bot(), USER_HELLO)

    assert quote(URL_KEY, safe="") in str(recorder.requests[0].url)  # the encoded form was really sent
    assert len(caplog.records) == 2
    assert_no_key(caplog.text)
    assert_no_key(str(excinfo.value))
    assert "***" in caplog.text and "***" in str(excinfo.value)


async def test_api_key_is_redacted_from_non_retried_request_errors() -> None:
    recorder = Recorder(httpx.ReadError(f"reset while sending b'{URL_KEY}' to {quote(URL_KEY, safe='')}"))
    with pytest.raises(ProviderError, match="connection error") as excinfo:
        await GeminiEngine(api_key=URL_KEY, http_client=httpx.AsyncClient(transport=recorder.transport)).generate_turn_async(
            bot(), USER_HELLO
        )
    assert len(recorder.requests) == 1
    assert_no_key(str(excinfo.value))


async def test_api_key_is_redacted_from_injected_client_runtime_errors() -> None:
    recorder = Recorder(RuntimeError(f"client for key {URL_KEY} is bound to a closed event loop"))
    engine = GeminiEngine(api_key=URL_KEY, http_client=httpx.AsyncClient(transport=recorder.transport))
    with pytest.raises(ProviderError, match="http_client is closed") as excinfo:
        await engine.generate_turn_async(bot(), USER_HELLO)
    assert_no_key(str(excinfo.value))


@pytest.mark.parametrize(
    ("error_cls", "expected"),
    [
        (httpx.ConnectError, "Gemini API request failed (ConnectError)"),
        (httpx.ReadTimeout, "Gemini API request failed (ReadTimeout)"),
        (httpx.ReadError, "Gemini API connection error (ReadError)"),
        (httpx.InvalidURL, "Gemini API request URL is invalid"),
        (RuntimeError, "the provided http_client is closed"),
    ],
)
@pytest.mark.parametrize("in_query", [False, True], ids=["header-key", "query-key"])
async def test_api_key_is_absent_from_the_formatted_traceback_and_exception_chain_of_provider_errors(
    error_cls: type[Exception], expected: str, in_query: bool
) -> None:
    """The original exception, and the one chained to it, are neither part of the traceback
    nor reachable from the error."""

    def fail(request: httpx.Request) -> httpx.Response:
        try:
            raise OSError(f"socket error while sending {URL_KEY}")
        except OSError as inner:
            if issubclass(error_cls, httpx.RequestError):
                # As httpx's transports do: the request, with the key in its URL or headers.
                raise error_cls(f"failed to send {request.url} with key {URL_KEY}", request=request) from inner
            raise error_cls(f"failed to send {request.url} with key {URL_KEY}") from inner

    recorder = Recorder(handler=fail)
    engine = GeminiEngine(
        api_key=URL_KEY,
        api_key_in_query=in_query,
        max_retries=1,
        base_delay=0.001,
        http_client=httpx.AsyncClient(transport=recorder.transport),
    )
    with pytest.raises(ProviderError) as excinfo:
        await engine.generate_turn_async(bot(), USER_HELLO)

    formatted = "".join(traceback.format_exception(excinfo.value))
    assert_no_key(formatted)
    assert "socket error" not in formatted
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
    # Code that walks __context__ regardless of __suppress_context__ finds nothing either.
    assert excinfo.value.__context__ is None
    assert expected in formatted


async def test_engine_owned_client_runtime_errors_are_not_wrapped(
    attempt_clients: Callable[[Recorder], list[httpx.AsyncClient]],
) -> None:
    attempt_clients(Recorder(RuntimeError("unexpected")))
    with pytest.raises(RuntimeError, match="unexpected"):
        await GeminiEngine(api_key=API_KEY).generate_turn_async(bot(), USER_HELLO)


@pytest.mark.parametrize(
    ("variable", "value", "secrets"),
    [
        ("SSL_CERT_FILE", "/nonexistent/secret-dir/ca.pem", ["secret-dir", "ca.pem"]),
        (
            "HTTPS_PROXY",
            "socks9://proxy-user:proxy-pass@proxy.internal:1080",
            ["proxy-user", "proxy-pass", "proxy.internal"],
        ),
    ],
    ids=["missing-ca-file", "unsupported-proxy-scheme"],
)
@pytest.mark.parametrize("in_scope", [False, True], ids=["per-attempt-client", "pooled-client"])
async def test_http_client_creation_errors_raise_provider_error_without_details(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, secrets: list[str], in_scope: bool
) -> None:
    """httpx reads proxy and TLS settings from the environment when a client is created."""
    monkeypatch.setenv(variable, value)
    runner = Runner(bot(), session_store=SessionStore(), engine=GeminiEngine(api_key=API_KEY))
    with pytest.raises(ProviderError, match="HTTP client could not be created") as excinfo:
        if in_scope:
            async with runner:
                await runner.run_async("s1", "Hello")
        else:
            await runner.run_async("s1", "Hello")

    message = str(excinfo.value)
    for secret in [API_KEY, value, *secrets]:
        assert secret not in message
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
    # The original error quotes the proxy URL or the file path; it is not reachable either.
    assert excinfo.value.__context__ is None
    assert runner._client_pools == {}


@pytest.mark.usefixtures("no_env_keys")
@pytest.mark.parametrize("raw", ["KEY-123\n", "KEY-123\r\n", "\tKEY-123 ", "  KEY-123\r"])
async def test_surrounding_whitespace_is_stripped_from_the_api_key(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", raw)
    recorder = Recorder(text_payload("Hi"))
    engine = GeminiEngine(http_client=httpx.AsyncClient(transport=recorder.transport))
    assert engine.api_key == "KEY-123"
    await engine.generate_turn_async(bot(), USER_HELLO)
    assert recorder.requests[0].headers["x-goog-api-key"] == "KEY-123"
    assert GeminiEngine(api_key=raw).api_key == "KEY-123"


@pytest.mark.usefixtures("no_env_keys")
def test_whitespace_only_api_key_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert GeminiEngine(api_key=" \n").api_key == "env-key"
    assert GeminiEngine(api_key="").api_key == "env-key"


@pytest.mark.parametrize("key", ["SECRET\rKEY", "SECRET\nKEY", "SECRET\tKEY", "SECRET KEY", "SECRET\x00KEY", "SECRÉTKEY"])
@pytest.mark.parametrize("in_query", [False, True])
async def test_invalid_api_keys_are_rejected_before_any_request_without_leaking(key: str, in_query: bool) -> None:
    recorder = Recorder(text_payload("Hi"))
    engine = GeminiEngine(
        api_key=key, api_key_in_query=in_query, http_client=httpx.AsyncClient(transport=recorder.transport)
    )
    with pytest.raises(ProviderError, match="Gemini API key is invalid") as excinfo:
        await engine.generate_turn_async(bot(), USER_HELLO)
    assert recorder.requests == []
    message = str(excinfo.value)
    for form in (key, key.encode("unicode_escape").decode(), ascii(key)[1:-1], "SECRET"):
        assert form not in message


# --- Backoff -----------------------------------------------------------------------------


def test_backoff_grows_exponentially_with_bounded_jitter() -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=100.0)
    for retries, base in enumerate([1.0, 2.0, 4.0, 8.0]):
        for _ in range(20):
            assert base <= engine._retry_delay(retries) <= base * 1.25


def test_backoff_is_capped_at_max_delay() -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=3.0)
    assert engine._retry_delay(5) == 3.0
    assert engine._retry_delay(0, "3600") == 3.0


@pytest.mark.parametrize("retry_after", ["2.5", "0"])
def test_numeric_retry_after_replaces_the_exponential_base(retry_after: str) -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=100.0)
    base = float(retry_after)
    for _ in range(20):
        assert base <= engine._retry_delay(3, retry_after) <= base * 1.25


@pytest.mark.parametrize(
    "retry_after",
    ["soon", "-5", "nan", "inf", "", "1e999", "Wed, 32 Oct 2015 07:28:00 GMT", "Wed, 21 Foo 2015 07:28:00 GMT"],
)
def test_unusable_retry_after_falls_back_to_exponential_backoff(retry_after: str) -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=100.0)
    for _ in range(20):
        assert 4.0 <= engine._retry_delay(2, retry_after) <= 5.0


def http_date(seconds_from_now: float) -> str:
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now), usegmt=True)


def test_future_http_date_retry_after_replaces_the_exponential_base() -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=100.0)
    for _ in range(20):
        # HTTP-dates have a resolution of one second.
        assert 28.0 <= engine._retry_delay(0, http_date(30)) <= 30.0 * 1.25


@pytest.mark.parametrize("retry_after", ["Wed, 21 Oct 2015 07:28:00 GMT", "Wed, 21 Oct 2015 07:28:00 -0000"])
def test_past_http_date_retry_after_means_no_delay(retry_after: str) -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=100.0)
    assert engine._retry_delay(2, retry_after) == 0.0


def test_http_date_retry_after_is_capped_at_max_delay() -> None:
    engine = GeminiEngine(api_key=API_KEY, base_delay=1.0, max_delay=3.0)
    assert engine._retry_delay(0, http_date(3600)) == 3.0


async def test_http_date_retry_after_is_honoured_when_retrying(caplog: pytest.LogCaptureFixture) -> None:
    recorder = Recorder(
        httpx.Response(503, json={}, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        text_payload("Recovered"),
    )
    # base_delay is large: if the date were ignored the computed delay would be ~3s.
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.engine.gemini"):
        result = await make_engine(recorder, base_delay=3.0).generate_turn_async(bot(), USER_HELLO)
    assert result.text == "Recovered"
    assert len(recorder.requests) == 2
    assert "HTTP 503. Retrying in 0.00s" in caplog.text


@pytest.mark.parametrize(
    "retry_after",
    ["99999999999999999999", "1" + "0" * 400, "Fri, 31 Dec 9999 23:59:59 GMT"],
    ids=["20-digits", "too-large-for-a-float", "far-future-http-date"],
)
async def test_excessive_retry_after_waits_exactly_max_delay(
    retry_after: str, caplog: pytest.LogCaptureFixture
) -> None:
    """max_delay bounds the wait however long the server asks for, even beyond a float."""
    recorder = Recorder(httpx.Response(429, json={}, headers={"Retry-After": retry_after}), text_payload("Recovered"))
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.engine.gemini"):
        result = await make_engine(recorder, max_delay=0.05).generate_turn_async(bot(), USER_HELLO)
    assert result.text == "Recovered"
    assert len(recorder.requests) == 2
    assert "HTTP 429. Retrying in 0.05s" in caplog.text


@pytest.mark.parametrize("retry_after", ["soon", "-5", "1e999", "Wed, 32 Oct 2015 07:28:00 GMT"])
async def test_malformed_retry_after_falls_back_to_exponential_backoff_when_retrying(
    retry_after: str, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(httpx.Response(503, json={}, headers={"Retry-After": retry_after}), text_payload("Recovered"))
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.engine.gemini"):
        result = await make_engine(recorder, base_delay=0.04).generate_turn_async(bot(), USER_HELLO)
    assert result.text == "Recovered"
    assert len(recorder.requests) == 2
    retrying = re.search(r"HTTP 503\. Retrying in ([0-9.]+)s", caplog.text)
    assert retrying is not None
    assert 0.04 <= float(retrying.group(1)) <= 0.05  # base_delay plus at most 25% jitter


@pytest.mark.parametrize(
    "retry_after",
    ["Sunday, 06-Nov-94 08:49:37 GMT", "Sun Nov  6 08:49:37 1994"],
    ids=["rfc850-date", "asctime-date"],
)
def test_obsolete_http_date_formats_are_accepted(retry_after: str) -> None:
    """RFC 9110 requires recipients to accept them. These dates are past, so there is no delay;
    were they not parsed, the delay would be the 50 s exponential base."""
    engine = GeminiEngine(api_key=API_KEY, base_delay=50.0, max_delay=100.0)
    assert engine._retry_delay(0, retry_after) == 0.0


async def test_cancellation_propagates_as_cancelled_error() -> None:
    started = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json=text_payload("late"))

    task = asyncio.create_task(make_engine(Recorder(handler=hang)).generate_turn_async(bot(), USER_HELLO))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- API key resolution and constructor -------------------------------------------------


@pytest.mark.usefixtures("no_env_keys")
async def test_missing_api_key_raises_at_call_time_without_creating_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_client(timeout: float) -> httpx.AsyncClient:
        raise AssertionError("no client should be created without an API key")

    monkeypatch.setattr(gemini_module, "_new_attempt_client", no_client)
    engine = GeminiEngine()
    with pytest.raises(ProviderError, match="Gemini API client is not configured"):
        await engine.generate_turn_async(bot(), USER_HELLO)


@pytest.mark.usefixtures("no_env_keys")
def test_api_key_precedence_argument_then_gemini_then_google(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GeminiEngine().api_key is None
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert GeminiEngine().api_key == "google-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert GeminiEngine().api_key == "gemini-key"
    assert GeminiEngine(api_key="explicit").api_key == "explicit"


@pytest.mark.usefixtures("no_env_keys")
def test_constructor_signature_and_defaults_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    params = inspect.signature(GeminiEngine.__init__).parameters
    positional = [
        (name, p.default)
        for name, p in params.items()
        if name != "self" and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
        ("api_key", None),
        ("timeout_seconds", 30.0),
        ("max_retries", 3),
        ("base_delay", 1.0),
        ("max_delay", 10.0),
        ("http_client", None),
    ]
    assert params["api_key_in_query"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["api_key_in_query"].default is False

    def no_client(timeout: float) -> httpx.AsyncClient:
        raise AssertionError("the constructor must not create a client")

    monkeypatch.setattr(gemini_module, "_new_attempt_client", no_client)
    engine = GeminiEngine()
    assert (engine.timeout_seconds, engine.max_retries, engine.base_delay, engine.max_delay) == (
        30.0,
        3,
        1.0,
        10.0,
    )
    assert engine.api_key_in_query is False


# --- Client ownership ----------------------------------------------------------------------


async def test_injected_client_is_used_for_every_attempt_and_never_closed() -> None:
    recorder = Recorder(
        httpx.Response(503, json={}),
        text_payload("ok"),
        httpx.Response(400, json={"error": "bad"}),
    )
    client = httpx.AsyncClient(transport=recorder.transport)
    engine = GeminiEngine(api_key=API_KEY, base_delay=0.001, http_client=client)

    assert (await engine.generate_turn_async(bot(), USER_HELLO)).text == "ok"
    with pytest.raises(ProviderError):
        await engine.generate_turn_async(bot(), USER_HELLO)

    assert len(recorder.requests) == 3
    assert not client.is_closed
    await client.aclose()


async def test_closed_injected_client_raises_provider_error() -> None:
    client = httpx.AsyncClient(transport=Recorder(text_payload("ok")).transport)
    await client.aclose()
    engine = GeminiEngine(api_key=API_KEY, http_client=client)
    with pytest.raises(ProviderError, match="http_client is closed"):
        await engine.generate_turn_async(bot(), USER_HELLO)


async def test_engine_created_clients_are_closed_after_success_errors_and_retries(
    attempt_clients: Callable[[Recorder], list[httpx.AsyncClient]],
) -> None:
    recorder = Recorder(
        text_payload("ok"),
        httpx.Response(503, json={}),
        text_payload("ok"),
        httpx.Response(400, json={"error": "bad"}),
        httpx.ReadError("reset"),
    )
    created = attempt_clients(recorder)
    engine = GeminiEngine(api_key=API_KEY, base_delay=0.001, timeout_seconds=7.5)

    await engine.generate_turn_async(bot(), USER_HELLO)
    await engine.generate_turn_async(bot(), USER_HELLO)
    for _ in range(2):
        with pytest.raises(ProviderError):
            await engine.generate_turn_async(bot(), USER_HELLO)

    assert len(created) == 5  # one client per HTTP attempt
    assert all(client.is_closed for client in created)
    assert all(client.timeout == httpx.Timeout(7.5) for client in created)


async def test_engine_created_clients_are_closed_after_timeout_and_cancellation(
    attempt_clients: Callable[[Recorder], list[httpx.AsyncClient]],
) -> None:
    started = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json=text_payload("late"))

    created = attempt_clients(Recorder(handler=hang))
    engine = GeminiEngine(api_key=API_KEY, timeout_seconds=0.05, max_retries=0)
    with pytest.raises(ProviderError):
        await engine.generate_turn_async(bot(), USER_HELLO)

    engine.timeout_seconds = 30.0
    task = asyncio.create_task(engine.generate_turn_async(bot(), USER_HELLO))
    started.clear()
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(created) == 2
    assert all(client.is_closed for client in created)


# --- Through the Runner ------------------------------------------------------------------


async def test_runner_multi_turn_conversation_preserves_context() -> None:
    recorder = Recorder(text_payload("Hi Ana!"), text_payload("Your name is Ana."))
    runner = Runner(bot(), session_store=SessionStore(), engine=make_engine(recorder))

    await runner.run_async("s1", "I am Ana")
    response = await runner.run_async("s1", "What is my name?")

    assert response.content == "Your name is Ana."
    assert recorder.bodies()[1]["contents"] == [
        {"role": "user", "parts": [{"text": "I am Ana"}]},
        {"role": "model", "parts": [{"text": "Hi Ana!"}]},
        {"role": "user", "parts": [{"text": "What is my name?"}]},
    ]


async def test_runner_tool_call_round_trip() -> None:
    def get_weather(city: str) -> dict[str, Any]:
        """Get the weather for a city."""
        return {"city": city, "forecast": "sunny"}

    recorder = Recorder(call_payload(("get_weather", {"city": "Paris"})), text_payload("Sunny in Paris."))
    runner = Runner(bot(tools=[get_weather]), session_store=SessionStore(), engine=make_engine(recorder))

    response = await runner.run_async("s1", "Weather in Paris?")

    assert response.content == "Sunny in Paris."
    assert response.tool_calls_executed == ["get_weather"]
    contents = recorder.bodies()[1]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"city": "Paris", "forecast": "sunny"},
                }
            }
        ],
    }


@pytest.mark.parametrize(
    "gemini_ids",
    [("fc-weather", "fc-time"), (None, "fc-time"), (None, None)],
    ids=["gemini-ids", "mixed", "no-ids"],
)
async def test_runner_sends_gemini_function_call_ids_back_with_the_matching_responses(
    gemini_ids: tuple[str | None, str | None],
) -> None:
    def get_weather(city: str) -> dict[str, Any]:
        """Get the weather for a city."""
        return {"forecast": "sunny"}

    def get_time(city: str) -> str:
        """Get the local time in a city."""
        return "noon"

    def with_id(part: dict[str, Any], call_id: str | None) -> dict[str, Any]:
        return part if call_id is None else {**part, "id": call_id}

    weather_call = {"name": "get_weather", "args": {"city": "Paris"}}
    time_call = {"name": "get_time", "args": {"city": "Paris"}}
    weather_id, time_id = gemini_ids
    calls = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": with_id(weather_call, weather_id)},
                        {"functionCall": with_id(time_call, time_id)},
                    ],
                }
            }
        ]
    }
    recorder = Recorder(calls, text_payload("Sunny, noon."))
    runner = Runner(bot(tools=[get_weather, get_time]), session_store=SessionStore(), engine=make_engine(recorder))

    await runner.run_async("s1", "Weather and time in Paris?")

    # Only IDs Gemini issued are sent; synthetic ones never leave the engine.
    contents = recorder.bodies()[1]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": with_id(weather_call, weather_id)}, {"functionCall": with_id(time_call, time_id)}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": with_id({"name": "get_weather", "response": {"forecast": "sunny"}}, weather_id)},
            {"functionResponse": with_id({"name": "get_time", "response": {"output": "noon"}}, time_id)},
        ],
    }


async def test_runner_handoff_through_gemini_engine() -> None:
    specialist = Agent(name="Billing", model="gemini-2.5-flash", instructions="You handle billing.")
    triage = bot(handoffs=[specialist])
    handoff_tool = triage.get_all_tools()[0].name

    recorder = Recorder(call_payload((handoff_tool, {"reason": "billing"})), text_payload("Billing here."))
    runner = Runner(triage, session_store=SessionStore(), engine=make_engine(recorder))

    response = await runner.run_async("s1", "I was double charged")

    assert response.active_agent_name == "Billing"
    assert response.content == "Billing here."
    second = recorder.bodies()[1]
    assert second["systemInstruction"]["parts"][0]["text"].startswith("You handle billing.")
    response_part = second["contents"][2]["parts"][0]["functionResponse"]
    assert response_part["name"] == handoff_tool
    assert response_part["response"]["status"] == "HANDOFF_SUCCESS"


@pytest.mark.usefixtures("no_env_keys")
def test_resolution_and_default_runner_need_no_api_key() -> None:
    assert isinstance(resolve_engine(Agent(name="A", instructions="x", model="gemini-2.5-pro")), GeminiEngine)
    assert isinstance(
        resolve_engine(Agent(name="B", instructions="x", model="anything", provider="gemini")), GeminiEngine
    )
    runner = Runner(bot())
    assert isinstance(runner.engine, GeminiEngine)
    assert runner.engine.api_key is None
