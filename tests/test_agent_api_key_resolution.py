"""Agent(api_key=...) on the default provider path (issue #163).

An API key set on an Agent is the key of the engine its configuration selects. On the
default Gemini path, the Runner's built-in GeminiEngine is used with the agent's key
instead of the Runner's; the provider never changes, and engines supplied explicitly
(``Agent(engine=...)``, ``Runner(engine=...)``, subclasses, mocks) are never replaced.
All keys are fictitious and every request is served by ``httpx.MockTransport``.
"""

import os
from typing import Any

import httpx
import pytest

from chatflow_agent import Agent, Runner
from chatflow_agent.core.engines import (
    AnthropicEngine,
    GeminiEngine,
    OpenAIEngine,
    resolve_engine,
)
from chatflow_agent.core.engines import gemini as gemini_module
from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message

AGENT_KEY = "AIza-FAKE-agent-0000"
RUNNER_KEY = "AIza-FAKE-runner-0000"
ENV_KEY = "AIza-FAKE-env-0000"
GEMINI_HOST = "generativelanguage.googleapis.com"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real credential or endpoint override may leak into these tests."""
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_BASE_URL")):
            monkeypatch.delenv(name)


def _agent(**kwargs: Any) -> Agent:
    kwargs.setdefault("model", "gemini-2.5-flash")
    return Agent(name=kwargs.pop("name", "Planner"), instructions="Reply.", **kwargs)


class _Recorder:
    """Records the host and credential of every Gemini request; replies with plain text."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[tuple[str, str | None, str]] = []
        self.replies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            (request.url.host, request.headers.get("x-goog-api-key"), str(request.url))
        )
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "denied"}})
        body = self.replies.pop(0) if self.replies else _text("ok")
        return httpx.Response(200, json=body)

    def keys(self) -> list[str | None]:
        return [key for _, key, _ in self.requests]


def _text(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(
        gemini_module,
        "_new_attempt_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(rec.handler), timeout=timeout),
    )
    return rec


# --- resolve_engine: the default Gemini path -----------------------------------------------


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "my-custom-model"])
def test_agent_key_replaces_the_key_of_the_runner_default_gemini_engine(model: str) -> None:
    default = GeminiEngine(api_key=RUNNER_KEY)
    engine = resolve_engine(_agent(model=model, api_key=AGENT_KEY), default_engine=default)
    assert type(engine) is GeminiEngine  # the provider stays Gemini
    assert engine is not default
    assert engine.api_key == AGENT_KEY
    assert default.api_key == RUNNER_KEY  # the shared default engine is not mutated


def test_copy_keeps_every_other_setting_of_the_default_engine() -> None:
    client = httpx.AsyncClient()
    default = GeminiEngine(
        api_key=RUNNER_KEY,
        timeout_seconds=5.0,
        max_retries=1,
        base_delay=0.5,
        max_delay=2.0,
        http_client=client,
        api_key_in_query=True,
    )
    engine = resolve_engine(_agent(api_key=AGENT_KEY), default_engine=default)
    assert isinstance(engine, GeminiEngine)
    assert engine.api_key == AGENT_KEY
    assert (engine.timeout_seconds, engine.max_retries, engine.base_delay, engine.max_delay) == (
        5.0,
        1,
        0.5,
        2.0,
    )
    assert engine._external_client is client
    assert engine.api_key_in_query is True
    # Every attribute but the key matches, so a setting added later cannot be dropped.
    assert {k: v for k, v in vars(engine).items() if k != "api_key"} == {
        k: v for k, v in vars(default).items() if k != "api_key"
    }


@pytest.mark.parametrize("agent_key", [None, "", "   \n"], ids=["none", "empty", "blank"])
def test_without_an_agent_key_the_default_engine_is_used_unchanged(agent_key: str | None) -> None:
    default = GeminiEngine(api_key=RUNNER_KEY)
    assert resolve_engine(_agent(api_key=agent_key), default_engine=default) is default


def test_agent_key_equal_to_the_default_key_keeps_the_default_engine() -> None:
    default = GeminiEngine(api_key=AGENT_KEY)
    assert resolve_engine(_agent(api_key=f" {AGENT_KEY}\n"), default_engine=default) is default


def test_custom_fallback_engines_are_never_replaced() -> None:
    class CustomGemini(GeminiEngine):
        pass

    class Mock(BaseEngine):
        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            return EngineTurnResult(text="mock")

    agent = _agent(api_key=AGENT_KEY)
    for fallback in (CustomGemini(api_key=RUNNER_KEY), Mock(), OpenAIEngine(api_key="sk-FAKE-0000")):
        assert resolve_engine(agent, default_engine=fallback) is fallback


def test_explicit_engines_win_over_the_agent_key() -> None:
    explicit = GeminiEngine(api_key=RUNNER_KEY)
    agent = _agent(api_key=AGENT_KEY)
    # Runner(engine=...) (step 4) and Agent(engine=...) (step 1) are deliberate choices.
    assert resolve_engine(agent, default_engine=explicit, prefer_default_engine=True) is explicit
    pinned = GeminiEngine(api_key="AIza-FAKE-pinned-0000")
    pinned_agent = _agent(api_key=AGENT_KEY, engine=pinned)
    assert resolve_engine(pinned_agent, default_engine=explicit) is pinned


def test_without_a_default_engine_the_gemini_fallback_uses_the_agent_key() -> None:
    engine = resolve_engine(_agent(api_key=AGENT_KEY))
    assert isinstance(engine, GeminiEngine)
    assert engine.api_key == AGENT_KEY


# --- other configurations keep their behavior ------------------------------------------------


def test_explicit_provider_uses_the_agent_key() -> None:
    default = GeminiEngine(api_key=RUNNER_KEY)
    gemini = resolve_engine(_agent(provider="gemini", api_key=AGENT_KEY), default_engine=default)
    assert isinstance(gemini, GeminiEngine) and gemini.api_key == AGENT_KEY
    openai = resolve_engine(
        _agent(provider="openai", model="gpt-4o", api_key="sk-FAKE-openai-0000"), default_engine=default
    )
    assert isinstance(openai, OpenAIEngine) and openai.api_key == "sk-FAKE-openai-0000"
    claude = resolve_engine(
        _agent(provider="anthropic", model="claude-x", api_key="sk-FAKE-ant-0000"), default_engine=default
    )
    assert isinstance(claude, AnthropicEngine) and claude.api_key == "sk-FAKE-ant-0000"


def test_explicit_base_url_uses_the_agent_key() -> None:
    default = GeminiEngine(api_key=RUNNER_KEY)
    engine = resolve_engine(
        _agent(model="my-model", base_url="https://proxy.example.com/v1", api_key="sk-FAKE-proxy-0000"),
        default_engine=default,
    )
    assert isinstance(engine, OpenAIEngine)
    assert engine.base_url == "https://proxy.example.com/v1"
    assert engine.api_key == "sk-FAKE-proxy-0000"


def test_model_prefixes_keep_their_provider_and_use_the_agent_key() -> None:
    default = GeminiEngine(api_key=RUNNER_KEY)
    engine = resolve_engine(_agent(model="gpt-4o", api_key="sk-FAKE-openai-0000"), default_engine=default)
    assert isinstance(engine, OpenAIEngine)
    assert engine.api_key == "sk-FAKE-openai-0000"


def test_environment_key_is_used_only_when_no_key_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", ENV_KEY)
    runner_default = GeminiEngine()  # Runner(api_key=None) builds this
    assert runner_default.api_key == ENV_KEY
    assert resolve_engine(_agent(), default_engine=runner_default) is runner_default
    with_key = resolve_engine(_agent(api_key=AGENT_KEY), default_engine=runner_default)
    assert isinstance(with_key, GeminiEngine) and with_key.api_key == AGENT_KEY


# --- end to end through Runner: what is actually sent -------------------------------------


async def test_runner_sends_the_agent_key_to_gemini(recorder: _Recorder) -> None:
    runner = Runner(starting_agent=_agent(api_key=AGENT_KEY), api_key=RUNNER_KEY)
    response = await runner.run_async("s", "hi")
    assert response.content == "ok"
    assert recorder.keys() == [AGENT_KEY]
    assert {host for host, _, _ in recorder.requests} == {GEMINI_HOST}


async def test_pooled_runner_scope_sends_the_agent_key(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder()
    monkeypatch.setattr(
        gemini_module,
        "_new_pooled_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(rec.handler)),
    )
    monkeypatch.setattr(
        gemini_module,
        "_new_attempt_client",
        lambda timeout: pytest.fail("the pooled client of the runner scope must be used"),
    )
    runner = Runner(starting_agent=_agent(api_key=AGENT_KEY), api_key=RUNNER_KEY)
    async with runner:
        assert (await runner.run_async("s", "hi")).content == "ok"
    assert rec.keys() == [AGENT_KEY]


async def test_runner_key_and_environment_key_still_apply_without_an_agent_key(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    await Runner(starting_agent=_agent(), api_key=RUNNER_KEY).run_async("s1", "hi")
    monkeypatch.setenv("GEMINI_API_KEY", ENV_KEY)
    await Runner(starting_agent=_agent()).run_async("s2", "hi")
    await Runner(starting_agent=_agent(api_key=AGENT_KEY)).run_async("s3", "hi")
    assert recorder.keys() == [RUNNER_KEY, ENV_KEY, AGENT_KEY]


async def test_each_agent_of_a_swarm_sends_only_its_own_key(recorder: _Recorder) -> None:
    specialist = _agent(name="Specialist", api_key=AGENT_KEY)
    triage = _agent(name="Triage", handoffs=[specialist])
    handoff_tool = next(t.name for t in triage.get_all_tools())
    recorder.replies = [
        {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": handoff_tool, "args": {}}}
        ]}}]},
        _text("specialist reply"),
    ]
    runner = Runner(starting_agent=triage, api_key=RUNNER_KEY)
    response = await runner.run_async("s", "hi")
    assert response.content == "specialist reply"
    assert response.active_agent_name == "Specialist"
    # The triage turn uses the runner key; after the handoff, the specialist uses its own.
    assert recorder.keys() == [RUNNER_KEY, AGENT_KEY]
    assert {host for host, _, _ in recorder.requests} == {GEMINI_HOST}


async def test_agent_key_is_not_exposed_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(status=403)
    monkeypatch.setattr(
        gemini_module,
        "_new_attempt_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(rec.handler), timeout=timeout),
    )
    runner = Runner(starting_agent=_agent(api_key=AGENT_KEY), api_key=RUNNER_KEY)
    with pytest.raises(ProviderError) as info:
        await runner.run_async("s", "hi")
    assert rec.keys() == [AGENT_KEY]
    assert AGENT_KEY not in str(info.value) and AGENT_KEY not in repr(info.value)
