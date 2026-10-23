"""Credential scoping for OpenAIEngine and AnthropicEngine.

An API key read from the environment may only be sent to the official endpoint of
the provider that issued it. All keys below are fictitious and every request is
served by ``httpx.MockTransport``; nothing leaves the process.
"""

import logging
import os
from collections.abc import Callable

import httpx
import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import (
    AnthropicEngine,
    OpenAIEngine,
    resolve_engine,
)
from chatflow_agent.core.engines.base import BaseEngine
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role

FAKE_KEYS = {
    "OPENAI_API_KEY": "sk-FAKE-openai-0000",
    "XAI_API_KEY": "sk-FAKE-xai-0000",
    "DEEPSEEK_API_KEY": "sk-FAKE-deepseek-0000",
    "GROQ_API_KEY": "sk-FAKE-groq-0000",
    "ANTHROPIC_API_KEY": "sk-FAKE-anthropic-0000",
    "OLLAMA_API_KEY": "sk-FAKE-ollama-0000",
}
EXPLICIT_KEY = "sk-FAKE-explicit-proxy-0000"
PROXY_URL = "https://proxy.example.com/v1"

# Both engines accept this body: OpenAI reads "choices", Anthropic reads "content".
_OK_BODY: dict[str, list[object]] = {"choices": [], "content": []}


@pytest.fixture(autouse=True)
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every real *_API_KEY / *_BASE_URL with fictitious keys."""
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_BASE_URL")):
            monkeypatch.delenv(name)
    for name, value in FAKE_KEYS.items():
        monkeypatch.setenv(name, value)


def _agent(**kwargs: str) -> Agent:
    kwargs.setdefault("model", "gpt-4o-mini")
    return Agent(name="Credentials", instructions="Reply.", **kwargs)  # type: ignore[arg-type]


def _install(
    engine: BaseEngine,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    follow_redirects: bool = False,
) -> None:
    engine._external_client = httpx.AsyncClient(  # type: ignore[attr-defined]
        transport=httpx.MockTransport(handler), follow_redirects=follow_redirects
    )


async def _send_one(engine: BaseEngine, agent: Agent) -> httpx.Request:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_OK_BODY)

    _install(engine, handler)
    await engine.generate_turn_async(agent, [Message(role=Role.USER, content="hi")])
    assert len(seen) == 1
    return seen[0]


def _credential(request: httpx.Request) -> str | None:
    if "x-api-key" in request.headers:
        return request.headers["x-api-key"]
    auth = request.headers.get("authorization")
    return auth.removeprefix("Bearer ") if auth else None


# --- Configurations that send a key: it must be the right key for the right host.

@pytest.mark.parametrize(
    ("agent_kwargs", "expected_engine", "expected_host", "expected_key"),
    [
        ({"provider": "openai"}, OpenAIEngine, "api.openai.com", FAKE_KEYS["OPENAI_API_KEY"]),
        ({"provider": "deepseek"}, OpenAIEngine, "api.deepseek.com", FAKE_KEYS["DEEPSEEK_API_KEY"]),
        (
            {"model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"},
            OpenAIEngine, "api.deepseek.com", FAKE_KEYS["DEEPSEEK_API_KEY"],
        ),
        (
            {"model": "grok-2", "base_url": "https://api.x.ai/v1"},
            OpenAIEngine, "api.x.ai", FAKE_KEYS["XAI_API_KEY"],
        ),
        (
            {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1/"},
            OpenAIEngine, "api.groq.com", FAKE_KEYS["GROQ_API_KEY"],
        ),
        (
            {"model": "gpt-4o", "base_url": "https://API.OPENAI.COM:443/v1/"},
            OpenAIEngine, "api.openai.com", FAKE_KEYS["OPENAI_API_KEY"],
        ),
        ({"model": "claude-3-5-sonnet"}, AnthropicEngine, "api.anthropic.com", FAKE_KEYS["ANTHROPIC_API_KEY"]),
        (
            {"model": "claude-3-5-sonnet", "provider": "anthropic", "base_url": "https://api.anthropic.com/v1/"},
            AnthropicEngine, "api.anthropic.com", FAKE_KEYS["ANTHROPIC_API_KEY"],
        ),
        (
            {"model": "gemma2:9b", "provider": "ollama", "base_url": "http://localhost:11434/v1"},
            OpenAIEngine, "localhost", FAKE_KEYS["OLLAMA_API_KEY"],
        ),
        # Explicit keys keep working with any endpoint (legitimate proxies and gateways).
        ({"base_url": PROXY_URL, "api_key": EXPLICIT_KEY}, OpenAIEngine, "proxy.example.com", EXPLICIT_KEY),
        (
            {"provider": "openai", "base_url": "https://api.groq.com/openai/v1", "api_key": EXPLICIT_KEY},
            OpenAIEngine, "api.groq.com", EXPLICIT_KEY,
        ),
        (
            {"model": "claude-3-5-sonnet", "provider": "anthropic", "base_url": PROXY_URL, "api_key": EXPLICIT_KEY},
            AnthropicEngine, "proxy.example.com", EXPLICIT_KEY,
        ),
    ],
)
async def test_environment_key_reaches_only_its_own_provider(
    agent_kwargs: dict[str, str],
    expected_engine: type[BaseEngine],
    expected_host: str,
    expected_key: str,
) -> None:
    agent = _agent(**agent_kwargs)
    engine = resolve_engine(agent)
    assert isinstance(engine, expected_engine)

    request = await _send_one(engine, agent)

    assert request.url.host == expected_host
    assert _credential(request) == expected_key


# --- Configurations that would send a key to a host that did not issue it.

@pytest.mark.parametrize(
    "agent_kwargs",
    [
        {"provider": "openai", "base_url": "https://api.groq.com/openai/v1"},
        {"provider": "grok", "base_url": "https://api.openai.com/v1"},
        {"provider": "deepseek", "base_url": "https://api.x.ai/v1"},
        {"provider": "openai", "base_url": PROXY_URL},
        {"provider": "anthropic", "base_url": PROXY_URL},
        {"provider": "claude", "base_url": "http://10.0.0.5:8080"},
        {"provider": "anthropic", "base_url": "http://localhost:4000/v1"},
        {"provider": "anthropic", "base_url": "https://api.openai.com/v1"},
        {"model": "claude-3-5-sonnet", "base_url": PROXY_URL},
        {"model": "gpt-4o", "base_url": "https://api.anthropic.com/v1"},
    ],
)
def test_environment_key_is_not_sent_to_other_endpoints(agent_kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="A custom or unofficial base_url was provided") as exc_info:
        resolve_engine(_agent(**agent_kwargs))
    _assert_no_fake_key(str(exc_info.value))


@pytest.mark.parametrize(
    ("official", "build"),
    [
        ("https://api.openai.com/v1", lambda url: OpenAIEngine(provider="openai", base_url=url)),
        ("https://api.x.ai/v1", lambda url: OpenAIEngine(provider="grok", base_url=url)),
        ("https://api.anthropic.com/v1", lambda url: AnthropicEngine(base_url=url)),
    ],
)
@pytest.mark.parametrize(
    "mutate",
    [
        lambda u: u.replace("https://", "http://"),
        lambda u: u.replace("/v1", ".evil.example/v1"),
        lambda u: u.replace("https://", "https://user:hunter2@"),
        lambda u: u.replace("/v1", "@evil.example/v1"),
        lambda u: u.replace("/v1", ":8443/v1"),
        lambda u: u + "/extra",
        lambda u: u.replace("/v1", "/v2"),
        lambda u: u + "?redirect=evil.example",
        lambda u: u + "#fragment",
        lambda u: "https://evil.example/" + u,
    ],
)
def test_lookalike_official_urls_never_get_environment_key(
    official: str,
    build: Callable[[str], BaseEngine],
    mutate: Callable[[str], str],
) -> None:
    with pytest.raises(ValueError, match="A custom or unofficial base_url was provided") as exc_info:
        build(mutate(official))
    message = str(exc_info.value)
    _assert_no_fake_key(message)
    assert "hunter2" not in message


def test_resolve_engine_does_not_infer_provider_from_lookalike_host() -> None:
    with pytest.raises(ValueError, match="A custom or unofficial base_url was provided"):
        resolve_engine(_agent(model="deepseek-chat", base_url="https://api.deepseek.com.evil.example/v1"))


# --- *_BASE_URL environment variables follow the same rule.

@pytest.mark.parametrize(
    ("env_var", "url", "provider"),
    [
        ("OPENAI_BASE_URL", "https://api.x.ai/v1", "openai"),
        ("OPENAI_BASE_URL", PROXY_URL, "openai"),
        ("GROK_BASE_URL", "https://api.openai.com/v1", "grok"),
        ("XAI_BASE_URL", "https://api.openai.com/v1", "xai"),
        ("DEEPSEEK_BASE_URL", "https://api.groq.com/openai/v1", "deepseek"),
    ],
)
def test_base_url_env_var_does_not_unlock_another_providers_key(
    monkeypatch: pytest.MonkeyPatch, env_var: str, url: str, provider: str
) -> None:
    monkeypatch.setenv(env_var, url)
    with pytest.raises(ValueError, match="A custom or unofficial base_url was provided"):
        OpenAIEngine(provider=provider)


async def test_base_url_env_var_pointing_to_own_official_endpoint_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/")
    engine = OpenAIEngine(provider="deepseek")
    request = await _send_one(engine, _agent(model="deepseek-chat"))
    assert request.url.host == "api.deepseek.com"
    assert _credential(request) == FAKE_KEYS["DEEPSEEK_API_KEY"]


# --- Explicit empty key: no credential at all, even with keys in the environment.

@pytest.mark.parametrize(
    "build",
    [
        lambda: OpenAIEngine(base_url=PROXY_URL, api_key=""),
        lambda: AnthropicEngine(base_url=PROXY_URL, api_key=""),
        lambda: AnthropicEngine(api_key=""),
        lambda: resolve_engine(_agent(base_url=PROXY_URL, api_key="")),
    ],
)
async def test_explicit_empty_api_key_sends_no_credential(build: Callable[[], BaseEngine]) -> None:
    engine = build()
    request = await _send_one(engine, _agent(model="claude-3-5-sonnet"))
    assert "authorization" not in request.headers
    assert "x-api-key" not in request.headers


@pytest.mark.parametrize("env_has_key", [True, False], ids=["env-key", "no-env-key"])
@pytest.mark.parametrize(
    ("build", "env_var", "header"),
    [
        (lambda key: OpenAIEngine(provider="openai", api_key=key), "OPENAI_API_KEY", "authorization"),
        (lambda key: AnthropicEngine(api_key=key), "ANTHROPIC_API_KEY", "x-api-key"),
    ],
    ids=["openai", "anthropic"],
)
async def test_none_reads_environment_but_empty_string_never_does(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[str | None], BaseEngine],
    env_var: str,
    header: str,
    env_has_key: bool,
) -> None:
    """On the official endpoint, None falls back to the env key; "" never does and sends no header."""
    if not env_has_key:
        monkeypatch.delenv(env_var)

    request = await _send_one(build(""), _agent())
    assert header not in request.headers

    engine = build(None)
    if env_has_key:
        request = await _send_one(engine, _agent())
        assert _credential(request) == FAKE_KEYS[env_var]
    elif isinstance(engine, AnthropicEngine):
        with pytest.raises(ProviderError, match="not configured"):
            await engine.generate_turn_async(_agent(), [Message(role=Role.USER, content="hi")])
    else:
        request = await _send_one(engine, _agent())
        assert header not in request.headers


async def test_anthropic_without_any_key_still_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    engine = AnthropicEngine()
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        await engine.generate_turn_async(_agent(), [Message(role=Role.USER, content="hi")])


# --- Redirects: never followed, so no credential is replayed to another origin.

@pytest.mark.parametrize(
    "location",
    ["https://evil.example/collect", "/v1/moved", "https://{host}/v1/moved"],
    ids=["cross-origin", "same-origin-relative", "same-origin-absolute"],
)
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "build",
    [
        lambda: OpenAIEngine(provider="openai"),
        lambda: AnthropicEngine(),
        lambda: OpenAIEngine(base_url=PROXY_URL, api_key=EXPLICIT_KEY),
        lambda: AnthropicEngine(base_url=PROXY_URL, api_key=EXPLICIT_KEY),
    ],
)
async def test_redirects_are_not_followed_even_by_injected_clients(
    build: Callable[[], BaseEngine], status: int, location: str
) -> None:
    engine = build()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "evil.example" or request.url.path.endswith("/moved"):
            return httpx.Response(200, json=_OK_BODY)
        return httpx.Response(
            status, headers={"Location": location.format(host=request.url.host)}
        )

    _install(engine, handler, follow_redirects=True)

    with pytest.raises(ProviderError, match="redirected") as exc_info:
        await engine.generate_turn_async(_agent(), [Message(role=Role.USER, content="hi")])

    assert [r.url.host for r in seen] == [httpx.URL(engine.base_url).host]  # type: ignore[attr-defined]
    _assert_no_fake_key(str(exc_info.value))


# --- Errors and logs never contain a key.

def _assert_no_fake_key(text: str) -> None:
    for secret in (*FAKE_KEYS.values(), EXPLICIT_KEY):
        assert secret not in text


@pytest.mark.parametrize(
    "build",
    [lambda: OpenAIEngine(provider="openai"), lambda: AnthropicEngine()],
)
@pytest.mark.parametrize(
    "respond",
    [
        lambda request: httpx.Response(401, json={"error": "invalid key"}),
        lambda request: httpx.Response(503, json={"error": "unavailable"}),
        lambda request: httpx.Response(307, headers={"Location": "https://evil.example/"}),
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request)),
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timed out", request=request)),
    ],
)
async def test_errors_and_logs_never_contain_api_keys(
    caplog: pytest.LogCaptureFixture,
    build: Callable[[], BaseEngine],
    respond: Callable[[httpx.Request], httpx.Response],
) -> None:
    engine = build()
    engine.base_delay = 0.0  # type: ignore[attr-defined]
    engine.max_delay = 0.0  # type: ignore[attr-defined]
    _install(engine, respond)

    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderError) as exc_info:
        await engine.generate_turn_async(_agent(), [Message(role=Role.USER, content="hi")])

    chain: list[BaseException] = []
    err: BaseException | None = exc_info.value
    while err is not None:
        chain.append(err)
        err = err.__cause__ or err.__context__
    _assert_no_fake_key(caplog.text)
    for item in chain:
        _assert_no_fake_key(str(item))
        _assert_no_fake_key(repr(item))
    _assert_no_fake_key(repr(engine))
