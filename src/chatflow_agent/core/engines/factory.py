"""Engine factory and dynamic resolver for multi-provider agent swarms."""

from typing import TYPE_CHECKING, cast

from chatflow_agent.core.engines.anthropic import AnthropicEngine
from chatflow_agent.core.engines.base import BaseEngine
from chatflow_agent.core.engines.gemini import GeminiEngine, _clean_key
from chatflow_agent.core.engines.openai_compatible import (
    PROVIDER_PRESETS,
    OpenAIEngine,
    _is_official_endpoint,
)

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent

_OPENAI_COMPATIBLE_PREFIXES = (
    "gpt-", "o1", "o3", "chatgpt-", "text-embedding-", "dall-e"
)
_CLAUDE_PREFIXES = ("claude-",)
_OLLAMA_PREFIXES = ("gemma", "llama", "mistral", "qwen", "phi")
_DEEPSEEK_PREFIXES = ("deepseek-",)


def _provider_for_official_base_url(base_url: str) -> str:
    """Return the preset whose official endpoint is exactly base_url, else "openai".

    The provider decides which environment key OpenAIEngine may read, so a
    base_url only selects a provider when it is that provider's official endpoint.
    Any other URL keeps "openai", whose engine then requires an explicit api_key
    (or applies the local-endpoint rule).
    """
    for name, preset in PROVIDER_PRESETS.items():
        if _is_official_endpoint(base_url, preset["base_url"]):
            return name
    return "openai"


def _gemini_engine_with_key(engine: GeminiEngine, api_key: str) -> GeminiEngine:
    """A GeminiEngine like ``engine`` that sends ``api_key``; every other setting is kept."""
    return GeminiEngine(
        api_key=api_key,
        timeout_seconds=engine.timeout_seconds,
        max_retries=engine.max_retries,
        base_delay=engine.base_delay,
        max_delay=engine.max_delay,
        http_client=engine._external_client,
        api_key_in_query=engine.api_key_in_query,
    )


def resolve_engine(
    agent: "Agent",
    default_engine: BaseEngine | None = None,
    *,
    prefer_default_engine: bool = False,
) -> BaseEngine:
    """Automatically resolve the appropriate LLM engine for an agent.

    Resolution order:
    1. agent.engine if explicitly attached.
    2. agent.provider if explicitly set (gemini, openai, grok, xai, ollama, gemma, anthropic, claude, deepseek).
    3. agent.base_url if a custom connection endpoint is set on agent.
    4. default_engine when it was explicitly supplied by the caller
       (``prefer_default_engine=True``). A deliberate caller choice must win over
       the model-name heuristics below.
    5. Heuristic matching on the agent.model name prefix when it names a concrete
       non-Gemini provider. These take precedence over a generic fallback
       ``default_engine`` so that a model such as ``claude-3-5-sonnet`` or
       ``deepseek-chat`` is not silently routed to the default (Gemini) engine.
    6. default_engine if provided (e.g. from Runner or test mock). Applies to
       Gemini models and to models without a recognizable provider prefix. If
       agent.api_key is set and default_engine is a plain GeminiEngine, a copy of
       it that uses agent.api_key is returned instead.
    7. Default GeminiEngine.

    agent.api_key never selects a provider by itself: it is the key of the engine
    that steps 2, 3, 5 or 7 select, and at step 6 only when default_engine is a plain
    GeminiEngine (any other default_engine is returned unchanged). It does not
    replace an engine attached to the agent (step 1) or supplied explicitly by the
    caller (step 4).
    """
    # 1. Custom engine explicitly on agent
    if hasattr(agent, "engine") and agent.engine is not None:
        return cast(BaseEngine, agent.engine)

    provider = (getattr(agent, "provider", None) or "").lower()
    api_key = getattr(agent, "api_key", None)
    base_url = getattr(agent, "base_url", None)
    model = (getattr(agent, "model", None) or "").lower()

    # 2. Provider explicit match
    if provider in ("gemini", "google"):
        return GeminiEngine(api_key=api_key)

    if provider in ("anthropic", "claude"):
        return AnthropicEngine(api_key=api_key, base_url=base_url or "https://api.anthropic.com/v1")

    if provider in ("openai", "grok", "xai", "ollama", "gemma", "deepseek", "groq"):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider=provider)

    # 3. Agent has custom base_url without provider. An official
    #    endpoint (e.g. DeepSeek's) selects its own provider and environment key;
    #    OPENAI_API_KEY is never assumed for another host.
    if base_url:
        return OpenAIEngine(
            api_key=api_key,
            base_url=base_url,
            provider=_provider_for_official_base_url(base_url),
        )

    # 4. An explicitly supplied engine (e.g. ``Runner(engine=...)``) is a
    #    deliberate caller choice, so it wins over the model-prefix heuristics.
    if prefer_default_engine and default_engine is not None:
        return default_engine

    # 5. Model prefix heuristics for a concrete non-Gemini provider. A model name
    #    that identifies a specific provider is more precise than a generic
    #    fallback engine, so it must win over ``default_engine`` below.
    if any(model.startswith(p) for p in _CLAUDE_PREFIXES):
        return AnthropicEngine(api_key=api_key, base_url=base_url or "https://api.anthropic.com/v1")

    if any(model.startswith(p) for p in _OPENAI_COMPATIBLE_PREFIXES):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="openai")

    if model.startswith("grok"):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="grok")

    if any(model.startswith(p) for p in _DEEPSEEK_PREFIXES):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="deepseek")

    if any(model.startswith(p) for p in _OLLAMA_PREFIXES):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="ollama")

    # 6. Fallback engine for Gemini models and unrecognized models (e.g. the
    #    engine Runner builds from its ``api_key``). Respect it as-is, except that
    #    an api_key set on the agent replaces the key of a plain GeminiEngine: the
    #    provider stays Gemini. Subclasses (custom engines, mocks) are kept unchanged.
    if default_engine is not None:
        agent_key = _clean_key(api_key)
        if (
            agent_key is not None
            and type(default_engine) is GeminiEngine
            and agent_key != default_engine.api_key
        ):
            return _gemini_engine_with_key(default_engine, agent_key)
        return default_engine

    # 7. Ultimate fallback: a Gemini engine.
    return GeminiEngine(api_key=api_key)
