"""Engine factory and dynamic resolver for multi-provider agent swarms."""

from typing import TYPE_CHECKING, cast

from chatflow_agent.core.engines.anthropic import AnthropicEngine
from chatflow_agent.core.engines.base import BaseEngine
from chatflow_agent.core.engines.gemini import GeminiEngine
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
_GEMINI_PREFIXES = ("gemini-",)
_OLLAMA_PREFIXES = ("gemma", "llama", "mistral", "qwen", "phi")


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


def resolve_engine(agent: "Agent", default_engine: BaseEngine | None = None) -> BaseEngine:
    """Automatically resolve the appropriate LLM engine for an agent.

    Resolution order:
    1. agent.engine if explicitly attached.
    2. agent.provider if explicitly set ('gemini', 'openai', 'grok', 'xai', 'ollama', 'gemma', 'anthropic', 'claude').
    3. agent.base_url or agent.api_key if custom connection options are set on agent.
    4. default_engine if provided (e.g. from Runner or test mock).
    5. Heuristic matching on agent.model name prefix.
    6. Default GeminiEngine.
    """
    # 1. Custom engine explicitly on agent
    if hasattr(agent, "engine") and agent.engine is not None:
        return cast(BaseEngine, agent.engine)

    provider = (getattr(agent, "provider", None) or "").lower()
    api_key = getattr(agent, "api_key", None)
    base_url = getattr(agent, "base_url", None)

    # 2. Provider explicit match
    if provider in ("gemini", "google"):
        return GeminiEngine(api_key=api_key)

    if provider in ("anthropic", "claude"):
        return AnthropicEngine(api_key=api_key, base_url=base_url or "https://api.anthropic.com/v1")

    if provider in ("openai", "grok", "xai", "ollama", "gemma", "deepseek", "groq"):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider=provider)

    # 3. Agent has custom base_url or api_key without provider. An official
    #    endpoint (e.g. DeepSeek's) selects its own provider and environment key;
    #    OPENAI_API_KEY is never assumed for another host.
    if base_url:
        return OpenAIEngine(
            api_key=api_key,
            base_url=base_url,
            provider=_provider_for_official_base_url(base_url),
        )

    # 4. If Runner provided a default_engine (e.g. MockEngine or pre-configured engine), respect it!
    if default_engine is not None:
        return default_engine

    # 5. Model prefix heuristics when no default_engine was given
    model = (agent.model or "").lower()
    if any(model.startswith(p) for p in _GEMINI_PREFIXES):
        return GeminiEngine(api_key=api_key)

    if any(model.startswith(p) for p in _CLAUDE_PREFIXES):
        return AnthropicEngine(api_key=api_key, base_url=base_url or "https://api.anthropic.com/v1")

    if any(model.startswith(p) for p in _OPENAI_COMPATIBLE_PREFIXES):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="openai")

    if model.startswith("grok"):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="grok")

    if any(model.startswith(p) for p in _OLLAMA_PREFIXES):
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="ollama")

    return GeminiEngine(api_key=api_key)
