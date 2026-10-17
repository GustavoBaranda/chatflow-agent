"""Engine factory and dynamic resolver for multi-provider agent swarms."""

from typing import TYPE_CHECKING, cast

from chatflow_agent.core.engines.anthropic import AnthropicEngine
from chatflow_agent.core.engines.base import BaseEngine
from chatflow_agent.core.engines.gemini import GeminiEngine
from chatflow_agent.core.engines.openai_compatible import OpenAIEngine

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent

_OPENAI_COMPATIBLE_PREFIXES = (
    "gpt-", "o1", "o3", "chatgpt-", "text-embedding-", "dall-e"
)
_CLAUDE_PREFIXES = ("claude-",)
_OLLAMA_PREFIXES = ("gemma", "llama", "mistral", "qwen", "phi")
_DEEPSEEK_PREFIXES = ("deepseek-",)


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
    3. agent.base_url or agent.api_key if custom connection options are set on agent.
    4. default_engine when it was explicitly supplied by the caller
       (``prefer_default_engine=True``). A deliberate caller choice must win over
       the model-name heuristics below.
    5. Heuristic matching on the agent.model name prefix when it names a concrete
       non-Gemini provider. These take precedence over a generic fallback
       ``default_engine`` so that a model such as ``claude-3-5-sonnet`` or
       ``deepseek-chat`` is not silently routed to the default (Gemini) engine.
    6. default_engine if provided (e.g. from Runner or test mock). Applies to
       Gemini models and to models without a recognizable provider prefix.
    7. Default GeminiEngine.
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

    # 3. Agent has custom base_url or api_key without provider
    if base_url:
        return OpenAIEngine(api_key=api_key, base_url=base_url, provider="openai")

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
    #    engine Runner builds from its ``api_key``). Respect it as-is.
    if default_engine is not None:
        return default_engine

    # 7. Ultimate fallback: a Gemini engine.
    return GeminiEngine(api_key=api_key)
