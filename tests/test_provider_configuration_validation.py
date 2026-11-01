"""Tests for provider and endpoint configuration validation (Issue #166).

Verifies that invalid or unsupported provider, engine, model, or endpoint configurations
never silently fall back to another provider or endpoint, and that configuration errors
surface early through ConfigurationError (subclass of ChatFlowError and ValueError).
"""

import pytest

from chatflow_agent import ChatFlowError, ConfigurationError
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import (
    AnthropicEngine,
    BaseEngine,
    GeminiEngine,
    OpenAIEngine,
    resolve_engine,
)
from chatflow_agent.core.engines.factory import _ALL_SUPPORTED_PROVIDERS
from chatflow_agent.core.runner import Runner


class DummyCustomEngine(BaseEngine):
    """Minimal engine mock for validation testing."""

    async def generate_turn_async(self, agent: Agent, history: list) -> object:
        raise NotImplementedError


# ============================================================================
# 1. Public Exception Hierarchy & Contracts
# ============================================================================


def test_configuration_error_hierarchy() -> None:
    """ConfigurationError must inherit from both ChatFlowError and ValueError."""
    assert issubclass(ConfigurationError, ChatFlowError)
    assert issubclass(ConfigurationError, ValueError)

    err = ConfigurationError("Test error message")
    assert isinstance(err, ChatFlowError)
    assert isinstance(err, ValueError)
    assert str(err) == "Test error message"


# ============================================================================
# 2. Failure Mode 1: Unsupported Explicit Provider (e.g. mistral)
# ============================================================================


def test_explicit_unsupported_provider_rejected_at_agent_init() -> None:
    """Agent(provider="mistral", model="mistral-large") must raise ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(name="MistralBot", instructions="Reply.", provider="mistral", model="mistral-large")

    msg = str(exc_info.value)
    assert "Unsupported provider 'mistral'" in msg
    assert "Supported providers are:" in msg
    assert "openai" in msg
    assert "gemini" in msg
    assert "anthropic" in msg


def test_explicit_unsupported_provider_rejected_at_resolve_engine() -> None:
    """resolve_engine must reject an agent with an unsupported provider without falling through."""
    agent = Agent(name="Bot", instructions="Reply.", model="mistral-large")
    # Mutating after construction to simulate dynamic attribute assignment
    agent.provider = "mistral"

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_engine(agent)

    msg = str(exc_info.value)
    assert "Unsupported provider 'mistral'" in msg
    # Verify it did not fall through to Ollama via model-prefix "mistral"
    assert "ollama" not in msg.lower() or "supported providers are" in msg.lower()


# ============================================================================
# 3. Failure Mode 2: Gemini / Google Provider with custom base_url
# ============================================================================


@pytest.mark.parametrize("gemini_provider", ["gemini", "google", "Gemini", "GOOGLE"])
def test_gemini_provider_with_custom_base_url_rejected_at_agent_init(gemini_provider: str) -> None:
    """Agent with provider='gemini' or 'google' and base_url must raise ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(
            name="GeminiBot",
            instructions="Reply.",
            provider=gemini_provider,
            base_url="https://proxy.example.com/v1",
        )

    msg = str(exc_info.value)
    assert f"Provider '{gemini_provider}' does not support a custom 'base_url'" in msg
    assert "GOOGLE_GEMINI_BASE_URL" in msg


@pytest.mark.parametrize("gemini_provider", ["gemini", "google"])
def test_gemini_provider_with_custom_base_url_rejected_at_resolve_engine(gemini_provider: str) -> None:
    """resolve_engine must reject gemini/google with base_url without silently ignoring it."""
    agent = Agent(name="GeminiBot", instructions="Reply.", base_url="https://proxy.example.com/v1")
    agent.provider = gemini_provider

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_engine(agent)

    msg = str(exc_info.value)
    assert f"Provider '{gemini_provider}' does not support a custom 'base_url'" in msg
    assert "GOOGLE_GEMINI_BASE_URL" in msg


# ============================================================================
# 4. Failure Mode 3: Unrecognized Model Prefix Fallback (Preserved & Documented)
# ============================================================================


def test_unrecognized_model_without_provider_falls_back_to_gemini() -> None:
    """model='o4-mini' without provider falls back to default GeminiEngine (intentional fallback)."""
    agent = Agent(name="O4Bot", instructions="Reply.", model="o4-mini")
    engine = resolve_engine(agent)
    assert isinstance(engine, GeminiEngine)


def test_unrecognized_model_with_explicit_provider_routes_correctly() -> None:
    """When provider='openai' is explicitly set, model='o4-mini' routes to OpenAIEngine."""
    agent = Agent(name="O4Bot", instructions="Reply.", provider="openai", model="o4-mini")
    engine = resolve_engine(agent)
    assert isinstance(engine, OpenAIEngine)
    assert engine.provider == "openai"


# ============================================================================
# 5. Failure Mode 4: Custom base_url without api_key & Early Detection
# ============================================================================


def test_custom_remote_base_url_without_key_raises_configuration_error() -> None:
    """Custom remote base_url without api_key raises ConfigurationError from resolve_engine."""
    agent = Agent(name="CustomBot", instructions="Reply.", base_url="https://llm.example.com/v1")
    with pytest.raises(ConfigurationError) as exc_info:
        resolve_engine(agent)

    msg = str(exc_info.value)
    assert "A custom or unofficial base_url was provided" in msg
    assert "https://llm.example.com/v1" in msg
    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, ChatFlowError)


def test_custom_remote_base_url_without_key_fails_fast_at_runner_init() -> None:
    """Runner fails fast at initialization when an agent has unresolvable engine configuration."""
    agent = Agent(name="CustomBot", instructions="Reply.", base_url="https://llm.example.com/v1")
    with pytest.raises(ConfigurationError) as exc_info:
        Runner(starting_agent=agent)

    assert "A custom or unofficial base_url was provided" in str(exc_info.value)


def test_transitive_handoff_with_unresolvable_engine_fails_fast_at_runner_init() -> None:
    """Runner validates all transitive agents in the handoff tree at construction time."""
    leaf = Agent(name="Leaf", instructions="Reply.", base_url="https://llm.example.com/v1")
    root = Agent(name="Root", instructions="Reply.", handoffs=[leaf])

    with pytest.raises(ConfigurationError) as exc_info:
        Runner(starting_agent=root)

    assert "A custom or unofficial base_url was provided" in str(exc_info.value)


def test_runner_register_agent_validates_configuration_early() -> None:
    """Runner.register_agent validates that newly registered agents can resolve their engine."""
    root = Agent(name="Root", instructions="Reply.")
    runner = Runner(starting_agent=root)

    invalid_agent = Agent(name="Invalid", instructions="Reply.", base_url="https://llm.example.com/v1")
    with pytest.raises(ConfigurationError) as exc_info:
        runner.register_agent(invalid_agent)

    assert "A custom or unofficial base_url was provided" in str(exc_info.value)


# ============================================================================
# 6. OpenAIEngine Constructor Direct Validation
# ============================================================================


def test_openai_engine_unknown_provider_without_base_url_raises_configuration_error() -> None:
    """OpenAIEngine(provider='unknown') must not silently fall back to openai endpoint."""
    with pytest.raises(ConfigurationError) as exc_info:
        OpenAIEngine(provider="unknown_provider")

    msg = str(exc_info.value)
    assert "Unsupported OpenAI-compatible provider 'unknown_provider'" in msg
    assert "Supported preset providers:" in msg


def test_openai_engine_custom_provider_with_explicit_base_url_succeeds() -> None:
    """OpenAIEngine with custom provider and explicit base_url is valid."""
    engine = OpenAIEngine(provider="vllm", base_url="http://localhost:8000/v1")
    assert engine.provider == "vllm"
    assert engine.base_url == "http://localhost:8000/v1"


def test_openai_engine_custom_remote_base_url_with_explicit_key_succeeds() -> None:
    """OpenAIEngine with custom remote base_url and explicit api_key succeeds."""
    engine = OpenAIEngine(
        provider="custom_llm",
        base_url="https://custom.remote.com/v1",
        api_key="sk-explicit-1234",
    )
    assert engine.provider == "custom_llm"
    assert engine.base_url == "https://custom.remote.com/v1"
    assert engine.api_key == "sk-explicit-1234"


# ============================================================================
# 7. Agent and Runner Constructor Type Validation
# ============================================================================


def test_agent_invalid_engine_type_raises_configuration_error() -> None:
    """Agent(engine=...) must be an instance of BaseEngine."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(name="Bot", instructions="Reply.", engine="not_an_engine")

    assert "Agent 'engine' must be an instance of BaseEngine or None, got str." in str(exc_info.value)


def test_agent_valid_engine_instance_accepted() -> None:
    """Agent accepts an instance of BaseEngine."""
    engine = DummyCustomEngine()
    agent = Agent(name="Bot", instructions="Reply.", engine=engine)
    assert agent.engine is engine
    assert resolve_engine(agent) is engine


def test_agent_invalid_provider_type_raises_configuration_error() -> None:
    """Agent(provider=123) raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(name="Bot", instructions="Reply.", provider=123)  # type: ignore[arg-type]

    assert "Agent 'provider' must be a string or None, got int." in str(exc_info.value)


def test_agent_empty_provider_raises_configuration_error() -> None:
    """Agent(provider='   ') raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(name="Bot", instructions="Reply.", provider="   ")

    assert "Agent 'provider' cannot be an empty or whitespace-only string." in str(exc_info.value)


def test_agent_invalid_base_url_type_raises_configuration_error() -> None:
    """Agent(base_url=123) raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(name="Bot", instructions="Reply.", base_url=123)  # type: ignore[arg-type]

    assert "Agent 'base_url' must be a string or None, got int." in str(exc_info.value)


def test_agent_empty_base_url_raises_configuration_error() -> None:
    """Agent(base_url='   ') raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        Agent(name="Bot", instructions="Reply.", base_url="   ")

    assert "Agent 'base_url' cannot be an empty or whitespace-only string." in str(exc_info.value)


def test_runner_invalid_engine_type_raises_configuration_error() -> None:
    """Runner(engine=...) must be an instance of BaseEngine."""
    agent = Agent(name="Bot", instructions="Reply.")
    with pytest.raises(ConfigurationError) as exc_info:
        Runner(starting_agent=agent, engine="not_an_engine")  # type: ignore[arg-type]

    assert "Runner 'engine' must be an instance of BaseEngine or None, got str." in str(exc_info.value)


# ============================================================================
# 8. All Supported Providers Resolution Tests
# ============================================================================


@pytest.mark.parametrize(
    ("provider_name", "expected_engine_type"),
    [
        ("gemini", GeminiEngine),
        ("google", GeminiEngine),
        ("anthropic", AnthropicEngine),
        ("claude", AnthropicEngine),
        ("openai", OpenAIEngine),
        ("grok", OpenAIEngine),
        ("xai", OpenAIEngine),
        ("ollama", OpenAIEngine),
        ("gemma", OpenAIEngine),
        ("deepseek", OpenAIEngine),
        ("groq", OpenAIEngine),
    ],
)
def test_all_supported_providers_resolve_expected_engine(
    provider_name: str, expected_engine_type: type[BaseEngine]
) -> None:
    """Every provider in _ALL_SUPPORTED_PROVIDERS resolves to its designated engine."""
    assert provider_name in _ALL_SUPPORTED_PROVIDERS
    agent = Agent(name="Bot", instructions="Reply.", provider=provider_name)
    engine = resolve_engine(agent)
    assert isinstance(engine, expected_engine_type)


@pytest.mark.parametrize("provider_name", ["OpenAI", "OLLAMA", "Anthropic", "DeepSeek", "Groq"])
def test_supported_providers_case_insensitive(provider_name: str) -> None:
    """Provider names are case-insensitive and whitespace-stripped."""
    agent = Agent(name="Bot", instructions="Reply.", provider=f"  {provider_name}  ")
    engine = resolve_engine(agent)
    assert isinstance(engine, (OpenAIEngine, AnthropicEngine))


# ============================================================================
# 9. Atomic Registry Updates on Validation Failure (Fix 1 Regression Tests)
# ============================================================================


def test_failed_register_agent_leaves_registry_and_manual_names_untouched() -> None:
    """When register_agent() fails validation, _agents_registry and _manual_agent_names remain untouched."""
    root = Agent(name="Root", instructions="Reply.")
    runner = Runner(starting_agent=root)

    initial_registry = dict(runner._agents_registry)
    initial_manual = set(runner._manual_agent_names)

    # Invalid agent: remote custom endpoint without api_key
    invalid = Agent(
        name="InvalidRemote",
        instructions="Reply.",
        base_url="https://llm.example.com/v1",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        runner.register_agent(invalid)

    assert "custom or unofficial base_url was provided" in str(exc_info.value)
    assert runner._agents_registry == initial_registry
    assert runner._manual_agent_names == initial_manual
    assert "InvalidRemote" not in runner._agents_registry
    assert "InvalidRemote" not in runner._manual_agent_names


def test_failed_register_agent_tree_leaves_entire_tree_unregistered() -> None:
    """When an agent in a transitive registration tree fails, no agents from the tree are registered."""
    root = Agent(name="Root", instructions="Reply.")
    runner = Runner(starting_agent=root)

    initial_registry = dict(runner._agents_registry)
    initial_manual = set(runner._manual_agent_names)

    # Valid intermediate agent pointing to invalid peer (remote endpoint without key)
    bad_leaf = Agent(name="BadLeaf", instructions="Reply.", base_url="https://llm.leaf.com/v1")
    valid_branch = Agent(name="ValidBranch", instructions="Reply.", handoffs=[bad_leaf])

    with pytest.raises(ConfigurationError) as exc_info:
        runner.register_agent(valid_branch)

    assert "custom or unofficial base_url was provided" in str(exc_info.value)
    assert runner._agents_registry == initial_registry
    assert runner._manual_agent_names == initial_manual
    assert "ValidBranch" not in runner._agents_registry
    assert "BadLeaf" not in runner._agents_registry


def test_successful_registration_after_previous_failure() -> None:
    """An agent name whose registration failed can subsequently be registered successfully."""
    root = Agent(name="Root", instructions="Reply.")
    runner = Runner(starting_agent=root)

    # 1. Attempt failed registration
    invalid = Agent(
        name="TargetAgent",
        instructions="Reply.",
        base_url="https://llm.example.com/v1",
    )
    with pytest.raises(ConfigurationError):
        runner.register_agent(invalid)

    assert "TargetAgent" not in runner._agents_registry
    assert "TargetAgent" not in runner._manual_agent_names

    # 2. Subsequent registration with corrected configuration succeeds
    valid = Agent(
        name="TargetAgent",
        instructions="Reply.",
        base_url="https://llm.example.com/v1",
        api_key="sk-explicit-valid",
    )
    runner.register_agent(valid)

    assert runner._agents_registry["TargetAgent"] is valid
    assert "TargetAgent" in runner._manual_agent_names


def test_failed_dynamic_handoff_leaves_registry_untouched() -> None:
    """When dynamic handoff resolution fails engine validation, registry remains untouched."""
    root = Agent(name="Root", instructions="Reply.")
    runner = Runner(starting_agent=root)

    initial_registry = dict(runner._agents_registry)

    # Add invalid peer to root's handoffs after runner creation (late handoff)
    invalid_peer = Agent(
        name="LateInvalid",
        instructions="Reply.",
        base_url="https://llm.late-handoff.com/v1",
    )
    root.add_handoff(invalid_peer)

    with pytest.raises(ConfigurationError):
        runner._resolve_handoff_target(root, "LateInvalid")

    assert runner._agents_registry == initial_registry
    assert "LateInvalid" not in runner._agents_registry


def test_failed_late_replacement_preserves_original_agent_in_registry() -> None:
    """When late replacement of an unreachable agent fails validation, original agent is preserved."""
    old_unreachable = Agent(name="Worker", instructions="Old worker.", tools=[])
    root = Agent(name="Root", instructions="Reply.")
    runner = Runner(starting_agent=root)
    # Put old_unreachable directly into registry (simulate previous dynamically resolved agent)
    with runner._registry_lock:
        runner._agents_registry["Worker"] = old_unreachable

    assert runner._agents_registry["Worker"] is old_unreachable
    assert not runner._agent_reachable_unlocked(old_unreachable)

    # Replacement candidate with invalid configuration
    new_invalid = Agent(
        name="Worker",
        instructions="New worker.",
        base_url="https://llm.example.com/v1",
    )
    root.add_handoff(new_invalid)

    with pytest.raises(ConfigurationError):
        runner._resolve_handoff_target(root, "Worker")

    # Original agent must still be in registry, untouched!
    assert runner._agents_registry["Worker"] is old_unreachable


