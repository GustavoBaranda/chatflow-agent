🌐 Language: **English** | [Español](es/MIGRATION_0.4.md)

# Migration Guide: `v0.3.4` → `0.4.0`

This guide describes everything you need to know when upgrading **`chatflow-agent`** from **`v0.3.4`** to **`0.4.0`**.

* **[1. Who Needs to Make Changes?](#1-who-needs-to-make-changes)**
* **[2. Confirmed Breaking Changes (`v0.3.4` → `0.4.0`)](#2-confirmed-breaking-changes-v034--040)**
  * [2.1. `GeminiEngine` Uses the Gemini REST API via `httpx` Instead of `google-genai`](#21-geminiengine-uses-the-gemini-rest-api-via-httpx-instead-of-google-genai)
  * [2.2. `GeminiEngine.client` Attribute Removed](#22-geminiengineclient-attribute-removed)
  * [2.3. Duplicate `Agent` Names Across Distinct Instances Are Rejected](#23-duplicate-agent-names-across-distinct-instances-are-rejected)
  * [2.4. Unused `dedup_ttl_hours` Argument Removed from `WhatsAppChannel`](#24-unused-dedup_ttl_hours-argument-removed-from-whatsappchannel)
* **[3. Other Relevant `0.4.0` Changes & Improvements](#3-other-relevant-040-changes--improvements)**
  * [3.1. New Features](#31-new-features)
  * [3.2. Compatible Behavior Changes & Fail-Closed Validations](#32-compatible-behavior-changes--fail-closed-validations)
  * [3.3. Bug Fixes](#33-bug-fixes)
  * [3.4. Internal Improvements Requiring No Migration](#34-internal-improvements-requiring-no-migration)
* **[4. Related Documentation](#4-related-documentation)**

---

## 1. Who Needs to Make Changes?

For most applications, upgrading from `v0.3.4` to `0.4.0` is a **drop-in upgrade requiring zero code changes**. Use the summary table below to check whether your project is affected by any of the four breaking changes in `0.4.0`.

| Usage Pattern | Action Required? | Details |
| :--- | :--- | :--- |
| Standard `Agent`, `Runner`, `@tool`, `SessionContext`, or `SQLiteSessionStore` usage | **No changes required** | Existing database files, agent definitions, tools, and session APIs remain compatible. |
| `GeminiEngine` / `provider="gemini"` with `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `api_key=...` | **No code changes required** | Now runs over `httpx` with zero extra SDK dependencies (`google-genai` is no longer required). |
| `OpenAIEngine`, `AnthropicEngine`, Groq, xAI Grok, DeepSeek, Ollama, or vLLM | **No changes required** | All OpenAI-compatible and Anthropic engines remain source-compatible. |
| `TelegramChannel` or `CLIChannel` | **No changes required** | Existing constructor calls and webhook handlers continue to work unchanged. |
| `WhatsAppChannel` without `dedup_ttl_hours` | **No changes required** | Constructor calls and webhook handlers that do not pass `dedup_ttl_hours` continue to work unchanged. |
| `WhatsAppChannel(..., dedup_ttl_hours=...)` | **Yes — Migration required** | Configure the TTL on `SessionStore` or `SQLiteSessionStore` instead. See [Section 2.4](#24-unused-dedup_ttl_hours-argument-removed-from-whatsappchannel). |
| `GeminiEngine` relying on **Google Cloud Vertex AI / Application Default Credentials (ADC)** without an API key | **Yes — Migration required** | See [Section 2.1](#21-geminiengine-uses-the-gemini-rest-api-via-httpx-instead-of-google-genai). |
| Code or tests accessing **`GeminiEngine.client`** directly | **Yes — Migration required** | See [Section 2.2](#22-geminiengineclient-attribute-removed). |
| Multi-agent graphs registering **distinct `Agent` instances that share the same `name`** | **Yes — Migration required** | See [Section 2.3](#23-duplicate-agent-names-across-distinct-instances-are-rejected). |

> [!NOTE]
> Security and isolation changes already released in **`v0.3.3`** (strict local hostname validation for keyless `OpenAIEngine(base_url=...)`, PRs #74/#75) and **`v0.3.4`** (strict per-provider environment API key isolation, PR #126 / `GHSA-mcgm-q29p-3984`) are already part of `v0.3.4` and are not new breaking changes in `0.4.0`.

---

## 2. Confirmed Breaking Changes (`v0.3.4` → `0.4.0`)

### 2.1. `GeminiEngine` Uses the Gemini REST API via `httpx` Instead of `google-genai`

* **Origin:** PR #130 (`308fbfc`, Issue #116)
* **Previous behavior (`v0.3.4`):**
  * `google-genai>=0.3.0` was a core package dependency in `pyproject.toml`.
  * `GeminiEngine` initialized `genai.Client(api_key=resolved_key)`. When neither `api_key`, `GEMINI_API_KEY`, nor `GOOGLE_API_KEY` was set, `GeminiEngine` logged a warning and allowed `genai.Client()` to fall back to Google Cloud Vertex AI / Application Default Credentials (`GOOGLE_GENAI_USE_VERTEXAI` / ADC).
* **Current behavior (`0.4.0`):**
  * `google-genai` has been removed from `pyproject.toml`. Core dependencies are now solely `pydantic>=2.6.0` and `httpx>=0.27.0`.
  * `GeminiEngine` communicates directly with the Google Gemini REST API (`https://generativelanguage.googleapis.com/v1beta` by default, or overridden via `GOOGLE_GEMINI_BASE_URL`) via `httpx.AsyncClient` using the `x-goog-api-key` request header (or `api_key_in_query=True`), and accepts `http_client: httpx.AsyncClient | None = None`.
  * If no API key is passed via `api_key=...` and neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` is set in the environment, `GeminiEngine.generate_turn_async()` raises `ProviderError` on the first turn (fail-closed) rather than attempting Vertex AI ADC authentication.
* **Affected users:**
  * Applications using `GeminiEngine` without an API key by relying on Vertex AI Application Default Credentials (`GOOGLE_GENAI_USE_VERTEXAI=true`).
  * Environments whose dependency scripts explicitly expect `chatflow-agent` to install `google-genai` transitively for unrelated application code.
* **Required migration:**
  1. For standard Gemini API usage, provide a Gemini API key via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, or `Agent(..., api_key="...")` / `GeminiEngine(api_key="...")`).
  2. For Google Cloud Vertex AI workloads, either route through Vertex AI's OpenAI-compatible endpoint using `OpenAIEngine(base_url=..., api_key=...)`, or implement a custom `BaseEngine` subclass wrapping `google-genai` (and list `google-genai` directly in your project's dependencies).

**Before (`v0.3.4` — implicit Vertex AI ADC fallback without API key):**

```python
import os
from chatflow_agent import Agent

# In v0.3.4, omitting GEMINI_API_KEY/GOOGLE_API_KEY logged a warning
# and relied on google-genai Vertex AI ADC env vars:
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    provider="gemini",
    model="gemini-2.5-flash",
)
```

**After (`0.4.0` — explicit Gemini API key or custom `http_client`):**

```python
from chatflow_agent import Agent, GeminiEngine

# Option A: Standard Gemini API key (via GEMINI_API_KEY env var or api_key parameter)
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    provider="gemini",
    model="gemini-2.5-flash",
    api_key="AIzaSyYourGeminiKey",
)

# Option B: Direct GeminiEngine instance
engine = GeminiEngine(
    api_key="AIzaSyYourGeminiKey",
    timeout_seconds=30.0,
)
assert engine.api_key == "AIzaSyYourGeminiKey"
```

---

### 2.2. `GeminiEngine.client` Attribute Removed

* **Origin:** PR #130 (`308fbfc`, Issue #116)
* **Previous behavior (`v0.3.4`):**
  * `GeminiEngine.__init__()` created `self.client = genai.Client(...)`. Some test suites or custom extensions monkeypatched `engine.client.aio.models.generate_content` to mock Gemini responses.
* **Current behavior (`0.4.0`):**
  * `GeminiEngine` no longer has a `.client` attribute (accessing `engine.client` raises `AttributeError`).
  * Instead, `GeminiEngine` accepts `http_client: httpx.AsyncClient | None = None` in its constructor (and also supports `Runner`'s per-event-loop `_ClientPool` contextvar when entered via `async with runner:`).
* **Affected users:**
  * Code or unit tests that read, mock, or monkeypatch `engine.client` or `engine.client.aio.models.generate_content`.
* **Required migration:**
  * Replace mocks targeting `engine.client` with either:
    1. A custom `BaseEngine` stub passed to `Runner(starting_agent=..., engine=...)`, or
    2. An `httpx.AsyncClient(transport=httpx.MockTransport(...))` injected via `GeminiEngine(..., http_client=client)`.

**Before (`v0.3.4` — monkeypatching `engine.client`):**

```python
from unittest.mock import AsyncMock
from chatflow_agent import GeminiEngine

# Legacy v0.3.4 pattern (raises AttributeError in 0.4.0):
# engine = GeminiEngine(api_key="test-key")
# engine.client.aio.models.generate_content = AsyncMock(...)
```

**After (`0.4.0` — injecting `http_client` with `httpx.MockTransport`):**

```python
import asyncio
import httpx
from chatflow_agent import Agent, GeminiEngine, Message, Role


def mock_gemini_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Mocked Gemini 0.4 response"}],
                    }
                }
            ]
        },
    )


async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a concise assistant.")
    transport = httpx.MockTransport(mock_gemini_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        engine = GeminiEngine(
            api_key="test-gemini-key",
            http_client=http_client,
        )
        turn_result = await engine.generate_turn_async(
            agent=agent,
            history=[Message(role=Role.USER, content="Hello!")],
        )
        assert turn_result.text == "Mocked Gemini 0.4 response"


asyncio.run(main())
```

---

### 2.3. Duplicate `Agent` Names Across Distinct Instances Are Rejected

* **Origin:** PR #128 (`a3bcd02`, Issue #123)
* **Previous behavior (`v0.3.4`):**
  * `Runner._register_agent_tree()` indexed agents by `agent.name` in `self._agents_registry` using `if agent.name in self._agents_registry: return`.
  * If two **distinct** `Agent` instances shared the same `name` within a handoff graph, the second instance was **silently ignored**, routing handoffs to the first instance and ignoring the second instance's instructions, tools, and sub-handoffs.
* **Current behavior (`0.4.0`):**
  * `Runner` compares object identity (`existing is agent`):
    * **Cyclic / shared references** to the **same** `Agent` instance (`a1 -> a2 -> a1`, or diamond graphs sharing a single `Agent` object) remain fully supported.
    * **Two distinct `Agent` instances** (`existing is not agent`) with the same `name` in the handoff graph now raise `ValueError: Duplicate agent name '<name>' detected in handoff graph. Each distinct Agent instance must have a unique name.` during `Runner(...)` initialization or dynamic post-init registration.
* **Affected users:**
  * Multi-agent applications that instantiate multiple separate `Agent(name="...")` objects with identical names in the same `Runner` handoff graph.
* **Required migration:**
  * Give each distinct `Agent` instance a unique `name`, or reuse the same `Agent` instance when multiple parent agents hand off to the same specialist.

**Before (`v0.3.4` — two distinct `Agent` objects with the same `name`):**

```python
from chatflow_agent import Agent, Runner

specialist_v1 = Agent(name="Support", instructions="Tier 1 support")
specialist_v2 = Agent(name="Support", instructions="Tier 2 support")
triage = Agent(
    name="Triage",
    instructions="Route requests to support.",
    handoffs=[specialist_v1, specialist_v2],
)

# In v0.3.4, specialist_v2 was silently dropped.
# In 0.4.0, this raises ValueError immediately:
try:
    Runner(starting_agent=triage)
except ValueError as exc:
    assert "Duplicate agent name 'Support'" in str(exc)
```

**After (`0.4.0` — unique names for distinct agents, or shared instance):**

```python
from chatflow_agent import Agent, Runner

# Fix 1: Assign distinct names to distinct Agent instances
tier1_support = Agent(name="Tier1Support", instructions="Tier 1 support")
tier2_support = Agent(name="Tier2Support", instructions="Tier 2 support")
triage = Agent(
    name="Triage",
    instructions="Route requests to support.",
    handoffs=[tier1_support, tier2_support],
)
runner = Runner(starting_agent=triage)

# Fix 2: Reuse the exact same Agent instance across cyclic/diamond handoffs
shared_billing = Agent(name="Billing", instructions="Shared billing specialist")
tier1_support.add_handoff(shared_billing)
tier2_support.add_handoff(shared_billing)
runner_shared = Runner(starting_agent=triage)
```

---

### 2.4. Unused `dedup_ttl_hours` Argument Removed from `WhatsAppChannel`

* **Origin:** PR #154
* **Previous behavior (`v0.3.4`):**
  * `WhatsAppChannel.__init__` accepted `dedup_ttl_hours` (default 192) between `on_24h_window_expired` and `max_body_size`.
  * The constructor stored that value. Dedup always used the TTL on `SessionStore` or `SQLiteSessionStore` (also default 192 hours).
* **Current behavior (`0.4.0`):**
  * Passing `dedup_ttl_hours` to `WhatsAppChannel` raises `TypeError`.
  * `max_body_size`, `enable_docs`, and `queue_timeout_seconds` are keyword-only. A positional value in the old `dedup_ttl_hours` slot raises `TypeError`.
* **Affected users:**
  * Code that passes `dedup_ttl_hours` to `WhatsAppChannel`, by name or by position.
* **Required migration:**
  * Remove `dedup_ttl_hours` from the `WhatsAppChannel` call.
  * Pass that same number to the session store, then give the store to the runner. For an in-memory bot, use `SessionStore(dedup_ttl_hours=...)` the same way.

**Before (`v0.3.4`):**

```python
channel = WhatsAppChannel(
    verify_token="token",
    app_secret="secret",
    dedup_ttl_hours=48,
)
```

**After (`0.4.0`):**

```python
from chatflow_agent import Agent, Runner, SQLiteSessionStore, WhatsAppChannel

agent = Agent(name="Bot", instructions="Help the user.")
store = SQLiteSessionStore("chatflow.db", dedup_ttl_hours=48)
runner = Runner(starting_agent=agent, session_store=store)
channel = WhatsAppChannel(verify_token="token", app_secret="secret")
channel.attach(runner)
```

---

## 3. Other Relevant `0.4.0` Changes & Improvements

The following enhancements, validations, and bug fixes were added between `v0.3.4` and `0.4.0`. None of them break valid `v0.3.4` configurations.

### 3.1. New Features

1. **`Runner` Async Context Manager for Shared HTTP Connection Pooling (`async with runner:`):**
   * Entering `async with runner:` creates an event-loop-scoped `_ClientPool` (`contextvars.ContextVar`) that reuses `httpx.AsyncClient` connections across turns for `GeminiEngine` on that loop, closing the pooled client cleanly on exit (PR #130).
   * Using `runner.run_async()` without `async with runner:` remains 100% supported (each HTTP attempt creates and closes its own client as fallback).
2. **Whole-Turn Timeout (`turn_timeout`) and `Runner.spawn_turn()`:**
   * `Runner.run_async(..., turn_timeout: float | None = None)` bounds the active execution of a turn (starting once the session lock is held), cancelling and rolling back the turn and raising `TimeoutError` on expiration (PRs #120, #131).
   * `Runner.spawn_turn(session_id, user_message, metadata=None) -> asyncio.Task[AgentResponse]` queues an independent follow-up turn as a background task on the running event loop (PR #120).
3. **Separate Queue Wait Timeout in `WhatsAppChannel` (`queue_timeout_seconds`):**
   * `WhatsAppChannel.__init__(..., queue_timeout_seconds: float = 120.0)` bounds how long a newly arrived webhook message waits for the per-session lock independently of the active execution budget `llm_timeout_seconds`; the argument is optional and defaults to `120.0`, and any value passed must be a positive, finite `float` (`TypeError` for `None`, `ValueError` for zero, negative or non-finite values) (PR #131).
4. **In-Memory `SQLiteSessionStore(db_path=":memory:")` and Lifecycle Methods:**
   * `SQLiteSessionStore` now supports `db_path=":memory:"` using a single persistent shared connection guarded by a threading lock, plus explicit `store.close()` and context manager support (`with SQLiteSessionStore(...) as store:`) (PRs #107, #120, #133).
5. **Public `Tool.to_openai_parameters()` Helper:**
   * `Tool.to_openai_parameters()` is now part of the public `Tool` API, returning the strict JSON Schema `parameters` object shared across OpenAI, Anthropic, and Gemini adapters (PR #118).
6. **Interactive `/reset` Command in `CLIChannel`:**
   * Users in `CLIChannel` can now type `/reset` during an interactive session to clear the conversation history without restarting the process; the active agent and metadata are kept (PR #119).
7. **Auto-Detection of `deepseek-*` Model Prefix in `resolve_engine()`:**
   * Setting `model="deepseek-chat"` or `model="deepseek-reasoner"` without an explicit `provider` now automatically selects `provider="deepseek"` (`https://api.deepseek.com`, `DEEPSEEK_API_KEY`) (PR #55).
8. **New Public Exceptions Exported from `chatflow_agent` and `chatflow_agent.exceptions`:**
   * `ToolSchemaError` (subclass of `ChatFlowError`, PRs #118, #136)
   * `ReentrantTurnError` (subclass of `ChatFlowError` and `RuntimeError`, PR #120)
   * `SessionDeletedError` (subclass of `ChatFlowError`, PR #120)

```python
import asyncio
import httpx
from chatflow_agent import (
    Agent,
    ChatFlowError,
    GeminiEngine,
    ReentrantTurnError,
    Runner,
    SQLiteSessionStore,
    SessionDeletedError,
    Tool,
    ToolSchemaError,
)


def lookup_order(order_id: str) -> str:
    """Look up an order status by ID."""
    return f"Order {order_id}: shipped"


tool_obj = Tool(func=lookup_order)
schema = tool_obj.to_openai_parameters()
assert schema["required"] == ["order_id"]
assert issubclass(ToolSchemaError, ChatFlowError)
assert issubclass(ReentrantTurnError, ChatFlowError)
assert issubclass(SessionDeletedError, ChatFlowError)


async def demo_new_features() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Order is shipped!"}]}}]},
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        engine = GeminiEngine(api_key="demo", http_client=client)
        agent = Agent(
            name="OrderBot",
            instructions="Help users check their order status.",
            tools=[lookup_order],
        )
        with SQLiteSessionStore(db_path=":memory:") as store:
            runner = Runner(starting_agent=agent, session_store=store, engine=engine)
            async with runner:
                reply = await runner.run_async("user-1", "Check order #42", turn_timeout=10.0)
                assert reply.content == "Order is shipped!"


asyncio.run(demo_new_features())
```

---

### 3.2. Compatible Behavior Changes & Fail-Closed Validations

* **Fail-Fast Re-Entrant Same-Session Detection (`ReentrantTurnError`):** Calling `runner.run_async(session, ...)` recursively from inside a tool executing on the **same** `session_id` now immediately raises `ReentrantTurnError` instead of deadlocking forever on the per-session lock (PR #120).
* **Fail-Closed Tool Schema Validation (`ToolSchemaError`):** Registering a `@tool` function with an unmappable parameter type annotation (such as an arbitrary custom class, `object`, `callable`, or bare `tuple`) now raises `ToolSchemaError` at decoration/registration time instead of silently coercing the parameter to `"type": "string"` (PRs #118, #136).
* **Deleted Session Guard (`SessionDeletedError`):** If a `SQLiteSessionContext` is deleted via `store.delete(session_id)` while a turn is queued or executing, the running turn fails with `SessionDeletedError` and rolls back without leaving rows behind, and a queued turn then starts a new, empty session (PR #120).
* **`WhatsAppChannel` Structured Timeout Warnings:** Timeout log messages in `WhatsAppChannel` were refined to distinguish queue lock contention (`"Session busy for %s after %ss; skipping turn."`), turn execution timeouts (`"Turn timed out for %s after %ss."`), and backstop expiration (`"Dispatch backstop expired for %s after %ss"` / `"TimeoutError raised while dispatching message from %s"`) (PR #131).
* **`Runner.run()` Inside an Active Event Loop:** Calling the synchronous helper `Runner.run()` from inside an already running `asyncio` event loop now raises a clear `RuntimeError("Runner.run() cannot be called from a running event loop. Use `await runner.run_async(...)` instead.")` (PR #57).
* **Constructor Input Validation (`ValueError`):**
  * `Agent(name="", instructions="...")` or whitespace-only `Agent.name` raises `ValueError` (PR #86).
  * `SQLiteSessionStore(..., default_max_turns=0)` or `SessionContext(..., max_turns=0)` (or `< 0`) raises `ValueError` (PR #63).
  * `split_message(text, max_len=0)` (or `< 0`) raises `ValueError` (PR #46).

---

### 3.3. Bug Fixes

* **SQLite Session Concurrency & Atomic Turn Journal (`SQLiteSessionStore`):**
  * Concurrent turns targeting the same `session_id` through distinct `SQLiteSessionContext` wrapper instances now serialize via a per-`(event_loop, session_id)` weak lock registry and automatically reload the latest committed SQLite state (`session.refresh()`) upon acquiring the lock (PR #120).
  * `SQLiteSessionContext.restore_history()` now executes inside a single `BEGIN IMMEDIATE` transaction using the in-memory turn journal, preventing race conditions and preserving rows inserted concurrently (PR #120).
* **Atomic Turn Rollback on Errors and Cancellation:**
  * When a turn fails with an unhandled exception, times out (`turn_timeout`), or is cancelled (`asyncio.CancelledError`), `Runner` atomically restores the full pre-turn checkpoint: message history (including any old messages pruned by `SessionContext.max_turns`), `session.active_agent`, and a deep copy of `session.metadata` (PRs #34, #52, #69).
* **Cross-Provider Tool Call ID Normalization:**
  * `replacement_tool_call_id(index, seed)` generates deterministic alphanumeric IDs when normalizing tool calls across providers, preventing collisions during multi-provider handoffs (PR #130).
* **`Role.SYSTEM` History Message Forwarding:**
  * Mid-conversation `Role.SYSTEM` messages stored in session history are now merged and forwarded alongside `system_prompt` across `GeminiEngine`, `OpenAIEngine`, and `AnthropicEngine` instead of being silently dropped (PR #100).
* **Resilient `ProviderError` Wrapping for Malformed Provider Payloads:**
  * Non-JSON HTTP responses, missing `choices`/`candidates`, and malformed tool call JSON arguments from upstream providers are now consistently wrapped in `ProviderError` rather than leaking raw `JSONDecodeError`, `IndexError`, or `KeyError` exceptions (PR #124).
* **`WhatsAppChannel` Webhook Hardening:**
  * Ignored `reaction` and `system` webhook notifications and malformed nested payloads no longer trigger unhandled exceptions, and non-object JSON webhook bodies (`[]`, `"str"`, `123`) now return HTTP `400 Bad Request` (PRs #110, #117, #129).
* **Handoff Naming & Dynamic Registration Fixes:**
  * Non-ASCII agent names (e.g., `"Política"`, `"Müller"`) transliterate cleanly into valid `[a-z0-9_]` handoff tool names, collisions after sanitization are automatically disambiguated (`transfer_to_support_2`), and calling `agent.add_handoff()` after `Runner` initialization dynamically registers the target tree (PRs #49, #108, #112).
* **`resolve_engine()` Precedence with `prefer_default_engine=True`:**
  * When an `Agent` inside a multi-agent swarm inherits default `provider="gemini"` and `model="gemini-2.5-flash"` without explicit overrides, `Runner` now respects the `Runner(engine=...)` fallback engine prior to falling back to environment defaults (PR #39).
* **`Tool` Introspection & JSON Schema Edge Cases:**
  * Fixed schema generation for modules using `from __future__ import annotations` (PR #72), PEP 604 unions (`str | int`, `str | None`, PRs #26, #118), `typing.Annotated[T, ...]` (PR #62), typed containers (`list[int]`, `set[str]`, `dict[str, int]`, PRs #40, #87), sync and async `functools.partial` callables (PRs #43, #111), variadic `*args`/`**kwargs` parameters (PR #35), and tool parameters literally named `default` (PR #68).

---

### 3.4. Internal Improvements Requiring No Migration

* **Non-Blocking File-Backed SQLite I/O:** File-backed `SQLiteSessionStore` instances offload blocking SQLite operations to a dedicated 4-worker `ThreadPoolExecutor` via `_run_blocking()` when called from an async context, preventing event loop stalls under load (PR #133).
* **Atomic `wamid` Deduplication:** `SQLiteSessionStore.record_processed_message()` records the message id with a single conditional upsert, `INSERT INTO chatflow_processed_messages ... ON CONFLICT(message_id) DO UPDATE SET processed_at = CURRENT_TIMESTAMP WHERE processed_at < datetime('now', ?)`: an id recorded within the TTL is left untouched (0 rows changed), and an expired record is renewed. Expired records are then pruned by a separate `DELETE` in the same transaction (PR #120).
* **Automatic Pool Scoping in Channels and `Runner.run()`:** `Runner.run()`, `TelegramChannel.run()`, and `WhatsAppChannel` lifespan automatically enter `async with runner:` so connection pooling is active out of the box (PR #130).
* **Full SQLite Schema Backward Compatibility:** Existing SQLite database files created in `v0.3.x` work without manual schema migration in `0.4.0`.

---

## 4. Related Documentation

* **[Architecture Guide](ARCHITECTURE.md):** Detailed breakdown of the `Runner` execution loop, engine resolution, tool introspection, handoffs, SQLite concurrency, and channels.
* **[English README](../README.md):** Quickstart, provider reference table, and complete feature overview.
* **[Changelog](../CHANGELOG.md):** Chronological release notes.
