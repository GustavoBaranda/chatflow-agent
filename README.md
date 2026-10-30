# chatflow-agent

[![CI](https://github.com/GustavoBaranda/chatflow-agent/actions/workflows/test.yml/badge.svg)](https://github.com/GustavoBaranda/chatflow-agent/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

🌐 **Language:** **English** | [Español](README.es.md) · [Architecture](docs/ARCHITECTURE.md) · [Changelog](CHANGELOG.md) · [Contributing](#contributing) · [PyPI Package](https://pypi.org/project/chatflow-agent/) · [Report Issue](https://github.com/GustavoBaranda/chatflow-agent/issues)

> **Build autonomous multi-agent swarms for WhatsApp, Telegram, Webhooks, and CLI in Python with a lightweight, provider-agnostic core.**

```bash
pip install chatflow-agent
```

`chatflow-agent` is an async-first Python framework for orchestrating multi-agent conversational workflows across real-world messaging channels. Inspired by lightweight agent handoff patterns, it combines provider-agnostic LLM engines with built-in concurrency locks, transactional turn rollbacks, SQLite persistence, and turnkey channel adapters for **WhatsApp (Meta Cloud API)**, **Telegram**, **FastAPI Webhooks**, and **CLI**.

![chatflow-agent Architecture: Input Channels (WhatsApp, Telegram, Webhook, CLI) connected to Runner and Session Memory, orchestrating a Triage Agent and a Specialist Agent with autonomous peer handoff and @agent.tool actions](docs/images/architecture-en.png)

---

## Table of Contents

1. [Key Features](#key-features)
2. [Installation & Requirements](#installation--requirements)
3. [Quickstart](#quickstart)
4. [Our Contributors](#our-contributors)
5. [Supported Providers & Configuration](#supported-providers--configuration)
6. [Agents, Tools & Handoffs](#agents-tools--handoffs)
7. [Sessions, Persistence & Concurrency](#sessions-persistence--concurrency)
8. [Tool Idempotency & Side-Effect Handling](#tool-idempotency--side-effect-handling)
9. [Available Channels](#available-channels)
10. [Usage Examples](#usage-examples)
11. [Technical Documentation](#technical-documentation)
12. [Contributing](#contributing)
13. [Support the Project](#support-the-project)
14. [License](#license)

---

## Key Features

* **Ultra-Lightweight Core:** Only 2 core runtime dependencies (`pydantic>=2.6.0`, `httpx>=0.27.0`). All LLM engines (**Google Gemini**, **OpenAI**, **Anthropic Claude**, **xAI Grok**, **DeepSeek**, **Groq**, **Ollama**, and **vLLM**) work out of the box over native `httpx` REST without heavy SDK chains; optional channel dependencies are installed only when needed.
* **Omnichannel Connectors:** Production-ready adapters for **WhatsApp (Meta Cloud API & generic webhooks)**, **Telegram**, **FastAPI Webhooks**, and interactive **CLI**.
* **Multi-Provider Freedom:** Run agents on **Google Gemini**, **OpenAI**, **Anthropic Claude**, **xAI Grok**, **DeepSeek**, **Groq**, or local models via **Ollama** (**Google Gemma**, Llama, Mistral, Qwen, Phi) and **vLLM**—even mixing different providers within the same multi-agent swarm.
* **Autonomous Peer Handoffs:** Agents delegate conversations to specialist peers dynamically based on user intent, persisting the active specialist and session metadata across turns and server restarts.
* **Production-Hardened Resilience:** Per-session `asyncio.Lock` serialization, non-blocking worker-thread offloading for file-backed SQLite stores, automatic history/agent/metadata rollback on mid-turn failures or timeouts, and an anti-500 webhook shield for Meta Cloud API.

---

## Installation & Requirements

* **Python:** `3.10`, `3.11`, `3.12`, or `3.13`
* **Core dependencies:** `pydantic>=2.6.0`, `httpx>=0.27.0`

Install only the extras your deployment requires:

```bash
# Core framework (all LLM engines via HTTPX, SQLite & in-memory stores, CLI channel)
pip install chatflow-agent

# With WhatsApp Webhook connector (FastAPI + Uvicorn)
pip install "chatflow-agent[whatsapp]"

# With Telegram Bot connector (python-telegram-bot)
pip install "chatflow-agent[telegram]"

# With custom FastAPI Webhook service dependencies (FastAPI + Uvicorn)
pip install "chatflow-agent[webhook]"

# Complete channel bundle (telegram, whatsapp, and webhook extras)
pip install "chatflow-agent[all]"
```

---

## Quickstart

### Async Multi-Agent Quickstart (Recommended for Webhooks & Bots)

```python
import asyncio
from chatflow_agent import Agent, Runner

# 1. Define specialist agents
tech_agent = Agent(
    name="TechSupport",
    model="gemini-2.5-flash",
    instructions="Help customers troubleshoot technical issues.",
)
billing_agent = Agent(
    name="Billing",
    model="gemini-2.5-flash",
    instructions="Help customers with invoices, subscriptions, and payments.",
)

# 2. Frontline agent with autonomous peer handoffs
receptionist = Agent(
    name="Receptionist",
    model="gemini-2.5-flash",
    instructions="Greet the customer and transfer them to TechSupport or Billing.",
    handoffs=[tech_agent, billing_agent],
)

async def main() -> None:
    # Using 'async with Runner(...)' pools HTTP connections across turns for GeminiEngine
    async with Runner(starting_agent=receptionist) as runner:
        result = await runner.run_async(
            session_id="user_whatsapp_1",
            user_message="Hi! I need help with an invoice error on my account.",
        )
        print(f"[{result.active_agent_name}]: {result.content}")
        # Output: [Billing]: I would be glad to help check your invoice. Could you share your account ID?

if __name__ == "__main__":
    asyncio.run(main())
```

### Synchronous Quickstart (`runner.run()`)

For CLI scripts, cron jobs, or synchronous applications where no event loop is running, call `runner.run()` directly:

```python
from chatflow_agent import Agent, Runner

agent = Agent(
    name="Assistant",
    model="gemini-2.5-flash",
    instructions="Answer concisely.",
)

runner = Runner(starting_agent=agent)
response = runner.run(session_id="sync_user_1", user_message="Hello!")
print(f"[{response.active_agent_name}]: {response.content}")
```

> [!NOTE]
> `runner.run()` manages its own event loop via `asyncio.run()`. Calling `runner.run()` from inside an already running event loop (such as a FastAPI endpoint or an `async def` function) raises a `RuntimeError` directing you to `await runner.run_async(...)` instead.

---

## Our Contributors

`chatflow-agent` is built with the help of an international community of developers who contribute code, bug fixes, security hardening, test coverage, and documentation improvements. Every contribution—large or small—helps keep the framework reliable and accessible.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/GustavoBaranda">
          <img src="https://github.com/GustavoBaranda.png?size=64" width="64" height="64" alt="@GustavoBaranda" /><br />
          <sub><b>@GustavoBaranda</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/KeuqS">
          <img src="https://github.com/KeuqS.png?size=64" width="64" height="64" alt="@KeuqS" /><br />
          <sub><b>@KeuqS</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/mehdimirzafaruk">
          <img src="https://github.com/mehdimirzafaruk.png?size=64" width="64" height="64" alt="@mehdimirzafaruk" /><br />
          <sub><b>@mehdimirzafaruk</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/wzd8888">
          <img src="https://github.com/wzd8888.png?size=64" width="64" height="64" alt="@wzd8888" /><br />
          <sub><b>@wzd8888</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/kudala-bharani">
          <img src="https://github.com/kudala-bharani.png?size=64" width="64" height="64" alt="@kudala-bharani" /><br />
          <sub><b>@kudala-bharani</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/mikemikimike">
          <img src="https://github.com/mikemikimike.png?size=64" width="64" height="64" alt="@mikemikimike" /><br />
          <sub><b>@mikemikimike</b></sub>
        </a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/ganeshmshetty">
          <img src="https://github.com/ganeshmshetty.png?size=64" width="64" height="64" alt="@ganeshmshetty" /><br />
          <sub><b>@ganeshmshetty</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/krishna2707">
          <img src="https://github.com/krishna2707.png?size=64" width="64" height="64" alt="@krishna2707" /><br />
          <sub><b>@krishna2707</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/jsdhwfmax">
          <img src="https://github.com/jsdhwfmax.png?size=64" width="64" height="64" alt="@jsdhwfmax" /><br />
          <sub><b>@jsdhwfmax</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/aazminu">
          <img src="https://github.com/aazminu.png?size=64" width="64" height="64" alt="@aazminu" /><br />
          <sub><b>@aazminu</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/HarshRajSinghania">
          <img src="https://github.com/HarshRajSinghania.png?size=64" width="64" height="64" alt="@HarshRajSinghania" /><br />
          <sub><b>@HarshRajSinghania</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/ompatil1906">
          <img src="https://github.com/ompatil1906.png?size=64" width="64" height="64" alt="@ompatil1906" /><br />
          <sub><b>@ompatil1906</b></sub>
        </a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/sarantez">
          <img src="https://github.com/sarantez.png?size=64" width="64" height="64" alt="@sarantez" /><br />
          <sub><b>@sarantez</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/hyverysmart">
          <img src="https://github.com/hyverysmart.png?size=64" width="64" height="64" alt="@hyverysmart" /><br />
          <sub><b>@hyverysmart</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/Aaqibhafeezkhan">
          <img src="https://github.com/Aaqibhafeezkhan.png?size=64" width="64" height="64" alt="@Aaqibhafeezkhan" /><br />
          <sub><b>@Aaqibhafeezkhan</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/SumiRann1">
          <img src="https://github.com/SumiRann1.png?size=64" width="64" height="64" alt="@SumiRann1" /><br />
          <sub><b>@SumiRann1</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/jay23606">
          <img src="https://github.com/jay23606.png?size=64" width="64" height="64" alt="@jay23606" /><br />
          <sub><b>@jay23606</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/Iqranawaz-debug">
          <img src="https://github.com/Iqranawaz-debug.png?size=64" width="64" height="64" alt="@Iqranawaz-debug" /><br />
          <sub><b>@Iqranawaz-debug</b></sub>
        </a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/dbdefault">
          <img src="https://github.com/dbdefault.png?size=64" width="64" height="64" alt="@dbdefault" /><br />
          <sub><b>@dbdefault</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/Erfouni">
          <img src="https://github.com/Erfouni.png?size=64" width="64" height="64" alt="@Erfouni" /><br />
          <sub><b>@Erfouni</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/bercedev">
          <img src="https://github.com/bercedev.png?size=64" width="64" height="64" alt="@bercedev" /><br />
          <sub><b>@bercedev</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/Gokulnaath07">
          <img src="https://github.com/Gokulnaath07.png?size=64" width="64" height="64" alt="@Gokulnaath07" /><br />
          <sub><b>@Gokulnaath07</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/tyza66">
          <img src="https://github.com/tyza66.png?size=64" width="64" height="64" alt="@tyza66" /><br />
          <sub><b>@tyza66</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/rahul05ranjan">
          <img src="https://github.com/rahul05ranjan.png?size=64" width="64" height="64" alt="@rahul05ranjan" /><br />
          <sub><b>@rahul05ranjan</b></sub>
        </a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/bfs2021">
          <img src="https://github.com/bfs2021.png?size=64" width="64" height="64" alt="@bfs2021" /><br />
          <sub><b>@bfs2021</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/MbappeWU">
          <img src="https://github.com/MbappeWU.png?size=64" width="64" height="64" alt="@MbappeWU" /><br />
          <sub><b>@MbappeWU</b></sub>
        </a>
      </td>
      <td align="center" valign="top" width="16.66%">
        <a href="https://github.com/madhavidayma21">
          <img src="https://github.com/madhavidayma21.png?size=64" width="64" height="64" alt="@madhavidayma21" /><br />
          <sub><b>@madhavidayma21</b></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

See the [full contributor graph on GitHub](https://github.com/GustavoBaranda/chatflow-agent/graphs/contributors) or check the [Contributing](#contributing) section to join the project.

---

## Supported Providers & Configuration

### Configuring API Keys

You can supply credentials using any of the following three methods:

#### 1. Direct in Python Code

Pass `api_key` to `Runner` (shared with the default engine) or directly to an individual `Agent`:

```python
# Pass to Runner (used by the default GeminiEngine)
runner = Runner(starting_agent=my_agent, api_key="AIzaSyYourGeminiKey")

# Or pass directly to a specific Agent
grok_agent = Agent(
    name="GrokSpecialist",
    provider="grok",
    model="grok-2-latest",
    api_key="xai-your-key-here",
    instructions="You are a research specialist.",
)
```

#### 2. Using a `.env` File

Create a `.env` file in your project root and load it with `python-dotenv` (optional external package: `pip install python-dotenv`):

```env
GEMINI_API_KEY="AIzaSy..."
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
XAI_API_KEY="xai-..."
DEEPSEEK_API_KEY="sk-..."
GROQ_API_KEY="gsk_..."
```

```python
from dotenv import load_dotenv

load_dotenv()
```

#### 3. Via Terminal Environment Variables

* **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="AIzaSy..."
  ```
* **Linux / macOS (Bash/Zsh):**
  ```bash
  export GEMINI_API_KEY="AIzaSy..."
  ```

### Provider Reference Table

| Provider | Engine | Environment Variable | `Agent` Configuration |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | `GeminiEngine` | `GEMINI_API_KEY` | `model="gemini-2.5-flash"` |
| **OpenAI** | `OpenAIEngine` | `OPENAI_API_KEY` | `provider="openai"`, `model="gpt-4o-mini"` |
| **Anthropic Claude** | `AnthropicEngine` | `ANTHROPIC_API_KEY` | `provider="anthropic"`, `model="claude-3-5-sonnet-latest"` |
| **xAI Grok** | `OpenAIEngine` | `XAI_API_KEY` | `provider="grok"`, `model="grok-2-latest"` |
| **DeepSeek** | `OpenAIEngine` | `DEEPSEEK_API_KEY` | `provider="deepseek"`, `model="deepseek-chat"` |
| **Groq** | `OpenAIEngine` | `GROQ_API_KEY` | `provider="groq"`, `model="llama-3.3-70b-versatile"` |
| **Ollama (Local)** | `OpenAIEngine` | *None required* | `provider="ollama"`, `model="gemma2:9b"` |
| **vLLM / Local Server** | `OpenAIEngine` | *None on `localhost`* | `base_url="http://localhost:8000/v1"`, `model="..."` |

* **Built-in HTTPX Engines:** All engines (`GeminiEngine`, `OpenAIEngine`, and `AnthropicEngine`) are included out of the box with `pip install chatflow-agent`—no extra provider SDKs required.
* **Additional Environment Variables:** `GeminiEngine` also accepts `GOOGLE_API_KEY` as a fallback and `GOOGLE_GEMINI_BASE_URL` for custom endpoints. `Ollama` optionally reads `OLLAMA_API_KEY`, and any `OpenAIEngine` preset honors `<PROVIDER>_BASE_URL` (e.g., `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`).
* **Model Prefix Auto-Detection:** When `provider` is omitted on `Agent`, `resolve_engine()` automatically selects the engine from `agent.model`:
  * **Google Gemini (`GeminiEngine`):** `gemini-*` (or default fallback)
  * **OpenAI (`OpenAIEngine`):** `gpt-*`, `o1*`, `o3*`, `chatgpt-*`
  * **Anthropic Claude (`AnthropicEngine`):** `claude-*`
  * **xAI Grok (`OpenAIEngine`):** `grok*`
  * **DeepSeek (`OpenAIEngine`):** `deepseek-*`
  * **Ollama Local (`OpenAIEngine`):** `gemma*`, `llama*`, `mistral*`, `qwen*`, `phi*`
  * **Groq & vLLM / Local Servers (`OpenAIEngine`):** Selected explicitly via `provider="groq"` or `base_url="http://localhost:8000/v1"`.

### Engine Resolution Precedence (`resolve_engine`)

When executing a turn for an `Agent`, the `Runner` resolves the LLM engine in this exact order:

1. **Agent-level engine (`agent.engine`):** If an explicit `BaseEngine` instance was passed to `Agent(engine=...)`, it is always used.
2. **Explicit Agent provider (`agent.provider`):** Resolved when `agent.provider` is set (`gemini`/`google`, `anthropic`/`claude`, `openai`, `grok`/`xai`, `ollama`/`gemma`, `deepseek`, or `groq`).
3. **Custom Agent endpoint (`agent.base_url`):** If `agent.base_url` is set without an explicit `provider`, routes to `OpenAIEngine` (matching the official provider preset if the URL matches an official endpoint, or requiring an explicit `api_key` unless targeting `localhost`).
4. **Explicit Runner engine (`Runner(engine=...)`):** If a custom `engine` was explicitly passed to `Runner`, it takes precedence over model-name prefix heuristics.
5. **Agent model prefix heuristics (`agent.model`):** Auto-detects `claude-*`, `gpt-*`/`o1*`/`o3*`/`chatgpt-*`, `grok*`, `deepseek-*`, and `gemma*`/`llama*`/`mistral*`/`qwen*`/`phi*`.
6. **Default Runner engine / Gemini fallback:** Uses the runner's default `GeminiEngine` (configured with `Runner(api_key=...)`) or falls back to `GeminiEngine(api_key=agent.api_key)`.

> [!IMPORTANT]
> **Host-Scoped Environment Credentials:** For security, environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`) are only sent when connecting to their official HTTPS provider endpoints (`api.openai.com`, `api.anthropic.com`, `api.x.ai`, `api.deepseek.com`, `api.groq.com`). They are never forwarded to custom `base_url` endpoints or local Ollama/vLLM servers. To authenticate against a custom OpenAI-compatible proxy, pass `api_key="..."` explicitly on the `Agent` or `OpenAIEngine` (or `api_key=""` if no authentication is required).

> [!WARNING]
> **Vertex AI is not supported.** `GeminiEngine` calls only the Gemini Developer API (Google AI Studio) with an API key. Application Default Credentials (ADC) and the `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` variables are not used, so setups that relied on them fail with `ProviderError("Gemini API client is not configured. ...")`. Use a Gemini API key, or provide your own engine through `Runner(engine=...)`.
>
> **`GeminiEngine.client` was removed in v0.4.0.** It held the `google-genai` client, which is no longer a dependency. Assigning a fake to it no longer mocks anything. To call the SDK directly, install `google-genai` and create your own client. To mock Gemini in tests or customize HTTP, pass an `httpx.AsyncClient` with `GeminiEngine(http_client=...)`; it stays owned by you and must be used on one event loop only. See the [CHANGELOG](CHANGELOG.md) for the migration steps.

---

## Agents, Tools & Handoffs

### A. Cloud LLM Agent with Tools

Register synchronous or asynchronous Python functions (including callable class instances and `functools.partial` objects) using `@agent.tool`, the standalone `@tool` decorator, or `agent.add_tool()`:

```python
import asyncio
from chatflow_agent import Agent, Runner

support_agent = Agent(
    name="SupportBot",
    model="gemini-2.5-flash",  # Or provider="openai", model="gpt-4o-mini"
    instructions="You are a helpful customer support agent for a retail store.",
)

@support_agent.tool
def get_order_status(order_id: str, include_items: bool = False) -> dict:
    """Look up shipping and tracking status for an order."""
    return {
        "order_id": order_id,
        "status": "Out for delivery",
        "carrier": "FedEx",
        "eta": "Today before 6:00 PM",
        "include_items": include_items,
    }

async def main() -> None:
    async with Runner(starting_agent=support_agent) as runner:
        response = await runner.run_async(
            session_id="user_session_101",
            user_message="Hi, can you check the status of my order #FDX-8821?",
        )
        print(f"[{response.active_agent_name}]: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### B. Local Offline Agent (Google Gemma via Ollama)

Run 100% locally on your machine with zero API costs and full data privacy using [Ollama](https://ollama.com):

1. Start Ollama with Gemma: `ollama run gemma2:9b`
2. Run the script below (works out of the box with `pip install chatflow-agent`):

```python
import asyncio
from chatflow_agent import Agent, Runner

# Connects to http://localhost:11434/v1 with no API key required
local_agent = Agent(
    name="LocalAnalyst",
    provider="ollama",
    model="gemma2:9b",
    instructions="You analyze confidential financial reports locally.",
)

@local_agent.tool
def calculate_vat(subtotal: float, rate_percentage: float = 21.0) -> dict:
    """Calculate VAT tax and total amount."""
    vat = subtotal * (rate_percentage / 100.0)
    return {"subtotal": subtotal, "vat": round(vat, 2), "total": round(subtotal + vat, 2)}

async def main() -> None:
    runner = Runner(starting_agent=local_agent)
    response = await runner.run_async(
        session_id="local_user",
        user_message="Calculate VAT for a $450 invoice.",
    )
    print(f"[{response.active_agent_name}]: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### C. Multi-Agent Swarm with Autonomous Handoffs

Agents can delegate conversations to peer specialists either through the `handoffs=[...]` declaration (which synthesizes `transfer_to_<agent>` tools automatically) or by returning a `Handoff` object with `context_updates` from a custom tool:

```python
import asyncio
from chatflow_agent import Agent, Handoff, Runner

tech_agent = Agent(
    name="TechSupport",
    model="gemini-2.5-flash",
    instructions="You diagnose hardware and software issues.",
)

@tech_agent.tool
def run_diagnostics(device_id: str) -> str:
    """Checks device telemetry."""
    return f"Device {device_id}: All sensors nominal. Firmware v2.1 up to date."

billing_agent = Agent(
    name="Billing",
    model="gemini-2.5-flash",
    instructions="You handle invoices, subscriptions, and refund requests.",
)

concierge = Agent(
    name="Reception",
    model="gemini-2.5-flash",
    instructions="Greet customers and transfer to TechSupport or Billing as required.",
    handoffs=[tech_agent, billing_agent],
)

# Optional: custom tool returning an explicit Handoff with metadata updates
@concierge.tool
def escalate_vip_billing(account_id: str) -> Handoff:
    """Escalate a VIP customer to Billing and attach their account ID to session metadata."""
    return Handoff(
        target_agent_name="Billing",
        reason="VIP billing inquiry",
        context_updates={"vip_account_id": account_id},
    )

async def main() -> None:
    async with Runner(starting_agent=concierge) as runner:
        resp = await runner.run_async(
            session_id="client_77",
            user_message="My device DEV-42 is blinking red. Can you run diagnostics?",
        )
        print(f"Active Agent: {resp.active_agent_name}")  # Output: TechSupport
        print(f"Response: {resp.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

> [!NOTE]
> * **Dynamic Instructions:** `Agent(instructions=...)` accepts either a `str` or a zero-argument callable `Callable[[], str]` evaluated before every LLM call.
> * **Unique Agent Names:** Every distinct `Agent` instance in a swarm (or registered later via `agent.add_handoff()` / `runner.register_agent()`) must have a unique, non-empty `name`. Registering two different `Agent` instances with the same `name` raises `ValueError` immediately.

### Tool Parameter Schemas & `ToolSchemaError`

`Tool` inspects Python type annotations and builds provider-compatible schemas automatically:

* **Primitives:** `str`, `int`, `float`, `bool`
* **Containers:** `list[T]`, `tuple[T, ...]`, `set[T]`, `frozenset[T]`, `Sequence[T]` (mapped to `ARRAY` with `items` schema), `dict[K, V]`, `Mapping[K, V]` (mapped to `OBJECT`)
* **Nullability & Unions:** `Optional[T]` / `T | None` preserves `nullable: True`; `int | float` widens to `NUMBER`; multi-type unions (`int | str`) emit `anyOf`.
* **Advanced Annotations:** Supports `typing.Annotated[T, ...]`, postponed annotations (`from __future__ import annotations`), and variadic `*args` / `**kwargs` (excluded from the schema).
* **Fail-Fast Validation (`ToolSchemaError`):** If a union or optional annotation contains a custom type that has no concrete JSON schema mapping (e.g., `CustomClass | None`), tool registration raises `ToolSchemaError` immediately instead of silently degrading to `STRING`.

### Runner Limits & Public Exceptions

| Parameter / Exception | Default / Base | Description |
| :--- | :--- | :--- |
| `max_tool_iterations` (`Runner`) | `10` | Maximum tool-execution loops allowed in a single turn. |
| `max_handoffs_per_turn` (`Runner`) | `5` | Maximum agent-to-agent transfers allowed within a single turn; when exceeded, records an error `ToolResult` (`is_error=True`) and skips remaining tool calls in that batch. |
| `turn_timeout` (`run_async`) | `None` | Optional per-turn execution limit in seconds (starts after acquiring the session lock). |
| `ChatFlowError` | `Exception` | Base class for all custom exceptions in `chatflow_agent`. |
| `ProviderError` | `ChatFlowError` | Raised on LLM API errors, blocked responses, or malformed payloads (error excerpts capped at 1,000 chars). |
| `ToolSchemaError` | `ChatFlowError`, `TypeError` | Raised at tool definition time when a union member cannot be mapped to a JSON schema type. |
| `ToolExecutionError` | `ChatFlowError` | Raised when a tool call fails directly via `Tool.execute_async()`. |
| `ToolArgumentValidationError` | `ToolExecutionError` | Raised when LLM-provided arguments fail schema validation prior to tool execution. |
| `AgentHandoffError` | `ChatFlowError` | Raised when a handoff targets an unknown or unauthorized agent (not declared in `current_agent.handoffs`). |
| `ReentrantTurnError` | `ChatFlowError`, `RuntimeError` | Raised when a turn attempts to start another turn for the same session inline. |
| `SessionDeletedError` | `ChatFlowError`, `RuntimeError` | Raised when a running turn's session is deleted concurrently. |
| `DependencyError` | `ChatFlowError` | Raised when an optional provider or channel package is not installed. |

---

## Sessions, Persistence & Concurrency

`chatflow-agent` provides zero-dependency session stores that preserve conversation history, the active specialist agent, and session metadata across turns and process restarts.

### In-Memory Storage (`SessionStore`)

Ideal for local scripting, unit tests, and ephemeral sessions:

```python
from chatflow_agent import Runner, SessionStore

runner = Runner(
    starting_agent=receptionist,
    session_store=SessionStore(default_max_turns=40),
)
```

### Persistent & In-Memory SQLite Storage (`SQLiteSessionStore`)

Built on Python's standard `sqlite3` module with Write-Ahead Logging (`WAL`), atomic transactions, sliding-window turn pruning, and persistent webhook message deduplication (`dedup_ttl_hours=192`):

```python
from chatflow_agent import Runner, SQLiteSessionStore

# File-backed SQLite store (persists across server restarts)
with SQLiteSessionStore(db_path="sessions.db", default_max_turns=30) as session_store:
    runner = Runner(starting_agent=receptionist, session_store=session_store)
    # ... run your application ...

# Or shared in-memory SQLite store for fast, isolated integration tests:
memory_sqlite = SQLiteSessionStore(":memory:")
try:
    runner = Runner(starting_agent=receptionist, session_store=memory_sqlite)
finally:
    memory_sqlite.close()
```

**Key Persistence Features:**

* **Autonomous Handoff Survival:** When a user is transferred to `Billing`, the active agent name and `metadata` are stored in SQLite. After a server restart, subsequent messages continue directly with `Billing`.
* **Sliding Window FIFO Pruning:** Retains the latest `default_max_turns` complete user turns while pruning older messages so LLM context windows stay bounded.
* **Explicit Lifecycle Management:** Use `with SQLiteSessionStore(...) as store:` or call `store.close()` on shutdown to release shared `:memory:` connections and shut down the background worker pool cleanly.

### Production Resilience & Concurrency

`chatflow-agent` includes built-in safeguards engineered for real-world messaging traffic:

* **Per-Session Concurrency Lock (`asyncio.Lock`):** When a user sends multiple messages in rapid succession (e.g., three WhatsApp voice transcriptions or quick texts in 2 seconds), turns of the same session run one at a time, preventing race conditions, overlapping tool executions, or corrupted conversation histories. They run in the order they start waiting for the session (`asyncio.Lock` is fair); that is the order in which the channel starts the turns, which is usually, but not necessarily, the order the messages were sent. With `SQLiteSessionStore`, each queued turn reloads the history, active agent and metadata committed by the previous turn before it starts.
  * **Scope:** the guarantee holds within one process, one store instance and one running event loop (for example, one Uvicorn worker, a Telegram bot, or successive `Runner.run()` calls). Turns for the same session coming from several store instances on the same file, several processes, or several threads with their own event loops are **not** serialized.
  * **Turns started from inside a turn:** code started by a turn (a tool, or a task or thread the tool starts) cannot start another turn of the same session while that turn is running: `runner.run_async()` / `runner.run()` raise `ReentrantTurnError` (a `RuntimeError`) instead of waiting for a lock the current turn may be waiting on. To queue a follow-up turn for the same session, call `runner.spawn_turn(session_id, message)`: it returns an `asyncio.Task` that runs after the current turn ends. Never await that task inside the turn that spawned it; await it (or read its exception) afterwards. Turns of other sessions can be awaited from a tool normally.
  * **Deleting a session:** `delete()` and `clear()` do not wait for running turns. A turn running when its session is deleted fails with `SessionDeletedError` and leaves nothing behind; a turn waiting for that session starts a new, empty one. Deleting erases a session's data; it does not block its id, so a new message creates the session again.
  * **SQLite and the event loop:** with a file-backed `SQLiteSessionStore`, session reads and writes made by the `Runner`, the WhatsApp webhook and Telegram's `/reset` run in a pool of 4 worker threads owned by the store, so a slow disk or a database locked by another writer does not stop other requests, sessions or timeouts. A SQLite call that has started cannot be interrupted: a cancelled or timed-out turn waits for its running write, rolls back, and only then raises (the timeout is detected on time but reported once the rollback is stored, up to `busy_timeout`, 10 s, per write while the database is locked). The WhatsApp webhook still acknowledges after its deduplication write, and answers HTTP 500 if the lock outlasts `busy_timeout` (Meta then retries). `SessionStore`, `:memory:` SQLite stores, subclasses and custom stores keep running on the event loop. This does not make one SQLite file safe for several workers or processes.
  * **Turn time limit:** `await runner.run_async(session_id, message, turn_timeout=30)` limits the turn itself (keyword-only, seconds). The limit starts once the turn holds the session, so time spent waiting behind earlier turns of the same session does not count. When it expires the turn is cancelled, the session is restored to its state before the turn, and `TimeoutError` is raised. It must be a positive finite number; the default, `None`, sets no limit.
  * **Synchronous tools after a timeout:** a synchronous (`def`) tool runs in a worker thread, and Python cannot stop a callable that has already started. The timeout and conversational rollback are still reported without waiting for that callable indefinitely, but the session remains fenced until the callable really returns or raises. Later turns of that session do not start while it is running; for channels, this residual-work wait counts toward `queue_timeout_seconds`. Other sessions continue normally. If the callable never finishes, the affected session remains fenced (bounded channel requests receive their configured queue-timeout fallback); use timeouts in the tool's own I/O and design side effects to be idempotent. Async tools remain cancellable normally.
  * **WhatsApp:** each message has two separate limits. `queue_timeout_seconds` (default 120) bounds the wait behind earlier turns of the same session: if it expires, the user receives `fallback_message` and the turn is never run (the message is not added to the history and, as before, a Meta retry of the same `wamid` is discarded). `llm_timeout_seconds` (default 120) bounds the turn itself and starts once the turn holds the session: when it expires the turn is cancelled and rolled back and the user receives `fallback_message`. A reply can therefore take up to `queue_timeout_seconds + llm_timeout_seconds`. The logs tell the cases apart: `Session busy ...` (queue limit), `Turn timed out ...` (turn limit), `Dispatch backstop expired ...` (a `dispatch_async()` override that never reached `Runner.run_async()`), and `TimeoutError raised while dispatching ...` (a `TimeoutError` from the provider or other code, not a channel limit). `llm_timeout_seconds=None` or `inf` sets no turn limit.

* **Meta Anti-500 Error Shield:** Meta's webhook infrastructure retries delivery aggressively if your endpoint returns HTTP 500. `WhatsAppChannel` catches any upstream LLM outages or rate limits, returns **HTTP 200** to Meta to prevent retry storms, dispatches the configured `fallback_message` to the user, and logs clean diagnostic details.

* **Unsupported Media & Ignored Events Handling:** Audio voice notes, photos, and PDF files are safely intercepted with `unsupported_media_message`, while WhatsApp reactions (`reaction`) and system notifications (`system`) are ignored cleanly with HTTP 200 without invoking the `Runner` or sending outbound messages.

* **Turn Failure Rollback:** If an LLM request times out, rate limits, or crashes mid-turn, ChatFlow automatically rolls back dialogue history to the last consistent turn in memory and SQLite (note: rollback reverts conversational history; external side effects of already executed tools are not rolled back), ensuring clean context without orphaned partial turns. See [Tool Idempotency & Side-Effect Handling](#tool-idempotency--side-effect-handling) for how to design tools that are safe to retry.
  * With `SQLiteSessionStore`, the rollback restores the history, active agent and metadata the turn started from, including messages pruned by the sliding window during the turn. If anything else changed the session during the turn (another context updating its metadata or agent, a reset, a delete, or a writer outside the supported scope above), only the failed turn's own changes are reverted, other writers' messages, agent and metadata are kept, and a warning is logged.

---

## Tool Idempotency & Side-Effect Handling

Tools that interact with external systems (process payments, create database records, send emails or messages, call external APIs, create remote resources, or trigger persistent business operations) may be executed again after a failed turn or a user retry. This section explains how to design them so a repeated call does not repeat the real-world effect.

### Conversational Rollback vs. External Effects

When a turn fails, ChatFlow restores the following to the last consistent turn, in memory and in SQLite:

* session history;
* the active agent;
* session metadata.

This does **not** undo operations performed outside `chatflow-agent`. A tool may succeed and produce an external side effect, and the same turn may then fail because of a provider or network error. Since the history is rolled back, the retry no longer contains evidence that the tool already executed, so the same business operation may be requested again.

This is expected behavior in distributed systems. It is handled through idempotent tool design, not through automatic deduplication in the `Runner`.

### Idempotent Tool Design

Identify each business operation with a stable business identifier, such as:

```text
order_id
payment_id
request_id
operation_id
```

Use it to build an idempotency key (a business operation ID) that stays the same across retries:

```python
operation_id = f"refund:{order_id}"
```

An idempotency key identifies one business operation. Reuse it for retries of that same operation, with the same semantic parameters (for example, the same order and amount); a different operation needs a different key.

> A tool only receives the arguments the model supplies, so do not let the model generate the key. Use an identifier that already exists in your system.

### Retry-Safe Tool Design

Claim the key atomically, run the side effect, then record the result. `payment_gateway` is a placeholder for your external system.

```python
import json
import sqlite3

DB_PATH = "operations.db"

with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operations (
            operation_id TEXT PRIMARY KEY,
            status       TEXT NOT NULL,   -- 'pending' | 'done'
            result       TEXT
        )
        """
    )


@billing_agent.tool
def create_refund(order_id: str, amount: float) -> dict:
    """Refund an order. Safe to call more than once for the same order."""
    operation_id = f"refund:{order_id}"

    with sqlite3.connect(DB_PATH) as conn:
        try:
            # 1. Claim the key atomically.
            conn.execute(
                "INSERT INTO operations (operation_id, status) VALUES (?, 'pending')",
                (operation_id,),
            )
        except sqlite3.IntegrityError:
            # Already seen: return the stored outcome instead of repeating the effect.
            status, result = conn.execute(
                "SELECT status, result FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if status == "done":
                return {**json.loads(result), "already_processed": True}
            return {"status": "in_progress", "operation_id": operation_id}

    # 2. Execute the side effect, forwarding the key downstream.
    result = payment_gateway.refund(
        order_id=order_id, amount=amount, idempotency_key=operation_id
    )

    # 3. Record the outcome.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE operations SET status = 'done', result = ? WHERE operation_id = ?",
            (json.dumps(result), operation_id),
        )
    return result
```

* Return the stored result when the operation already ran, so the model can answer consistently instead of repeating the action.
* Forward the key to the external system when it supports one.
* **This example does not recover pending operations.** A `pending` row means either that the operation is still running or that execution was interrupted before it completed, and the example cannot tell which. Calling the tool again does not retry a pending operation; it only returns `in_progress`. Your application should reconcile or retry pending operations deliberately, using the same idempotency key.

### Replay Protection

A tool can be called again because a turn is retried after a rollback, the user resends a message, or a channel redelivers a webhook. Idempotent tools make all of these safe: the second call finds the recorded key and returns the stored result.

### External Transaction Tracking

Record what each tool did outside the conversation history (the `operations` table above is a minimal example): the idempotency key, the external system's ID (payment ID, message ID, remote resource ID), the status, and the result returned to the model. A history rollback does not affect this record.

### Recommended Patterns

| Tool type | Recommended approach |
| --- | --- |
| **Payments** | Key on `payment_id` or `order_id` plus the operation; forward the key to the payment provider. |
| **Database records** | Use `UNIQUE` constraints or upserts (`INSERT ... ON CONFLICT`) on a business key. |
| **Emails / messages** | Reserve a pending delivery per `(recipient, message_kind, business_id)`, send it, then confirm the result (see the note below). |
| **External API calls** | Send the provider's idempotency key when available; otherwise look up existing state first. |
| **Remote resources** | Use deterministic names or tags derived from a business ID, and look them up before creating. |
| **Persistent business operations** | Assign an `operation_id` per business action and record it before executing. |

**Emails and messages.** Treat delivery as three separate steps: reserve a pending delivery, send the message, then confirm the result. Recording a "sent" marker *before* sending can lose a message if the process stops before delivery. If delivery succeeds but recording the result fails, a local marker alone cannot guarantee both no duplicates and guaranteed delivery; that requires an external provider that supports idempotency or reconciliation.

---

## Available Channels

### WhatsApp Channel (Meta Cloud API)

Run a production webhook server compatible with the official Meta WhatsApp Business Cloud API (requires `pip install "chatflow-agent[whatsapp]"`):

```python
from chatflow_agent import Agent, Runner, WhatsAppChannel

agent = Agent(name="WhatsAppConcierge", instructions="Answer customer inquiries.")
runner = Runner(starting_agent=agent)

channel = WhatsAppChannel(
    verify_token="my_custom_webhook_secret",  # Verification token configured in Meta App
    access_token="EAA...",                    # Meta Permanent/System User Token
    app_secret="a1b2c3d4...",                 # Meta App Secret (required when verify_signature=True)
    phone_number_id="109876543210987",        # WhatsApp Phone Number ID from Meta Dashboard
    fallback_message="We are experiencing a temporary delay. Please try again shortly.",
    unsupported_media_message="Currently I can only process text messages.",
    llm_timeout_seconds=120.0,
    queue_timeout_seconds=120.0,
)

channel.attach(runner)

if __name__ == "__main__":
    # Exposes GET /webhook (verification), POST /webhook (messages), and GET /health
    channel.run(host="0.0.0.0", port=8000)
```

* **Fail-Closed HMAC Verification (`verify_signature=True` by default):** Validates `X-Hub-Signature-256` using `app_secret` on every `POST /webhook` request before parsing JSON.
* **Payload Guards:** Enforces `max_body_size` (default 1 MB, returning HTTP 413 when exceeded or HTTP 411 when `Content-Length` is missing) and returns HTTP 400 for malformed or non-object JSON payloads.
* **Automatic Message Chunking & Outbound Retries:** Outbound replies exceeding 4,096 characters are split cleanly with `split_message()`, and transient network/rate-limit delivery errors are retried with exponential backoff (`max_outbound_retries=3`).

> **Testing locally?** Use a tunnel like [ngrok](https://ngrok.com) (`ngrok http 8000`) or Cloudflare Tunnels to provide Meta with a public HTTPS URL (`https://your-domain.ngrok-free.app/webhook`).

> [!NOTE]
> **API Documentation in Development:** For security, FastAPI interactive documentation (`/docs`, `/redoc`, and `/openapi.json`) is disabled by default in production. To enable Swagger UI and OpenAPI schemas during local development or debugging, pass `enable_docs=True` when instantiating `WhatsAppChannel`:
> ```python
> channel = WhatsAppChannel(
>     ...,
>     enable_docs=True,  # Enables /docs, /redoc, and /openapi.json for local development
> )
> ```

> [!TIP]
> **Production Rate Limiting & Reverse Proxy:** To protect `/webhook` and `/health` endpoints against flood attacks, denial of service (DoS), and CPU exhaustion from unauthenticated requests without risking dropping legitimate Meta webhook bursts at the application level, always place Uvicorn behind a reverse proxy in production (such as Nginx with `limit_req_zone` or Cloudflare Rate Limiting):
> ```nginx
> limit_req_zone $binary_remote_addr zone=whatsapp_limit:10m rate=10r/s;
> location /webhook {
>     limit_req zone=whatsapp_limit burst=20 nodelay;
>     proxy_pass http://127.0.0.1:8000;
>     proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
> }
> ```

### Telegram Channel

Connect your agent swarm to Telegram with async polling, typing indicators, automatic 4,096-character message splitting, and built-in `/start` and `/reset` commands (requires `pip install "chatflow-agent[telegram]"`):

```python
from chatflow_agent import Agent, Runner, TelegramChannel

agent = Agent(name="TelegramBot", instructions="You assist Telegram users.")
runner = Runner(starting_agent=agent)

channel = TelegramChannel(
    token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",  # From @BotFather
    fallback_message="Sorry, a temporary issue occurred. Please retry in a few moments.",
)

channel.attach(runner)

if __name__ == "__main__":
    # Runs async polling; in group chats, /reset is restricted to group administrators
    channel.run()
```

### Interactive Terminal (CLI)

Test and debug multi-agent swarms interactively in your terminal:

```python
from chatflow_agent import Agent, CLIChannel, Runner

agent = Agent(name="TerminalAssistant", instructions="Answer user questions concisely.")
runner = Runner(starting_agent=agent)

channel = CLIChannel(session_id="dev_test")
channel.attach(runner)

if __name__ == "__main__":
    # Supports /reset to clear session memory, and 'exit' / 'quit' / 'salir' or Ctrl+C to quit
    channel.run()
```

---

## Usage Examples

Ready-to-run reference scripts are available in the [`examples/`](examples/) directory:

* **[`examples/whatsapp_persistent_bot.py`](examples/whatsapp_persistent_bot.py):** Meta Cloud API WhatsApp bot with SQLite persistence, per-session concurrency locks, and a 3-agent handoff hierarchy.
* **[`examples/whatsapp_service.py`](examples/whatsapp_service.py):** Minimal WhatsApp Business Cloud API webhook starter with custom tools and specialist handoffs.
* **[`examples/telegram_persistent_bot.py`](examples/telegram_persistent_bot.py):** Telegram bot with SQLite state persistence across polling restarts.
* **[`examples/telegram_triage.py`](examples/telegram_triage.py):** Multi-agent Telegram triage bot with order lookup and technical support tools.
* **[`examples/webhook_api_service.py`](examples/webhook_api_service.py):** Custom FastAPI REST service exposing `/api/chat`, `/api/sessions/{session_id}`, and `/health` with SQLite persistence.
* **[`examples/local_gemma_cli.py`](examples/local_gemma_cli.py):** 100% offline CLI assistant powered by Google Gemma 2 via Ollama with persistent SQLite sessions.
* **[`examples/multi_provider_swarm.py`](examples/multi_provider_swarm.py):** Multi-provider swarm coordinating Gemini, local Gemma, Grok, and Claude in a single conversation.
* **[`examples/cli_multiagent.py`](examples/cli_multiagent.py):** Interactive CLI swarm demonstrating autonomous handoffs between a receptionist, technical support, and billing agent.

---

## Technical Documentation

* **[Architecture Guide](docs/ARCHITECTURE.md):** Deep dive into the internal execution loop, message schemas, engine adapters, and session persistence.
* **[Migration Guide (0.3.4 → 0.4.0)](docs/MIGRATION_0.4.md):** Upgrade instructions, breaking changes, and new features in `0.4.0`.
* **[Changelog](CHANGELOG.md):** Release history, new features, security advisories, and breaking changes.
* **[Security Policy](SECURITY.md):** Supported versions and vulnerability reporting instructions.
* **[Code of Conduct](CODE_OF_CONDUCT.md):** Community standards and expectations.
* **[Guía en Español](README.es.md):** Complete Spanish documentation (`README.es.md`).

---

<a id="contributing"></a>
## 🤝 Contributing

Contributions are welcome! Whether you are fixing a bug, improving documentation, adding tests, or proposing new channel connectors, your help is appreciated.

If you would like to help improve `chatflow-agent`, check out our open issues:

* 🟢 **[Good First Issues](https://github.com/GustavoBaranda/chatflow-agent/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** — Well-scoped tasks ideal for first-time contributors.
* 🟡 **[Help Wanted](https://github.com/GustavoBaranda/chatflow-agent/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)** — Intermediate and advanced tasks looking for assistance.
* 📋 **[All Open Issues](https://github.com/GustavoBaranda/chatflow-agent/issues)** — Browse all open bug reports, enhancements, and contribution opportunities.

Please read our **[Contributing Guide](CONTRIBUTING.md)** before submitting a Pull Request for setup instructions, code style (`ruff`, `mypy --strict`), and test requirements (`pytest`).

---

<a id="support-the-project"></a>
## 💖 Support the Project

If you find `chatflow-agent` useful for your projects or organization, consider supporting its ongoing open-source development through **[GitHub Sponsors](https://github.com/sponsors/GustavoBaranda)**.

Sponsorship helps maintain the project, improve documentation, expand test coverage, enhance reliability across messaging channels, and support continued feature development.

Sponsorship is completely voluntary and does not affect access to any features, releases, or documentation of the open-source project.

---

## License

Distributed under the **[MIT License](LICENSE)**.

Created and maintained by **[Gustavo Baranda](https://github.com/GustavoBaranda)**, with contributions from the [community](#our-contributors).
