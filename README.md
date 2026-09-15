# chatflow-agent

[![PyPI version](https://img.shields.io/pypi/v/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![Python versions](https://img.shields.io/pypi/pyversions/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Lightweight, async multi-agent framework with native handoffs for real-world channels.**  
> Supports **Google Gemini**, **Google Gemma (Local)**, **xAI Grok**, **OpenAI (GPT-4o)**, and **Anthropic Claude**.  
> Connect autonomous swarms directly to **WhatsApp**, **Telegram**, **Webhooks**, and **CLI**.
>  
> *Framework multiagente asíncrono y liviano con transferencias nativas (handoffs) para canales reales.*  
> *Compatible con **Google Gemini**, **Google Gemma (Local)**, **xAI Grok**, **OpenAI** y **Anthropic Claude**.*  
> *Conecta equipos de agentes autónomos a **WhatsApp**, **Telegram**, **Webhooks** y **Terminal**.*

---

## English Documentation

### Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Supported Providers & API Keys](#supported-providers--api-keys)
3. [Installation](#installation)
4. [Step-by-Step Quickstarts](#step-by-step-quickstarts)
   - [A. Cloud LLM (Gemini / OpenAI / Claude)](#a-cloud-llm-quickstart)
   - [B. 100% Free Local LLM (Google Gemma via Ollama)](#b-local-offline-quickstart-google-gemma)
   - [C. Multi-Agent Swarm with Handoffs](#c-multi-agent-swarm-with-handoffs)
5. [Channel Connectors](#channel-connectors)
   - [WhatsApp (Meta Cloud API)](#whatsapp-channel-meta-cloud-api)
   - [Telegram Bot](#telegram-channel)
   - [Interactive Terminal (CLI)](#interactive-terminal-cli)
6. [Session Memory & Persistence](#session-memory--persistence)
7. [Guía en Español](#guía-completa-en-español)

---

### Overview & Architecture

`chatflow-agent` is designed for developers who want the multi-agent power of OpenAI Swarm without being locked into a single provider, combined with turnkey connectors for messaging platforms like **WhatsApp** and **Telegram**.

```
[ WhatsApp / Telegram / Webhook / CLI ]
                   │
                   ▼
         ┌───────────────────┐
         │      Runner       │ ◄─── Session Memory (per-user history)
         └─────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  ┌──────────────┐    ┌──────────────┐
  │ Triage Agent │───►│  Spec. Agent │ (Autonomous Peer Handoff)
  │ (Gemini/Grok)│    │ (Local Gemma)│
  └──────────────┘    └──────────────┘
         │                   │
         ▼                   ▼
    [@agent.tool]       [@agent.tool]
```

---

### Supported Providers & API Keys

You can configure credentials either via **environment variables** or **directly in code**:

| Provider | Supported Models | Environment Variable | In-Code Configuration |
| :--- | :--- | :--- | :--- |
| **Google Gemini** *(Default)* | `gemini-2.5-flash`, `gemini-2.5-pro` | `GEMINI_API_KEY` | `Agent(provider="gemini", api_key="...")` |
| **Google Gemma / Ollama** | `gemma2:9b`, `gemma2:2b`, `llama3.2`, etc. | *None needed (Local)* | `Agent(provider="ollama", model="gemma2:9b")` |
| **xAI Grok** | `grok-2`, `grok-2-mini` | `XAI_API_KEY` | `Agent(provider="grok", api_key="...")` |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `o1` | `OPENAI_API_KEY` | `Agent(provider="openai", api_key="...")` |
| **Anthropic Claude** | `claude-3-5-sonnet`, `claude-3-5-haiku` | `ANTHROPIC_API_KEY` | `Agent(provider="anthropic", api_key="...")` |

#### How to set environment variables:
* **Windows (PowerShell):** `$env:GEMINI_API_KEY="your-api-key"`
* **Linux / macOS:** `export GEMINI_API_KEY="your-api-key"`
* **Direct in Python (No terminal setup required):**
  ```python
  runner = Runner(starting_agent=my_agent, api_key="your-api-key")
  ```

---

### Installation

Install only what you need:

```bash
# Core framework (Gemini, Gemma, Grok, OpenAI, Claude, CLI)
pip install chatflow-agent

# With WhatsApp Webhook connector (FastAPI + Uvicorn)
pip install "chatflow-agent[whatsapp]"

# With Telegram Bot connector (python-telegram-bot)
pip install "chatflow-agent[telegram]"

# Complete bundle (All channels & dev tools)
pip install "chatflow-agent[all]"
```

---

### Step-by-Step Quickstarts

#### A. Cloud LLM Quickstart

Save as `bot.py` and run with `python bot.py`:

```python
import asyncio
from chatflow_agent import Agent, Runner

# Define your agent
support_agent = Agent(
    name="SupportBot",
    model="gemini-2.5-flash",  # Or provider="openai", model="gpt-4o-mini"
    instructions="You are a helpful customer support agent for a retail store.",
)

# Register business tools using Python decorators and type hints
@support_agent.tool
def get_order_status(order_id: str) -> dict:
    """Look up shipping and tracking status for an order."""
    return {
        "order_id": order_id,
        "status": "Out for delivery",
        "carrier": "FedEx",
        "eta": "Today before 6:00 PM",
    }

async def main():
    # Pass api_key here or rely on GEMINI_API_KEY environment variable
    runner = Runner(starting_agent=support_agent)

    response = await runner.run_async(
        session_id="user_session_101",
        user_message="Hi, can you check the status of my order #FDX-8821?",
    )
    print(f"[{response.active_agent_name}]: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

#### B. Local Offline Quickstart (Google Gemma)

Run 100% locally on your machine with **zero API costs** and **total privacy** using [Ollama](https://ollama.com):

1. Start Ollama with Gemma: `ollama run gemma2:9b`
2. Run this script:

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

async def main():
    runner = Runner(starting_agent=local_agent)
    response = await runner.run_async(
        session_id="local_user",
        user_message="Calculate VAT for a $450 invoice.",
    )
    print(f"[{response.active_agent_name}]: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

#### C. Multi-Agent Swarm with Handoffs

Agents can autonomously delegate tasks to specialists:

```python
import asyncio
from chatflow_agent import Agent, Runner

# 1. Specialist: Technical Support
tech_agent = Agent(
    name="TechSupport",
    model="gemini-2.5-flash",
    instructions="You diagnose hardware and software issues.",
)

@tech_agent.tool
def run_diagnostics(device_id: str) -> str:
    """Checks device telemetry."""
    return f"Device {device_id}: All sensors nominal. Firmware v2.1 up to date."

# 2. Specialist: Billing & Invoices
billing_agent = Agent(
    name="Billing",
    model="gemini-2.5-flash",
    instructions="You handle invoices, subscriptions, and refund requests.",
)

# 3. Receptionist (Frontline) with handoffs to specialists
concierge = Agent(
    name="Reception",
    model="gemini-2.5-flash",
    instructions="Greet customers and transfer to TechSupport or Billing as required.",
    handoffs=[tech_agent, billing_agent],  # Swarm handoff capability
)

async def main():
    runner = Runner(starting_agent=concierge)

    # The model detects it is a technical query and hands off to TechSupport automatically
    resp = await runner.run_async(
        session_id="client_77",
        user_message="My device DEV-42 is blinking red. Can you run diagnostics?",
    )
    print(f"Active Agent: {resp.active_agent_name}")  # Output: TechSupport
    print(f"Response: {resp.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Channel Connectors

#### WhatsApp Channel (Meta Cloud API)

Run a production webhook server compatible with Meta WhatsApp Business Cloud API:

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import WhatsAppChannel

agent = Agent(name="WhatsAppConcierge", instructions="Answer customer inquiries.")
runner = Runner(starting_agent=agent)

channel = WhatsAppChannel(
    runner=runner,
    verify_token="my_custom_webhook_secret",  # Verification token configured in Meta App
    access_token="EAA...",                    # Meta Permanent/System User Token
    phone_number_id="109876543210987",        # WhatsApp Phone Number ID from Meta Dashboard
)

if __name__ == "__main__":
    # Exposes GET /webhook (verification handshake) and POST /webhook (incoming messages)
    channel.run(host="0.0.0.0", port=8000)
```

> **Testing locally?** Use a tunnel like [ngrok](https://ngrok.com) (`ngrok http 8000`) or Cloudflare Tunnels to provide Meta with a public HTTPS URL (`https://your-domain.ngrok-free.app/webhook`).

---

#### Telegram Channel

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import TelegramChannel

agent = Agent(name="TelegramBot", instructions="You assist Telegram users.")
runner = Runner(starting_agent=agent)

channel = TelegramChannel(
    runner=runner,
    bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",  # From @BotFather
)

if __name__ == "__main__":
    # Runs async polling; automatically handles /start, /reset, and typing indicators
    channel.run()
```

---

#### Interactive Terminal (CLI)

Test swarms in your terminal with colored chat output:

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import CLIChannel

agent = Agent(name="TerminalAssistant", instructions="Answer user questions concisely.")
runner = Runner(starting_agent=agent)

channel = CLIChannel(runner=runner, session_id="dev_test")
channel.run()
```

---

### Session Memory & Persistence

Every conversation turn is tracked by `SessionContext`:
* **Sliding Window Pruning:** Prevents LLM context overflow by keeping the last `max_messages` (default: 50).
* **Active Agent State:** If a handoff occurs (e.g., from `Reception` to `Billing`), subsequent messages from that user automatically continue talking to `Billing`.
* **Resetting:** Call `session.clear()` or send `/reset` on Telegram to start fresh.

---

## Guía Completa en Español

### Modelos y Proveedores Compatibles

`chatflow-agent` te permite elegir libremente entre servicios en la nube o modelos que corren en tu propia máquina:

| Proveedor | Modelos Ejemplo | Configuración | Costo |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | `gemini-2.5-flash`, `gemini-2.5-pro` | Variable `GEMINI_API_KEY` o en código | Gratis (tier estándar) |
| **Google Gemma / Ollama** | `gemma2:9b`, `llama3.2`, `mistral` | `provider="ollama"` (sin clave) | **100% Gratis y Local** |
| **xAI Grok** | `grok-2` | Variable `XAI_API_KEY` o en código | Según uso en xAI |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | Variable `OPENAI_API_KEY` o en código | Según uso en OpenAI |
| **Anthropic Claude** | `claude-3-5-sonnet`, `claude-3-5-haiku` | Variable `ANTHROPIC_API_KEY` o en código | Según uso en Anthropic |

---

### Ejemplo Rápido: De 0 a Funcionando en 2 Minutos

Crea un archivo `mi_asistente.py`:

```python
import asyncio
from chatflow_agent import Agent, Runner

# 1. Definir el agente con sus herramientas
soporte = Agent(
    name="SoporteClientes",
    model="gemini-2.5-flash",
    instructions="Eres un asistente cordial de atención al cliente.",
)

@soporte.tool
def consultar_stock(articulo: str) -> dict:
    """Consulta la disponibilidad y precio de un artículo en inventario."""
    catalogo = {
        "laptop": {"stock": 5, "precio": "$1,200"},
        "teclado": {"stock": 18, "precio": "$45"},
        "mouse": {"stock": 30, "precio": "$25"},
    }
    return catalogo.get(articulo.lower(), {"stock": 0, "precio": "No disponible"})

# 2. Ejecutar la conversación
async def main():
    # Puedes pasar tu api_key aquí directamente si no usas variables de terminal:
    # runner = Runner(starting_agent=soporte, api_key="AIzaSy...")
    runner = Runner(starting_agent=soporte)

    respuesta = await runner.run_async(
        session_id="cliente_whatsapp_1",
        user_message="Hola, ¿tienen stock de la laptop y a cuánto está?",
    )
    print(f"[{respuesta.active_agent_name}]: {respuesta.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

Para ejecutar:
```bash
# Si tienes la clave como variable:
python mi_asistente.py
```

---

### Ejemplo: Servidor de WhatsApp en Producción

Crea `servicio_whatsapp.py`:

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import WhatsAppChannel

agente_ventas = Agent(
    name="VentasWhatsApp",
    model="gemini-2.5-flash",
    instructions="Ayudas a los clientes a cotizar y realizar compras.",
)

runner = Runner(starting_agent=agente_ventas)

# Conector oficial para Meta Cloud API
servidor_whatsapp = WhatsAppChannel(
    runner=runner,
    verify_token="tu_token_verificacion_meta", # El token que configuras en Meta Developers
    access_token="EAA...",                     # Token de acceso de sistema de Meta
    phone_number_id="102938475610293",         # ID de número telefónico de WhatsApp
)

if __name__ == "__main__":
    # Levanta el servidor en http://localhost:8000/webhook
    servidor_whatsapp.run(host="0.0.0.0", port=8000)
```

---

## License
Distributed under the **MIT License**.
