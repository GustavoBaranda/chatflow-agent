# chatflow-agent

[![CI](https://github.com/GustavoBaranda/chatflow-agent/actions/workflows/test.yml/badge.svg)](https://github.com/GustavoBaranda/chatflow-agent/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

> **Build autonomous multi-agent swarms for WhatsApp, Telegram, and Webhooks in seconds without heavy dependency chains.**  
> *Construye equipos de agentes autonomos para WhatsApp, Telegram y Webhooks en segundos sin lidiar con dependencias monstruosas.*

```bash
pip install chatflow-agent
```

[English Documentation](#english-documentation) | [Guia en Espanol](#guia-completa-en-espanol) | [PyPI Package](https://pypi.org/project/chatflow-agent/) | [Report Issue](https://github.com/GustavoBaranda/chatflow-agent/issues)

---

## Key Pillars (Why chatflow-agent?)



* **Ultra-Lightweight Core:** Only 3 runtime dependencies (`google-genai`, `pydantic`, `httpx`). A tiny ~72 KB package footprint that installs in seconds on Docker, serverless, and cloud VPS environments.

* **Omnichannel Native:** Production-ready connectors for **WhatsApp (Official Meta Cloud API)**, **Telegram**, **Webhooks (FastAPI)**, and interactive **CLI**.

* **Multi-Provider Freedom:** Switch between **Google Gemini**, **Google Gemma (100% free local via Ollama)**, **xAI Grok**, **OpenAI (GPT-4o)**, and **Anthropic Claude** with a single parameter. Zero vendor lock-in.

* **Production-Hardened Shields:** Native per-session concurrency locking (`asyncio.Lock`) prevents out-of-order race conditions from fast typers. Webhook anti-500 shield prevents Meta retry bombardment storms during LLM downtime.

* **Autonomous Peer Handoffs:** Agents delegate conversations dynamically based on customer intent, without rigid state machines or complex graph builders.



---



## Quickstart in 10 Seconds



```python

import asyncio

from chatflow_agent import Agent, Runner



# 1. Define specialist agents

tech_agent = Agent(name="TechSupport", model="gemini-2.5-flash", instructions="Help with bugs.")

billing_agent = Agent(name="Billing", model="gemini-2.5-flash", instructions="Help with invoices.")



# 2. Receptionist agent with autonomous peer handoffs

receptionist = Agent(

    name="Receptionist",

    model="gemini-2.5-flash",

    instructions="Greet the customer and route them to TechSupport or Billing.",

    handoffs=[tech_agent, billing_agent],

)



async def main():

    runner = Runner(starting_agent=receptionist)

    result = await runner.run_async(

        session_id="user_whatsapp_1",

        user_message="Hi! I need help with an invoice error on my account.",

    )

    print(f"[{result.active_agent_name}]: {result.content}")

    # Output: [Billing]: I would be glad to help check your invoice. Could you share your account ID?



if __name__ == "__main__":

    asyncio.run(main())

```



---



## English Documentation



### Table of Contents

1. [Overview & Architecture](#overview--architecture)

2. [API Keys & Configuration](#api-keys--configuration)

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

7. [Guía Completa en Español](#guía-completa-en-español)



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



### API Keys & Configuration



You can provide API keys using any of the following 3 methods:



#### 1. Direct in Python Code (Easiest)

Pass your API key directly when instantiating the `Runner` or `Agent`:

```python

# Pass to Runner (used by all agents with that provider)

runner = Runner(starting_agent=my_agent, api_key="AIzaSyYourGeminiKey")



# Or pass directly to a specific Agent:

grok_agent = Agent(

    name="GrokSpecialist",

    provider="grok",

    api_key="xai-your-key-here",

    instructions="..."

)

```



#### 2. Using a `.env` File

Create a `.env` file in your project root:

```env

# Google Gemini (Get free key at https://aistudio.google.com/)

GEMINI_API_KEY="AIzaSy..."



# OpenAI (https://platform.openai.com/api-keys)

OPENAI_API_KEY="sk-..."



# xAI Grok (https://console.x.ai/)

XAI_API_KEY="xai-..."



# Anthropic Claude (https://console.anthropic.com/)

ANTHROPIC_API_KEY="sk-ant-..."

```

Then in your script:

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



#### Provider Credential Reference Table



| Provider | Where to get Key | Environment Variable | In-Code Parameter | Free Tier Available? |

| :--- | :--- | :--- | :--- | :--- |

| **Google Gemini** | [Google AI Studio](https://aistudio.google.com/app/apikey) | `GEMINI_API_KEY` | `api_key="..."` | **Yes (Generous free tier)** |

| **Google Gemma (Local)** | [Ollama](https://ollama.com) | *None needed* | `provider="ollama"` | **100% Free & Offline** |

| **xAI Grok** | [xAI Console](https://console.x.ai/) | `XAI_API_KEY` | `api_key="..."` | Pay-as-you-go |

| **OpenAI** | [OpenAI Platform](https://platform.openai.com/) | `OPENAI_API_KEY` | `api_key="..."` | Pay-as-you-go |

| **Anthropic Claude** | [Anthropic Console](https://console.anthropic.com/) | `ANTHROPIC_API_KEY` | `api_key="..."` | Pay-as-you-go |



---



### Installation



Install only what you need:



```bash

# Core framework (Gemini, Gemma, Grok, OpenAI, Claude, CLI)

pip install chatflow-agent
`


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

    # Pass api_key directly or set GEMINI_API_KEY in environment/.env

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

    verify_token="my_custom_webhook_secret",  # Verification token configured in Meta App

    access_token="EAA...",                    # Meta Permanent/System User Token

    phone_number_id="109876543210987",        # WhatsApp Phone Number ID from Meta Dashboard

    fallback_message="We are experiencing a temporary delay. Please try again shortly.",

    unsupported_media_message="Currently I can only process text messages.",

)

channel.attach(runner)



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

    token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",  # From @BotFather

    fallback_message="Sorry, a temporary issue occurred. Please retry in a few moments.",

)

channel.attach(runner)



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



channel = CLIChannel(session_id="dev_test")

channel.attach(runner)

channel.run()

```



---



### Production Resilience & Concurrency



ChatFlow includes built-in safeguards engineered specifically for real-world messaging traffic:



* **Per-Session Concurrency Lock (`asyncio.Lock`):** When a user sends multiple messages in rapid succession (e.g., three WhatsApp voice transcriptions or quick texts in 2 seconds), an async lock guarantees FIFO execution. Turns are processed in strict sequential order, preventing race conditions, overlapping tool executions, or corrupted conversation histories.

* **Meta Anti-500 Error Shield:** Meta's webhook infrastructure retries delivery aggressively if your endpoint returns HTTP 500. `WhatsAppChannel` catches any upstream LLM outages or rate limits, returns **HTTP 200** to Meta to prevent retry storms, dispatches the configured `fallback_message` to the user, and logs clean diagnostic details.

* **Unsupported Media Handling:** Audio voice notes, photos, and PDF files are safely intercepted with `unsupported_media_message` without throwing unhandled exceptions or disrupting ongoing chat sessions.



---



### Session Memory & Persistence



ChatFlow provides native, zero-dependency persistence to ensure conversations and active specialist agents survive server restarts, worker redeployments, and crashes.



#### In-Memory Storage (`SessionStore`)

Ideal for testing, local scripting, and ephemeral sessions:

```python

from chatflow_agent import Runner, SessionStore



runner = Runner(starting_agent=triage_agent, session_store=SessionStore(default_max_turns=20))

```



#### Persistent SQLite Storage (`SQLiteSessionStore`)

Built on Python's standard `sqlite3` library with zero additional external packages. Safely manages Windows file locks and atomic transactions:

```python

from chatflow_agent import Runner, SQLiteSessionStore



# History and active specialist agent state persist across server reboots

session_store = SQLiteSessionStore(db_path="sessions.db", default_max_turns=30)

runner = Runner(starting_agent=triage_agent, session_store=session_store)

```



**Key Persistence Features:**

* **Autonomous Handoff Survival:** If a customer is handed off to `Billing Specialist`, the active agent pointer is recorded in SQLite. When the server reboots or workers recycle, subsequent customer messages continue directly with `Billing Specialist` instead of resetting to the triage agent.

* **Sliding Window FIFO Pruning:** Maintains the latest conversation turns while safely discarding older messages in SQLite to avoid LLM context limits.

* **Omnichannel Compatibility:** Works seamlessly across WhatsApp, Telegram, Webhook APIs, and CLI.



---



### Production Starter Templates



Ready-to-run reference implementations are available in the [`examples/`](examples/) directory:



* **[examples/whatsapp_persistent_bot.py](examples/whatsapp_persistent_bot.py):** Meta Cloud API WhatsApp bot with SQLite persistence, concurrency locks, and 3-agent handoff hierarchy.

* **[examples/telegram_persistent_bot.py](examples/telegram_persistent_bot.py):** Telegram bot with state persistence across polling restarts.

* **[examples/webhook_api_service.py](examples/webhook_api_service.py):** Async REST API service with CORS, `/api/chat` and `/api/sessions/{session_id}` endpoints for React, Next.js, and mobile frontends.

* **[examples/local_gemma_cli.py](examples/local_gemma_cli.py):** 100% offline, free CLI assistant powered by Google Gemma 2 via Ollama with persistent terminal sessions.

* **[examples/multi_provider_swarm.py](examples/multi_provider_swarm.py):** Multi-provider swarm coordinating Gemini, Gemma, Grok, and Claude in a single conversational flow.



---



## Guia Completa en Espanol



### Tabla de Contenidos

1. [Vision General y Arquitectura](#vision-general-y-arquitectura)

2. [Configuracion de API Keys y Credenciales](#configuracion-de-api-keys-y-credenciales)

3. [Instalacion](#instalacion-1)

4. [Guias Paso a Paso (Quickstarts)](#guias-paso-a-paso-quickstarts)

   - [A. Agente Cloud con Herramientas (Tools)](#a-agente-cloud-con-herramientas)

   - [B. Agente 100% Offline y Gratuito (Google Gemma via Ollama)](#b-agente-100-offline-y-gratuito-google-gemma)

   - [C. Swarm Multi-Agente con Handoffs Autonomos](#c-swarm-multi-agente-con-handoffs-autonomos)

5. [Conectores de Canales Oficiales](#conectores-de-canales-oficiales)

   - [WhatsApp (Meta Cloud API)](#canal-de-whatsapp-meta-cloud-api)

   - [Telegram Bot](#canal-de-telegram)

   - [Terminal Interactiva (CLI)](#canal-de-terminal-interactiva-cli)

6. [Resiliencia y Concurrencia en Produccion](#resiliencia-y-concurrencia-en-produccion-1)

7. [Memoria de Sesiones y Persistencia en SQLite](#memoria-de-sesiones-y-persistencia-en-sqlite-1)

8. [Plantillas de Produccion Listas para Usar](#plantillas-de-produccion-listas-para-usar-1)



---



### Vision General y Arquitectura



`chatflow-agent` fue disenado para desarrolladores que buscan la potencia multi-agente de OpenAI Swarm sin quedar atados a un unico proveedor, combinado con conectores oficiales listos para produccion en plataformas de mensajeria como **WhatsApp** y **Telegram**.



```

[ WhatsApp / Telegram / Webhook / CLI ]

                   │

                   ▼

         ┌───────────────────┐

         │      Runner       │ ◄─── Memoria de Sesion (Historial por usuario)

         └─────────┬─────────┘

                   │

         ┌─────────┴─────────┐

         ▼                   ▼

  ┌──────────────┐    ┌──────────────┐

  │ Agente Triage│───►│Especialista  │ (Handoff Autonomo entre Pares)

  │ (Gemini/Grok)│    │(Gemma Local) │

  └──────────────┘    └──────────────┘

         │                   │

         ▼                   ▼

    [@agent.tool]       [@agent.tool]

```



---



### Configuracion de API Keys y Credenciales



Puedes configurar tus credenciales utilizando cualquiera de estos **3 metodos**:



#### Opcion 1: Directo en tu Codigo Python (La mas rapida)

Pasa tu clave al instanciar el `Runner` o el `Agent`:

```python

# Pasándola al Runner (la usan todos los agentes de ese proveedor):

runner = Runner(starting_agent=mi_agente, api_key="AIzaSyTuClaveDeGemini")



# O a un agente especifico:

agente_grok = Agent(

    name="GrokSpecialist",

    provider="grok",

    api_key="xai-tu-clave-aqui",

    instructions="Eres un analista de datos.",

)

```



#### Opcion 2: Usando un Archivo `.env` (Recomendado en Produccion)

Crea un archivo `.env` en la raiz de tu proyecto:

```env

# Google Gemini (Obtén tu clave gratuita en https://aistudio.google.com/)

GEMINI_API_KEY="AIzaSy..."



# OpenAI (https://platform.openai.com/api-keys)

OPENAI_API_KEY="sk-..."



# xAI Grok (https://console.x.ai/)

XAI_API_KEY="xai-..."



# Anthropic Claude (https://console.anthropic.com/)

ANTHROPIC_API_KEY="sk-ant-..."

```



#### Opcion 3: Variables de Entorno en la Terminal

```bash

# En Linux / macOS

export GEMINI_API_KEY="AIzaSy..."



# En Windows PowerShell

$env:GEMINI_API_KEY="AIzaSy..."

```



#### Tabla Comparativa de Proveedores y Variables



| Proveedor | Parametro `provider` | Variable de Entorno | Modelo por Defecto | Nivel de Costo |

| :--- | :--- | :--- | :--- | :--- |

| **Google Gemini** | `"gemini"` *(por defecto)* | `GEMINI_API_KEY` | `gemini-2.5-flash` | Nivel gratuito generoso en AI Studio |

| **Google Gemma (Local)**| `"ollama"` | *No requiere clave* | `gemma2:9b` o `gemma2:2b`| **100% Gratis y Offline** |

| **xAI Grok** | `"grok"` | `XAI_API_KEY` | `grok-2` | Pago por uso |

| **OpenAI** | `"openai"` | `OPENAI_API_KEY` | `gpt-4o-mini` | Pago por uso |

| **Anthropic** | `"anthropic"` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet` | Pago por uso |



---



### Instalacion



Instala unicamente los componentes que tu proyecto requiera:



```bash

# Nucleo base (Gemini, Gemma, Grok, OpenAI, Claude, CLI)

pip install chatflow-agent
`


# Con conector de Webhooks para WhatsApp (FastAPI + Uvicorn)

pip install "chatflow-agent[whatsapp]"



# Con conector de Bot para Telegram (python-telegram-bot)

pip install "chatflow-agent[telegram]"



# Paquete completo (Todos los canales y herramientas de desarrollo)

pip install "chatflow-agent[all]"

```



---



### Guias Paso a Paso (Quickstarts)



#### A. Agente Cloud con Herramientas



Guarda este codigo como `asistente.py` y ejecutalo con `python asistente.py`:



```python

import asyncio

from chatflow_agent import Agent, Runner



# 1. Definir el agente

agente_soporte = Agent(

    name="SoporteTienda",

    model="gemini-2.5-flash",  # O provider="openai", model="gpt-4o-mini"

    instructions="Eres un asistente cordial de atencion al cliente para una tienda online.",

)



# 2. Registrar herramientas de negocio usando decoradores y tipado estándar

@agente_soporte.tool

def consultar_pedido(pedido_id: str) -> dict:

    """Consulta el estado de despacho y envio de un pedido por su identificador."""

    return {

        "pedido_id": pedido_id,

        "estado": "En distribucion",

        "transporte": "Envios Express",

        "llegada_estimada": "Hoy antes de las 18:00 hs",

    }



async def main():

    runner = Runner(starting_agent=agente_soporte)

    respuesta = await runner.run_async(

        session_id="cliente_101",

        user_message="Hola, ¿podrias revisar en que estado se encuentra mi pedido #PED-9921?",

    )

    print(f"[{respuesta.active_agent_name}]: {respuesta.content}")



if __name__ == "__main__":

    asyncio.run(main())

```



---



#### B. Agente 100% Offline y Gratuito (Google Gemma)



Ejecuta agentes de forma local en tu propia computadora, sin costos de API y con total privacidad de datos usando [Ollama](https://ollama.com):



1. Inicia Ollama con Gemma: `ollama run gemma2:2b` (o `gemma2:9b`)

2. Ejecuta este script:



```python

import asyncio

from chatflow_agent import Agent, Runner



# Se conecta a http://localhost:11434/v1 sin requerir claves de API

agente_local = Agent(

    name="AnalistaLocal",

    provider="ollama",

    model="gemma2:2b",

    instructions="Eres un analista financiero que procesa informacion confidencial de manera local.",

)



@agente_local.tool

def calcular_iva(subtotal: float, tasa_porcentaje: float = 21.0) -> dict:

    """Calcula el impuesto de IVA y el monto total de una operacion comercial."""

    iva = subtotal * (tasa_porcentaje / 100.0)

    return {"subtotal": subtotal, "iva": round(iva, 2), "total": round(subtotal + iva, 2)}



async def main():

    runner = Runner(starting_agent=agente_local)

    respuesta = await runner.run_async(

        session_id="usuario_local",

        user_message="Calcula el IVA de una factura de 550 dolares al 21%.",

    )

    print(f"[{respuesta.active_agent_name}]: {respuesta.content}")



if __name__ == "__main__":

    asyncio.run(main())

```



---



#### C. Swarm Multi-Agente con Handoffs Autonomos



Los agentes pueden delegar la conversacion a especialistas de forma autonoma segun la intencion del usuario:



```python

import asyncio

from chatflow_agent import Agent, Runner



# 1. Especialista: Soporte Tecnico

agente_tecnico = Agent(

    name="SoporteTecnico",

    model="gemini-2.5-flash",

    instructions="Resuelves incidentes tecnicos de conectividad y servidores.",

)



@agente_tecnico.tool

def diagnosticar_nodo(nodo_id: str) -> dict:

    """Diagnostica el estado de un nodo de red."""

    return {"nodo_id": nodo_id, "estado": "Operativo", "latencia_ms": 22}



# 2. Especialista: Facturacion y Cobranzas

agente_facturacion = Agent(

    name="Facturacion",

    model="gemini-2.5-flash",

    instructions="Resuelves dudas sobre cobros, medios de pago y recibos.",

)



# 3. Recepcionista de Primera Linea (Frontline) con handoffs a especialistas

recepcionista = Agent(

    name="Recepcion",

    model="gemini-2.5-flash",

    instructions="Saludas al usuario y derivas a SoporteTecnico o Facturacion segun corresponda.",

    handoffs=[agente_tecnico, agente_facturacion],

)



async def main():

    runner = Runner(starting_agent=recepcionista)



    # El modelo detecta que es un problema tecnico y realiza la transferencia de forma automatica

    respuesta = await runner.run_async(

        session_id="cliente_77",

        user_message="Tengo una caida en el nodo SRV-04. ¿Pueden revisarlo?",

    )

    print(f"Agente Activo: {respuesta.active_agent_name}")  # Salida: SoporteTecnico

    print(f"Respuesta: {respuesta.content}")



if __name__ == "__main__":

    asyncio.run(main())

```



---



### Conectores de Canales Oficiales



#### Canal de WhatsApp (Meta Cloud API)



Servidor de webhook listo para produccion, compatible con la API oficial de WhatsApp Business en Meta Cloud:



```python

from chatflow_agent import Agent, Runner

from chatflow_agent.channels import WhatsAppChannel



agente_ventas = Agent(name="ConciergeWhatsApp", instructions="Atiende consultas comerciales.")

runner = Runner(starting_agent=agente_ventas)



canal_whatsapp = WhatsAppChannel(

    verify_token="tu_token_verificacion_meta", # Configurado en tu app de Meta Developers

    access_token="EAA...",                     # Token permanente de System User en Meta

    phone_number_id="109876543210987",         # WhatsApp Phone Number ID del panel de Meta

    fallback_message="Disculpa, estamos experimentando una demora temporal. Por favor intenta en unos momentos.",

    unsupported_media_message="Por el momento solo puedo procesar mensajes de texto.",

)

canal_whatsapp.attach(runner)



if __name__ == "__main__":

    # Expone GET /webhook (verificacion de Meta) y POST /webhook (recepcion de mensajes)

    canal_whatsapp.run(host="0.0.0.0", port=8000)

```



> **¿Probando localmente?** Usa un tunel como [ngrok](https://ngrok.com) (`ngrok http 8000`) o Cloudflare Tunnels para brindarle a Meta la URL publica HTTPS (`https://tu-dominio.ngrok-free.app/webhook`).



---



#### Canal de Telegram



Conecta tu enjambre de agentes a un Bot de Telegram con polling asincrono y soporte nativo de comandos:



```python

from chatflow_agent import Agent, Runner

from chatflow_agent.channels import TelegramChannel



agente_bot = Agent(name="TelegramBot", instructions="Asistes a los usuarios en Telegram.")

runner = Runner(starting_agent=agente_bot)



canal_telegram = TelegramChannel(

    token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",  # Token provisto por @BotFather

    fallback_message="Disculpa, ocurrio un inconveniente temporal. Por favor reintenta en breve.",

    start_message="Bienvenido al bot. ¿En que podemos ayudarte hoy?",

)

canal_telegram.attach(runner)



if __name__ == "__main__":

    # Ejecuta el bucle asincrono; gestiona /start, /reset e indicadores de escritura de forma automatica

    canal_telegram.run()

```



---



#### Canal de Terminal Interactiva (CLI)



Prueba y depura enjambres de agentes directamente en tu consola con formato legible:



```python

from chatflow_agent import Agent, Runner

from chatflow_agent.channels import CLIChannel



agente_terminal = Agent(name="AsistenteCLI", instructions="Responde de manera concisa y tecnica.")

runner = Runner(starting_agent=agente_terminal)



canal_cli = CLIChannel(session_id="prueba_desarrollo")

canal_cli.attach(runner)

canal_cli.run()

```



---



### Resiliencia y Concurrencia en Produccion



`chatflow-agent` incorpora protecciones nativas disenadas especificamente para el trafico real de canales de mensajeria:



* **Candado de Concurrencia por Sesion (`asyncio.Lock`):** Si un usuario envia varios mensajes consecutivos en pocos segundos (mensajes rapidos o transcripciones de notas de voz), un cerrojo asincrono garantiza el procesamiento en estricto orden FIFO. Esto evita condiciones de carrera, ejecuciones cruzadas de herramientas y corrupcion de historiales.

* **Escudo Anti-500 para Webhooks de Meta:** Si tu endpoint devuelve un error 500, los servidores de Meta reintentan agresivamente la entrega, saturando el servidor. `WhatsAppChannel` captura caidas o limites de cuota de los LLMs, responde **HTTP 200** a Meta para detener la tormenta de reintentos, envia al cliente un mensaje amigable (`fallback_message`) y registra el diagnostico de forma limpia.

* **Manejo Seguro de Archivos Multimedia:** Audios, fotos y documentos son interceptados con un aviso claro (`unsupported_media_message`) sin generar excepciones no controladas ni interrumpir la sesion en curso.



---



### Memoria de Sesiones y Persistencia en SQLite



`chatflow-agent` ofrece persistencia nativa con **cero dependencias externas adicionales**, garantizando que las conversaciones y los agentes especialistas activos sobrevivan a reinicios del servidor o reciclado de procesos.



#### Almacenamiento Volatil en Memoria RAM (`SessionStore`)

Ideal para pruebas, scripts locales o sesiones efimeras:

```python

from chatflow_agent import Runner, SessionStore



runner = Runner(starting_agent=agente_recepcion, session_store=SessionStore(default_max_turns=20))

```



#### Almacenamiento Persistente en Disco SQLite (`SQLiteSessionStore`)

Basado en la libreria estandar `sqlite3` de Python. Gestiona de manera segura transacciones atomicas y bloqueos de archivo en Windows:

```python

from chatflow_agent import Runner, SQLiteSessionStore



# El historial y la derivacion al especialista activo se conservan entre reinicios del servidor

session_store = SQLiteSessionStore(db_path="conversaciones.db", default_max_turns=30)

runner = Runner(starting_agent=agente_recepcion, session_store=session_store)

```



**Ventajas clave:**

* **Persistencia del Agente Activo:** Si un cliente fue derivado a `Facturacion`, el puntero se guarda en SQLite. Al reiniciar el servidor, los nuevos mensajes continuan directamente con `Facturacion` sin volver a comenzar en la recepcion.

* **Poda por Ventana Deslizante (FIFO):** Mantiene acotado el tamano del historial en la base de datos para no desbordar la ventana de contexto de los modelos.

* **Compatibilidad Omnicanal:** Funciona exactamente igual en WhatsApp, Telegram, REST APIs y CLI.



---



### Plantillas de Produccion Listas para Usar



En el directorio [`examples/`](examples/) encontraras proyectos completos listos para clonar y ejecutar en tu entorno:



* **[examples/whatsapp_persistent_bot.py](examples/whatsapp_persistent_bot.py):** Bot de WhatsApp Business (Meta Cloud API) con persistencia SQLite, candados de concurrencia y jerarquia de 3 agentes con handoffs.

* **[examples/telegram_persistent_bot.py](examples/telegram_persistent_bot.py):** Bot de Telegram con memoria de sesion y retencion de especialista entre reinicios de polling.

* **[examples/webhook_api_service.py](examples/webhook_api_service.py):** Servicio REST API asincrono con CORS y endpoints `/api/chat` y `/api/sessions/{session_id}` listo para frontends en React, Next.js y apps moviles.

* **[examples/local_gemma_cli.py](examples/local_gemma_cli.py):** Asistente de terminal 100% offline y gratuito con Google Gemma 2 via Ollama y sesiones persistentes en SQLite.

* **[examples/multi_provider_swarm.py](examples/multi_provider_swarm.py):** Enjambre multi-proveedor que coordina Gemini, Gemma local, Grok y Claude en un unico flujo conversacional.



---



## Author & Community

Created and maintained by **[Gustavo Baranda](https://github.com/GustavoBaranda)**.



If you find `chatflow-agent` useful for your projects, consider giving it a star on GitHub!  

Contributions, issues, and feature requests are always welcome.



---



## License

Distributed under the **MIT License**.

