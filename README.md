# chatflow-agent

[![PyPI version](https://img.shields.io/pypi/v/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![Python versions](https://img.shields.io/pypi/pyversions/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> Lightweight, async multi-agent framework with native handoffs.  
> Connect autonomous agents to **WhatsApp**, **Telegram**, **Webhooks**, and **CLI**.  
>  
> *Framework multiagente asíncrono y liviano con transferencias nativas (handoffs).  
> Conecta agentes autónomos a **WhatsApp**, **Telegram**, **Webhooks** y **Terminal**.*

---

## English

### Quick Installation

```bash
# Core framework
pip install chatflow-agent

# With WhatsApp connector
pip install "chatflow-agent[whatsapp]"

# With Telegram connector
pip install "chatflow-agent[telegram]"

# All connectors included
pip install "chatflow-agent[all]"
```

---

### 1. Basic Quickstart (Copy & Paste)

Set your Gemini API Key in your terminal:
```bash
# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key"

# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key"
```

Create `app.py` and run it:

```python
import asyncio
from chatflow_agent import Agent, Runner

# 1. Define specialist agent
support_agent = Agent(
    name="SupportSpecialist",
    model="gemini-2.5-flash",
    instructions="You help customers with order status and returns.",
)

@support_agent.tool
def get_order_status(order_id: str) -> dict:
    """Fetches live status of a customer order."""
    return {
        "order_id": order_id,
        "status": "In transit",
        "expected_delivery": "Tomorrow by 5 PM",
    }

# 2. Define triage / receptionist agent
reception_agent = Agent(
    name="Receptionist",
    model="gemini-2.5-flash",
    instructions="Greet the customer and transfer to SupportSpecialist if they ask about orders.",
    handoffs=[support_agent],  # Direct Swarm handoff
)

# 3. Run conversation turn
async def main():
    runner = Runner(starting_agent=reception_agent)
    response = await runner.run_async(
        session_id="user_123",
        user_message="Hi! Where is my order #ORD-9921?",
    )
    print(f"[{response.active_agent_name}]: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. WhatsApp Channel (Meta Cloud API Webhook)

Run a production-ready webhook server using FastAPI and Uvicorn:

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import WhatsAppChannel

support = Agent(
    name="WhatsAppBot",
    model="gemini-2.5-flash",
    instructions="You are a 24/7 customer service representative on WhatsApp.",
)

runner = Runner(starting_agent=support)

channel = WhatsAppChannel(
    runner=runner,
    verify_token="my_meta_secret_token",  # Set this in Meta App Dashboard
    access_token="EAA...",                # Meta System User Token
    phone_number_id="1234567890",         # Meta WhatsApp Phone Number ID
)

if __name__ == "__main__":
    # Starts server on http://localhost:8000/webhook
    channel.run(host="0.0.0.0", port=8000)
```

---

### 3. Telegram Bot Channel

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import TelegramChannel

sales = Agent(
    name="SalesBot",
    model="gemini-2.5-flash",
    instructions="You answer product inquiries and help users buy items.",
)

runner = Runner(starting_agent=sales)

channel = TelegramChannel(
    runner=runner,
    bot_token="YOUR_TELEGRAM_BOT_TOKEN",  # From @BotFather
)

if __name__ == "__main__":
    channel.run()
```

---

### 4. Multi-Provider AI (Gemma, Grok, OpenAI, Claude)

`chatflow-agent` supports multiple LLMs across agents in the same session:

```python
from chatflow_agent import Agent

# Offline local model with Ollama (requires Ollama running on localhost:11434)
local_gemma = Agent(
    name="LocalProcessor",
    provider="ollama",
    model="gemma2:9b",
    instructions="Process private records locally.",
)

# xAI Grok (requires XAI_API_KEY)
grok_agent = Agent(
    name="GrokAnalyst",
    provider="grok",
    model="grok-2",
    instructions="Perform deep technical research.",
)

# Anthropic Claude (requires ANTHROPIC_API_KEY)
claude_agent = Agent(
    name="ClaudeAuditor",
    provider="anthropic",
    model="claude-3-5-sonnet",
    instructions="Validate strict regulatory compliance.",
)
```

---

## Español

### Instalación

```bash
# Instalación base
pip install chatflow-agent

# Con conector de WhatsApp (FastAPI + Uvicorn)
pip install "chatflow-agent[whatsapp]"

# Con conector de Telegram
pip install "chatflow-agent[telegram]"

# Con todos los canales
pip install "chatflow-agent[all]"
```

---

### 1. Guía Rápida Funcional (Copiar y Pegar)

Configura tu API Key de Gemini en la terminal:
```bash
# En Windows PowerShell
$env:GEMINI_API_KEY="tu-clave-gemini"

# En Linux o Mac
export GEMINI_API_KEY="tu-clave-gemini"
```

Crea un archivo `ejemplo.py`:

```python
import asyncio
from chatflow_agent import Agent, Runner

# 1. Agente Especialista con Herramientas
soporte = Agent(
    name="Soporte",
    model="gemini-2.5-flash",
    instructions="Resuelves dudas sobre el estado de pedidos y envíos.",
)

@soporte.tool
def consultar_pedido(id_pedido: str) -> dict:
    """Consulta la información en vivo de un pedido."""
    return {
        "pedido": id_pedido,
        "estado": "En reparto",
        "entrega_estimada": "Hoy antes de las 18:00hs",
    }

# 2. Agente Recepcionista que deriva al especialista
recepcion = Agent(
    name="Recepcion",
    model="gemini-2.5-flash",
    instructions="Saluda amablemente y deriva a Soporte si preguntan por pedidos.",
    handoffs=[soporte],  # Transferencia de control nativa
)

# 3. Bucle de ejecución
async def main():
    runner = Runner(starting_agent=recepcion)
    
    # El usuario escribe y la IA decide a qué agente pasar y qué herramienta llamar
    respuesta = await runner.run_async(
        session_id="cliente_01",
        user_message="Hola, ¿dónde está mi pedido #1042?",
    )
    
    print(f"Respondió [{respuesta.active_agent_name}]: {respuesta.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

Ejecuta el archivo:
```bash
python ejemplo.py
```

---

### 2. Conexión con WhatsApp (Meta Cloud API)

Levanta un servidor Webhook listo para producción:

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import WhatsAppChannel

agente = Agent(
    name="AtencionWhatsApp",
    model="gemini-2.5-flash",
    instructions="Atiendes a clientes por WhatsApp las 24 horas.",
)

runner = Runner(starting_agent=agente)

canal_whatsapp = WhatsAppChannel(
    runner=runner,
    verify_token="mi_token_de_verificacion",  # El que colocas en Meta Developers
    access_token="EAA...",                    # Token de acceso de Meta
    phone_number_id="1234567890",             # ID de número de teléfono de WhatsApp
)

if __name__ == "__main__":
    # Abre el servidor en http://localhost:8000/webhook
    canal_whatsapp.run(host="0.0.0.0", port=8000)
```

---

### 3. Conexión con Telegram

```python
from chatflow_agent import Agent, Runner
from chatflow_agent.channels import TelegramChannel

agente = Agent(
    name="BotTelegram",
    model="gemini-2.5-flash",
    instructions="Atiendes consultas en Telegram de forma rápida.",
)

runner = Runner(starting_agent=agente)

canal_telegram = TelegramChannel(
    runner=runner,
    bot_token="TU_TOKEN_DE_BOTFATHER",
)

if __name__ == "__main__":
    canal_telegram.run()
```

---

### 4. Soporte Multi-Proveedor (Opcional)

Si quieres usar otros modelos además de Gemini:

```python
from chatflow_agent import Agent

# Gemma Local (Gratis, requiere Ollama corriendo en tu PC)
gemma_local = Agent(
    name="GemmaLocal",
    provider="ollama",          # Conecta a http://localhost:11434/v1
    model="gemma2:9b",
    instructions="Procesas datos confidenciales sin conexión a internet.",
)

# xAI Grok (Requiere variable XAI_API_KEY)
grok_agent = Agent(
    name="Grok",
    provider="grok",
    model="grok-2",
    instructions="Especialista en análisis e investigación técnica.",
)

# Anthropic Claude (Requiere variable ANTHROPIC_API_KEY)
claude_agent = Agent(
    name="Claude",
    provider="anthropic",
    model="claude-3-5-sonnet",
    instructions="Auditoría y validación de seguridad.",
)
```

---

## License
Distributed under the **MIT License**.
