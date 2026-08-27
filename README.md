# relay-agent

[![PyPI version](https://img.shields.io/pypi/v/relay-agent.svg)](https://pypi.org/project/relay-agent/)
[![Python versions](https://img.shields.io/pypi/pyversions/relay-agent.svg)](https://pypi.org/project/relay-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> Lightweight, async multi-agent framework with native handoffs powered by Google Gemini.  
> Connect autonomous agents to **WhatsApp**, **Telegram**, and **Webhooks**.  
> *Marco de trabajo multiagente asíncrono y ultra liviano con transferencias nativas (handoffs) impulsado por Google Gemini. Conecta agentes autónomos a **WhatsApp**, **Telegram** y **Webhooks**.*

---

## English

### Overview
`relay-agent` is designed for simplicity, speed, and seamless multi-agent orchestration. Build specialized autonomous agents and connect them directly to real-world channels (**WhatsApp**, **Telegram**, **Webhooks**, **CLI**) with native tool calling and Swarm-style peer handoffs.

### Key Features
- **Ultra-light Core:** Built on top of `google-genai` and `pydantic` with zero bloated dependencies.
- **Native Peer Handoffs:** Declarative agent transfers (`handoffs=[billing_agent, tech_agent]`) without complex graph definitions.
- **WhatsApp & Telegram Ready:** Direct connectors for Telegram polling/webhooks and WhatsApp (via Meta Cloud API, Evolution API, or generic webhooks).
- **Pythonic Tool Registry:** Register tools using the `@agent.tool` decorator; schema extraction is automated via type hints and docstrings.
- **Pluggable Channels:** Run across chat platforms or locally via an interactive CLI.

### Quickstart: Telegram Bot
```python
from relay_agent import Agent, Runner, TelegramChannel

# 1. Define specialized agents
billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle invoices and billing inquiries.",
)

@billing_agent.tool
def get_invoice_status(invoice_id: str) -> dict:
    """Retrieve payment status for a specific invoice."""
    return {"invoice_id": invoice_id, "status": "PAID", "amount": 120.0}

# 2. Define root triage agent with handoff
triage_agent = Agent(
    name="Triage Agent",
    instructions="Frontline triage. Route billing queries to Billing Agent.",
    handoffs=[billing_agent],
)

# 3. Run multi-agent orchestrator on Telegram
runner = Runner(starting_agent=triage_agent)
channel = TelegramChannel(token="YOUR_TELEGRAM_BOT_TOKEN")
channel.attach(runner)

if __name__ == "__main__":
    channel.run()
```

### Quickstart: WhatsApp (Webhook / Meta Cloud API)
```python
from relay_agent import Agent, Runner, WhatsAppChannel

# 1. Define agents and tools as usual
support_agent = Agent(
    name="WhatsApp Support",
    instructions="You provide friendly 24/7 customer assistance.",
)

@support_agent.tool
def check_order(order_id: str) -> dict:
    """Check shipment and delivery status of an order."""
    return {"order_id": order_id, "status": "Out for delivery", "eta": "2 hours"}

# 2. Attach runner to WhatsApp Channel
runner = Runner(starting_agent=support_agent)
whatsapp = WhatsAppChannel(
    verify_token="YOUR_WEBHOOK_VERIFY_TOKEN",
    access_token="YOUR_WHATSAPP_ACCESS_TOKEN",
    phone_number_id="YOUR_PHONE_NUMBER_ID",
)
whatsapp.attach(runner)

if __name__ == "__main__":
    whatsapp.run(port=8000)  # Starts webhook server ready for Meta / Evolution API
```

---

## Español

### Descripción General
`relay-agent` está diseñado para ofrecer máxima simplicidad y velocidad en la orquestación multiagente. Permite construir agentes autónomos especializados y conectarlos directamente a canales reales (**WhatsApp**, **Telegram**, **Webhooks HTTP**, **terminal interactiva**) con soporte nativo de *Function Calling* y transferencias fluidas estilo Swarm.

### Características Principales
- **Núcleo ultra liviano:** Basado únicamente en `google-genai` y `pydantic`.
- **Integración con WhatsApp y Telegram:** Conectores directos para Telegram y WhatsApp (API oficial de Meta Cloud, Evolution API o Webhooks FastAPI).
- **Handoffs nativos entre agentes:** Conexiones declarativas entre especialistas sin grafos complejos ni dependencias innecesarias.
- **Registro idiomático de herramientas:** Usa el decorador `@agent.tool`; los esquemas JSON se generan automáticamente a partir de *type hints* y *docstrings*.
- **Canales modulares desacoplados:** Instala solo lo que necesitas mediante dependencias opcionales (*extras*).

---

## Installation / Instalación

```bash
# Core package only / Solo el núcleo
pip install relay-agent

# With WhatsApp support / Con soporte para WhatsApp
pip install "relay-agent[whatsapp]"

# With Telegram support / Con soporte para Telegram
pip install "relay-agent[telegram]"

# Full installation (all channels) / Instalación completa (todos los canales)
pip install "relay-agent[all]"
```

## License / Licencia
MIT License.
