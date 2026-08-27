# relay-agent

[![PyPI version](https://img.shields.io/pypi/v/relay-agent.svg)](https://pypi.org/project/relay-agent/)
[![Python versions](https://img.shields.io/pypi/pyversions/relay-agent.svg)](https://pypi.org/project/relay-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> Lightweight, async multi-agent framework with native handoffs powered by Google Gemini.  
> *Marco de trabajo multiagente asíncrono y ultra liviano con transferencias nativas (handoffs) impulsado por Google Gemini.*

---

## English

### Overview
`relay-agent` is designed for simplicity, speed, and seamless multi-agent orchestration. Build specialized autonomous agents and connect them directly to real-world channels (Telegram, Webhooks, CLI) with native tool calling and Swarm-style peer handoffs.

### Key Features
- **Ultra-light Core:** Built on top of `google-genai` and `pydantic` with zero bloated dependencies.
- **Native Peer Handoffs:** Declarative agent transfers (`handoffs=[billing_agent, tech_agent]`) without complex graph definitions.
- **Pythonic Tool Registry:** Register tools using the `@agent.tool` decorator; schema extraction is automated via type hints and docstrings.
- **Pluggable Channels:** Connect seamlessly to Telegram or Webhook receivers, or run locally via interactive CLI.

### Quickstart
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

# 2. Define root agent with handoffs
triage_agent = Agent(
    name="Triage Agent",
    instructions="Frontline triage. Route billing queries to Billing Agent.",
    handoffs=[billing_agent],
)

# 3. Run multi-agent orchestrator
runner = Runner(starting_agent=triage_agent)
channel = TelegramChannel(token="YOUR_TELEGRAM_BOT_TOKEN")
channel.attach(runner)

if __name__ == "__main__":
    channel.run()
```

---

## Español

### Descripción General
`relay-agent` está diseñado para ofrecer máxima simplicidad y velocidad en la orquestación multiagente. Permite construir agentes autónomos especializados y conectarlos directamente a canales reales (Telegram, Webhooks, terminal interactiva) con soporte nativo de *Function Calling* y transferencias fluidas estilo Swarm.

### Características Principales
- **Núcleo ultra liviano:** Basado únicamente en `google-genai` y `pydantic`.
- **Handoffs nativos entre agentes:** Conexiones declarativas entre agentes sin grafos complejos ni dependencias pesadas.
- **Registro idiomático de herramientas:** Usa `@agent.tool`; los esquemas JSON se generan automáticamente a partir de *type hints* y *docstrings*.
- **Canales modulares:** Conectores listos para Telegram y Webhooks (FastAPI).

---

## Installation / Instalación

```bash
# Core package only / Solo el núcleo
pip install relay-agent

# With Telegram support / Con soporte para Telegram
pip install "relay-agent[telegram]"

# With Webhook support / Con soporte para Webhooks
pip install "relay-agent[webhook]"

# Full installation / Instalación completa
pip install "relay-agent[all]"
```

## License / Licencia
MIT License.
