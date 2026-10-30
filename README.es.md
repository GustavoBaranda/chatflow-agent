# chatflow-agent

[![CI](https://github.com/GustavoBaranda/chatflow-agent/actions/workflows/test.yml/badge.svg)](https://github.com/GustavoBaranda/chatflow-agent/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/chatflow-agent.svg)](https://pypi.org/project/chatflow-agent/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

🌐 **Idioma:** [English](README.md) | **Español** · [Arquitectura](docs/es/ARCHITECTURE.md) · [Changelog](CHANGELOG.md) · [Contribuciones](#contribuciones) · [Paquete en PyPI](https://pypi.org/project/chatflow-agent/) · [Reportar Issue](https://github.com/GustavoBaranda/chatflow-agent/issues)

> **Construye equipos de agentes autónomos para WhatsApp, Telegram, Webhooks y CLI en Python con un núcleo ligero y agnóstico al proveedor.**

```bash
pip install chatflow-agent
```

`chatflow-agent` es un framework asíncrono de Python diseñado para orquestar flujos conversacionales multi-agente en canales de mensajería reales. Inspirado en patrones ligeros de transferencia entre agentes (handoffs), combina motores LLM multi-proveedor con candados de concurrencia por sesión, reversión transaccional ante fallos (rollback), persistencia en SQLite y conectores listos para producción en **WhatsApp (Meta Cloud API)**, **Telegram**, **Webhooks con FastAPI** y **CLI**.

![Arquitectura de chatflow-agent: Canales de Entrada (WhatsApp, Telegram, Webhook, CLI) conectados a Runner y Memoria de Sesión, orquestando un Agente Triage y un Especialista con handoff autónomo entre pares y herramientas @agent.tool](docs/images/architecture-es.png)

---

## Tabla de contenidos

1. [Características principales](#características-principales)
2. [Instalación y requisitos](#instalación-y-requisitos)
3. [Inicio rápido](#inicio-rápido)
4. [Nuestros colaboradores](#nuestros-colaboradores)
5. [Proveedores compatibles y configuración](#proveedores-compatibles-y-configuración)
6. [Agentes, herramientas y handoffs](#agentes-herramientas-y-handoffs)
7. [Sesiones, persistencia y concurrencia](#sesiones-persistencia-y-concurrencia)
8. [Idempotencia de tools y efectos secundarios](#idempotencia-de-tools-y-efectos-secundarios)
9. [Canales disponibles](#canales-disponibles)
10. [Ejemplos de uso](#ejemplos-de-uso)
11. [Documentación técnica](#documentación-técnica)
12. [Contribuciones](#contribuciones)
13. [Apoyo al proyecto](#apoyo-al-proyecto)
14. [Licencia](#licencia)

---

## Características principales

* **Núcleo ultraligero:** Solo 2 dependencias de ejecución en el núcleo (`pydantic>=2.6.0`, `httpx>=0.27.0`). Todos los motores LLM (**Google Gemini**, **OpenAI**, **Anthropic Claude**, **xAI Grok**, **DeepSeek**, **Groq**, **Ollama** y **vLLM**) funcionan de serie mediante cliente REST `httpx` nativo sin cadenas pesadas de SDKs; las dependencias opcionales de los canales se instalan únicamente cuando se necesitan.
* **Conectores omnicanal:** Adaptadores listos para producción para **WhatsApp (Meta Cloud API oficial y webhooks genéricos)**, **Telegram**, **Webhooks con FastAPI** y **CLI** interactiva.
* **Libertad multi-proveedor:** Ejecuta agentes sobre **Google Gemini**, **OpenAI**, **Anthropic Claude**, **xAI Grok**, **DeepSeek**, **Groq** o modelos locales mediante **Ollama** (**Google Gemma**, Llama, Mistral, Qwen, Phi) y **vLLM**, incluso combinando distintos proveedores en un mismo enjambre.
* **Handoffs autónomos entre pares:** Los agentes delegan conversaciones dinámicamente a especialistas según la intención del usuario, conservando el agente activo y los metadatos de la sesión entre turnos y reinicios del servidor.
* **Resiliencia en producción:** Serialización por sesión mediante `asyncio.Lock`, ejecución en pool de hilos de trabajo para bases `SQLiteSessionStore` en disco sin bloquear el event loop, rollback automático de historial/agente/metadatos ante errores o timeouts a mitad de turno y escudo anti-500 para webhooks de Meta Cloud API.

---

## Instalación y requisitos

* **Python:** `3.10`, `3.11`, `3.12` o `3.13`
* **Dependencias del núcleo:** `pydantic>=2.6.0`, `httpx>=0.27.0`

Instala únicamente los extras que tu despliegue requiera:

```bash
# Núcleo base (todos los motores LLM vía HTTPX, almacenes SQLite y en memoria, canal CLI)
pip install chatflow-agent

# Con conector de Webhooks para WhatsApp (FastAPI + Uvicorn)
pip install "chatflow-agent[whatsapp]"

# Con conector de Bot para Telegram (python-telegram-bot)
pip install "chatflow-agent[telegram]"

# Con dependencias para servicio Webhook FastAPI personalizado (FastAPI + Uvicorn)
pip install "chatflow-agent[webhook]"

# Paquete completo de canales (extras telegram, whatsapp y webhook)
pip install "chatflow-agent[all]"
```

---

## Inicio rápido

### Inicio rápido asíncrono multi-agente (Recomendado para Webhooks y Bots)

```python
import asyncio
from chatflow_agent import Agent, Runner

# 1. Definir agentes especialistas
tech_agent = Agent(
    name="TechSupport",
    model="gemini-2.5-flash",
    instructions="Ayuda a los clientes a resolver problemas técnicos.",
)
billing_agent = Agent(
    name="Billing",
    model="gemini-2.5-flash",
    instructions="Ayuda a los clientes con facturas, suscripciones y pagos.",
)

# 2. Agente recepcionista con handoffs autónomos entre pares
receptionist = Agent(
    name="Receptionist",
    model="gemini-2.5-flash",
    instructions="Saluda cordialmente al cliente y derívalo a TechSupport o Billing según su consulta.",
    handoffs=[tech_agent, billing_agent],
)

async def main() -> None:
    # Usar 'async with Runner(...)' reutiliza el pool de conexiones HTTP entre turnos en GeminiEngine
    async with Runner(starting_agent=receptionist) as runner:
        result = await runner.run_async(
            session_id="usuario_whatsapp_1",
            user_message="¡Hola! Necesito ayuda con un error en la factura de mi cuenta.",
        )
        print(f"[{result.active_agent_name}]: {result.content}")
        # Salida: [Billing]: Con gusto te ayudo a revisar tu factura. ¿Me podrías indicar tu número de cuenta?

if __name__ == "__main__":
    asyncio.run(main())
```

### Inicio rápido síncrono (`runner.run()`)

Para scripts de consola, tareas programadas o aplicaciones síncronas donde no hay un event loop activo, puedes invocar `runner.run()` directamente:

```python
from chatflow_agent import Agent, Runner

agent = Agent(
    name="Asistente",
    model="gemini-2.5-flash",
    instructions="Responde de forma concisa.",
)

runner = Runner(starting_agent=agent)
response = runner.run(session_id="usuario_sync_1", user_message="¡Hola!")
print(f"[{response.active_agent_name}]: {response.content}")
```

> [!NOTE]
> `runner.run()` administra su propio bucle de eventos mediante `asyncio.run()`. Si se invoca `runner.run()` dentro de un event loop que ya está en ejecución (por ejemplo, en un endpoint de FastAPI o dentro de una función `async def`), lanza un `RuntimeError` indicando que debe utilizarse `await runner.run_async(...)`.

---

## Nuestros colaboradores

`chatflow-agent` se construye gracias al aporte de una comunidad internacional de desarrolladores que contribuyen con código, correcciones de errores, mejoras de seguridad, pruebas automatizadas y documentación. Cada contribución —grande o pequeña— ayuda a mantener la librería sólida y accesible.

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

Consulta el [listado completo de colaboradores en GitHub](https://github.com/GustavoBaranda/chatflow-agent/graphs/contributors) o revisa la sección de [Contribuciones](#contribuciones) para sumarte al proyecto.

---

## Proveedores compatibles y configuración

### Configuración de claves de API

Puedes proporcionar tus credenciales mediante cualquiera de estos tres métodos:

#### 1. Directamente en el código Python

Pasa `api_key` al instanciar el `Runner` (compartida con el motor por defecto) o directamente a un `Agent` específico:

```python
# Pasándola al Runner (utilizada por el GeminiEngine predeterminado)
runner = Runner(starting_agent=mi_agente, api_key="AIzaSyTuClaveDeGemini")

# O directamente a un Agente específico
agente_grok = Agent(
    name="GrokSpecialist",
    provider="grok",
    model="grok-2-latest",
    api_key="xai-tu-clave-aqui",
    instructions="Eres un especialista en análisis de datos.",
)
```

#### 2. Usando un archivo `.env`

Crea un archivo `.env` en la raíz de tu proyecto y cárgalo con `python-dotenv` (paquete externo opcional: `pip install python-dotenv`):

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

#### 3. Mediante variables de entorno en la terminal

* **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="AIzaSy..."
  ```
* **Linux / macOS (Bash/Zsh):**
  ```bash
  export GEMINI_API_KEY="AIzaSy..."
  ```

### Tabla de referencia de proveedores

| Proveedor | Motor | Variable de Entorno | Configuración en `Agent` |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | `GeminiEngine` | `GEMINI_API_KEY` | `model="gemini-2.5-flash"` |
| **OpenAI** | `OpenAIEngine` | `OPENAI_API_KEY` | `provider="openai"`, `model="gpt-4o-mini"` |
| **Anthropic Claude** | `AnthropicEngine` | `ANTHROPIC_API_KEY` | `provider="anthropic"`, `model="claude-3-5-sonnet-latest"` |
| **xAI Grok** | `OpenAIEngine` | `XAI_API_KEY` | `provider="grok"`, `model="grok-2-latest"` |
| **DeepSeek** | `OpenAIEngine` | `DEEPSEEK_API_KEY` | `provider="deepseek"`, `model="deepseek-chat"` |
| **Groq** | `OpenAIEngine` | `GROQ_API_KEY` | `provider="groq"`, `model="llama-3.3-70b-versatile"` |
| **Ollama (Local)** | `OpenAIEngine` | *No requiere clave* | `provider="ollama"`, `model="gemma2:9b"` |
| **vLLM / Servidor Local** | `OpenAIEngine` | *No requiere en `localhost`* | `base_url="http://localhost:8000/v1"`, `model="..."` |

* **Motores HTTPX integrados:** Todos los motores (`GeminiEngine`, `OpenAIEngine` y `AnthropicEngine`) vienen incluidos de serie con `pip install chatflow-agent`, sin necesidad de instalar SDKs adicionales por proveedor.
* **Variables de entorno adicionales:** `GeminiEngine` también acepta `GOOGLE_API_KEY` como alternativa y `GOOGLE_GEMINI_BASE_URL` para endpoints personalizados. `Ollama` lee opcionalmente `OLLAMA_API_KEY`, y cualquier preset de `OpenAIEngine` respeta `<PROVEEDOR>_BASE_URL` (por ejemplo, `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`).
* **Autodetección por prefijo de modelo:** Cuando se omite `provider` en `Agent`, `resolve_engine()` selecciona automáticamente el motor según el prefijo de `agent.model`:
  * **Google Gemini (`GeminiEngine`):** `gemini-*` (o motor por defecto)
  * **OpenAI (`OpenAIEngine`):** `gpt-*`, `o1*`, `o3*`, `chatgpt-*`
  * **Anthropic Claude (`AnthropicEngine`):** `claude-*`
  * **xAI Grok (`OpenAIEngine`):** `grok*`
  * **DeepSeek (`OpenAIEngine`):** `deepseek-*`
  * **Ollama Local (`OpenAIEngine`):** `gemma*`, `llama*`, `mistral*`, `qwen*`, `phi*`
  * **Groq y vLLM / Servidores Locales (`OpenAIEngine`):** Se seleccionan explícitamente mediante `provider="groq"` o `base_url="http://localhost:8000/v1"`.

### Orden de precedencia en la resolución del motor (`resolve_engine`)

Al ejecutar un turno para un `Agent`, el `Runner` resuelve el motor LLM en este orden exacto:

1. **Motor explícito del agente (`agent.engine`):** Si se pasó una instancia de `BaseEngine` a `Agent(engine=...)`, se utiliza siempre.
2. **Proveedor explícito del agente (`agent.provider`):** Se resuelve cuando `agent.provider` está definido (`gemini`/`google`, `anthropic`/`claude`, `openai`, `grok`/`xai`, `ollama`/`gemma`, `deepseek` o `groq`).
3. **Endpoint personalizado del agente (`agent.base_url`):** Si se define `agent.base_url` sin un `provider` explícito, enruta a `OpenAIEngine` (seleccionando el preset oficial si la URL coincide con un endpoint oficial, o exigiendo `api_key` explícita salvo en `localhost`).
4. **Motor explícito del Runner (`Runner(engine=...)`):** Si se pasó un `engine` personalizado al `Runner`, tiene prioridad sobre las heurísticas por prefijo de nombre de modelo.
5. **Heurísticas por prefijo del modelo (`agent.model`):** Detecta automáticamente `claude-*`, `gpt-*`/`o1*`/`o3*`/`chatgpt-*`, `grok*`, `deepseek-*` y `gemma*`/`llama*`/`mistral*`/`qwen*`/`phi*`.
6. **Motor por defecto del Runner / fallback a Gemini:** Utiliza el `GeminiEngine` por defecto del `Runner` (configurado con `Runner(api_key=...)`) o recurre a `GeminiEngine(api_key=agent.api_key)`.

> [!IMPORTANT]
> **Aislamiento de credenciales de entorno por host:** Por seguridad, las variables de entorno (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`) solo se envían cuando el cliente se conecta a los endpoints HTTPS oficiales de cada proveedor (`api.openai.com`, `api.anthropic.com`, `api.x.ai`, `api.deepseek.com`, `api.groq.com`). Nunca se reenvían a endpoints `base_url` personalizados ni a servidores locales Ollama/vLLM. Para autenticar contra un proxy compatible con OpenAI propio, pasa `api_key="..."` explícitamente en el `Agent` o en `OpenAIEngine` (o `api_key=""` si no requiere autenticación).

> [!WARNING]
> **Vertex AI no está soportado.** `GeminiEngine` invoca únicamente la API de Gemini Developer (Google AI Studio) mediante una clave de API. Application Default Credentials (ADC) y las variables `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` y `GOOGLE_CLOUD_LOCATION` no se utilizan, por lo que las configuraciones que dependían de ellas fallarán con `ProviderError("Gemini API client is not configured. ...")`. Utiliza una clave de API de Gemini o proporciona tu propio motor mediante `Runner(engine=...)`.
>
> **`GeminiEngine.client` fue eliminado en v0.4.0.** Ese atributo almacenaba el cliente de `google-genai`, que ya no es una dependencia del proyecto. Asignarle un objeto simulado (mock) ya no tiene efecto. Si necesitas invocar el SDK directamente, instala `google-genai` y crea tu propio cliente. Para simular respuestas de Gemini en tests o personalizar el transporte HTTP, pasa un `httpx.AsyncClient` mediante `GeminiEngine(http_client=...)`; el cliente permanece bajo tu control y debe usarse en un único event loop. Consulta el [CHANGELOG](CHANGELOG.md) para ver los pasos de migración.

---

## Agentes, herramientas y handoffs

### A. Agente Cloud con herramientas (Tools)

Registra funciones de Python síncronas o asíncronas (incluyendo instancias de clases invocables y objetos `functools.partial`) mediante `@agent.tool`, el decorador independiente `@tool` o `agent.add_tool()`:

```python
import asyncio
from chatflow_agent import Agent, Runner

agente_soporte = Agent(
    name="SoporteTienda",
    model="gemini-2.5-flash",  # O provider="openai", model="gpt-4o-mini"
    instructions="Eres un asistente cordial de atención al cliente para una tienda online.",
)

@agente_soporte.tool
def consultar_pedido(pedido_id: str, incluir_detalle: bool = False) -> dict:
    """Consulta el estado de despacho y envío de un pedido por su identificador."""
    return {
        "pedido_id": pedido_id,
        "estado": "En distribución",
        "transporte": "Envíos Express",
        "llegada_estimada": "Hoy antes de las 18:00 hs",
        "incluir_detalle": incluir_detalle,
    }

async def main() -> None:
    async with Runner(starting_agent=agente_soporte) as runner:
        respuesta = await runner.run_async(
            session_id="cliente_101",
            user_message="Hola, ¿podrías revisar en qué estado se encuentra mi pedido #PED-9921?",
        )
        print(f"[{respuesta.active_agent_name}]: {respuesta.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### B. Agente 100% local y gratuito (Google Gemma vía Ollama)

Ejecuta agentes de forma 100% local en tu equipo, sin costos de API y con privacidad total de datos utilizando [Ollama](https://ollama.com):

1. Inicia Ollama con Gemma: `ollama run gemma2:9b` (o `gemma2:2b`)
2. Ejecuta este script (funciona de serie con `pip install chatflow-agent`):

```python
import asyncio
from chatflow_agent import Agent, Runner

# Se conecta a http://localhost:11434/v1 sin requerir claves de API
agente_local = Agent(
    name="AnalistaLocal",
    provider="ollama",
    model="gemma2:9b",
    instructions="Analizas reportes financieros confidenciales de manera local.",
)

@agente_local.tool
def calcular_iva(subtotal: float, tasa_porcentaje: float = 21.0) -> dict:
    """Calcula el impuesto de IVA y el monto total de una operación comercial."""
    iva = subtotal * (tasa_porcentaje / 100.0)
    return {"subtotal": subtotal, "iva": round(iva, 2), "total": round(subtotal + iva, 2)}

async def main() -> None:
    runner = Runner(starting_agent=agente_local)
    respuesta = await runner.run_async(
        session_id="usuario_local",
        user_message="Calcula el IVA de una factura de 450 dólares.",
    )
    print(f"[{respuesta.active_agent_name}]: {respuesta.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### C. Swarm multi-agente con handoffs autónomos

Los agentes pueden delegar la conversación a especialistas pares declarando `handoffs=[...]` (lo que genera automáticamente herramientas `transfer_to_<agente>`) o devolviendo un objeto `Handoff` con `context_updates` desde una herramienta personalizada:

```python
import asyncio
from chatflow_agent import Agent, Handoff, Runner

agente_tecnico = Agent(
    name="SoporteTecnico",
    model="gemini-2.5-flash",
    instructions="Diagnosticas problemas de hardware y software.",
)

@agente_tecnico.tool
def diagnosticar_dispositivo(dispositivo_id: str) -> str:
    """Verifica la telemetría de un dispositivo."""
    return f"Dispositivo {dispositivo_id}: Todos los sensores nominales. Firmware v2.1 actualizado."

agente_facturacion = Agent(
    name="Facturacion",
    model="gemini-2.5-flash",
    instructions="Gestionas facturas, suscripciones y solicitudes de reembolso.",
)

recepcionista = Agent(
    name="Recepcion",
    model="gemini-2.5-flash",
    instructions="Saluda al cliente y transfiere a SoporteTecnico o Facturacion según corresponda.",
    handoffs=[agente_tecnico, agente_facturacion],
)

# Opcional: herramienta personalizada que devuelve un Handoff explícito con actualización de metadatos
@recepcionista.tool
def escalar_facturacion_vip(cuenta_id: str) -> Handoff:
    """Deriva a un cliente VIP a Facturacion y guarda su ID de cuenta en los metadatos de la sesión."""
    return Handoff(
        target_agent_name="Facturacion",
        reason="Consulta de facturación VIP",
        context_updates={"vip_account_id": cuenta_id},
    )

async def main() -> None:
    async with Runner(starting_agent=recepcionista) as runner:
        respuesta = await runner.run_async(
            session_id="cliente_77",
            user_message="Mi dispositivo DEV-42 parpadea en rojo. ¿Pueden ejecutar un diagnóstico?",
        )
        print(f"Agente Activo: {respuesta.active_agent_name}")  # Salida: SoporteTecnico
        print(f"Respuesta: {respuesta.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

> [!NOTE]
> * **Instrucciones dinámicas:** `Agent(instructions=...)` acepta tanto un `str` como un invocable sin argumentos `Callable[[], str]` que se evalúa antes de cada llamada al LLM.
> * **Unicidad de nombres de agentes:** Cada instancia distinta de `Agent` en un enjambre (o registrada posteriormente mediante `agent.add_handoff()` / `runner.register_agent()`) debe tener un `name` único y no vacío. Registrar dos instancias distintas de `Agent` con el mismo `name` lanza `ValueError` de forma inmediata.

### Esquemas de parámetros de herramientas y `ToolSchemaError`

`Tool` inspecciona las anotaciones de tipos de Python y genera esquemas compatibles con cada proveedor de forma automática:

* **Tipos primitivos:** `str`, `int`, `float`, `bool`
* **Contenedores:** `list[T]`, `tuple[T, ...]`, `set[T]`, `frozenset[T]`, `Sequence[T]` (mapeados a `ARRAY` con esquema `items`), `dict[K, V]`, `Mapping[K, V]` (mapeados a `OBJECT`)
* **Nulabilidad y uniones:** `Optional[T]` / `T | None` conserva `nullable: True`; `int | float` se amplía a `NUMBER`; las uniones de múltiples tipos (`int | str`) generan `anyOf`.
* **Anotaciones avanzadas:** Soporta `typing.Annotated[T, ...]`, anotaciones pospuestas (`from __future__ import annotations`) y argumentos variádicos `*args` / `**kwargs` (excluidos del esquema).
* **Validación fail-fast (`ToolSchemaError`):** Si una unión o tipo opcional contiene una clase personalizada sin mapeo concreto a JSON Schema (por ejemplo, `MiClase | None`), el registro de la herramienta lanza `ToolSchemaError` inmediatamente en lugar de degradarse silenciosamente a `STRING`.

### Límites del Runner y excepciones públicas

| Parámetro / Excepción | Valor por defecto / Base | Descripción |
| :--- | :--- | :--- |
| `max_tool_iterations` (`Runner`) | `10` | Cantidad máxima de ciclos de ejecución de herramientas permitidos en un turno. |
| `max_handoffs_per_turn` (`Runner`) | `5` | Cantidad máxima de transferencias entre agentes permitidas en un mismo turno; al superarse, registra un `ToolResult` de error (`is_error=True`) y omite las llamadas a herramientas restantes de ese lote. |
| `turn_timeout` (`run_async`) | `None` | Límite de tiempo opcional por turno en segundos (comienza tras adquirir el lock de la sesión). |
| `ChatFlowError` | `Exception` | Clase base de todas las excepciones públicas de `chatflow_agent`. |
| `ProviderError` | `ChatFlowError` | Lanzada ante errores de API del proveedor LLM, respuestas bloqueadas o payloads malformados (extractos acotados a 1.000 caracteres). |
| `ToolSchemaError` | `ChatFlowError`, `TypeError` | Lanzada al registrar una herramienta cuando un miembro de una unión no puede mapearse a JSON Schema. |
| `ToolExecutionError` | `ChatFlowError` | Lanzada cuando falla la ejecución directa de una herramienta mediante `Tool.execute_async()`. |
| `ToolArgumentValidationError` | `ToolExecutionError` | Lanzada cuando los argumentos provistos por el LLM no superan la validación de esquema antes de ejecutar la herramienta. |
| `AgentHandoffError` | `ChatFlowError` | Lanzada cuando un handoff apunta a un agente destino desconocido o no autorizado (no declarado en `current_agent.handoffs`). |
| `ReentrantTurnError` | `ChatFlowError`, `RuntimeError` | Lanzada cuando un turno intenta iniciar otro turno de la misma sesión en línea. |
| `SessionDeletedError` | `ChatFlowError`, `RuntimeError` | Lanzada cuando la sesión de un turno en curso se elimina concurrentemente. |
| `DependencyError` | `ChatFlowError` | Lanzada cuando falta instalar un paquete opcional de proveedor o canal. |

---

## Sesiones, persistencia y concurrencia

`chatflow-agent` ofrece almacenes de sesión sin dependencias externas adicionales que conservan el historial de conversación, el agente especialista activo y los metadatos de la sesión entre turnos y reinicios del proceso.

### Almacenamiento en memoria (`SessionStore`)

Ideal para scripts locales, pruebas unitarias y sesiones efímeras:

```python
from chatflow_agent import Runner, SessionStore

runner = Runner(
    starting_agent=receptionist,
    session_store=SessionStore(default_max_turns=40),
)
```

### Almacenamiento SQLite en disco y en memoria (`SQLiteSessionStore`)

Basado en el módulo estándar `sqlite3` de Python con Write-Ahead Logging (`WAL`), transacciones atómicas, poda por ventana deslizante y deduplicación persistente de mensajes de webhooks (`dedup_ttl_hours=192`):

```python
from chatflow_agent import Runner, SQLiteSessionStore

# Almacén SQLite en archivo (persiste entre reinicios del servidor)
with SQLiteSessionStore(db_path="conversaciones.db", default_max_turns=30) as session_store:
    runner = Runner(starting_agent=receptionist, session_store=session_store)
    # ... ejecutar la aplicación ...

# O almacén SQLite en memoria compartida para pruebas de integración rápidas y aisladas:
memory_sqlite = SQLiteSessionStore(":memory:")
try:
    runner = Runner(starting_agent=receptionist, session_store=memory_sqlite)
finally:
    memory_sqlite.close()
```

**Ventajas clave de persistencia:**

* **Persistencia del agente activo:** Cuando un cliente es derivado a `Facturacion`, el nombre del agente activo y los `metadata` se guardan en SQLite. Tras reiniciar el servidor, los mensajes siguientes continúan directamente con `Facturacion`.
* **Poda por ventana deslizante (FIFO):** Conserva los últimos `default_max_turns` turnos completos del usuario y descarta mensajes antiguos para acotar la ventana de contexto enviada al LLM.
* **Gestión explícita del ciclo de vida:** Utiliza `with SQLiteSessionStore(...) as store:` o invoca `store.close()` al apagar el servicio para liberar conexiones `:memory:` y cerrar limpiamente el pool de hilos de trabajo.

### Resiliencia y concurrencia en producción

`chatflow-agent` incorpora protecciones nativas diseñadas específicamente para el tráfico real de canales de mensajería:

* **Candado de concurrencia por sesión (`asyncio.Lock`):** Si un usuario envía varios mensajes consecutivos en pocos segundos (mensajes rápidos o transcripciones de notas de voz), los turnos de una misma sesión se ejecutan de a uno. Esto evita condiciones de carrera, ejecuciones cruzadas de herramientas y corrupción de historiales. Se ejecutan en el orden en que empiezan a esperar la sesión (`asyncio.Lock` es justo); ese es el orden en que el canal inicia los turnos, que normalmente, aunque no siempre, coincide con el orden en que se enviaron los mensajes. Con `SQLiteSessionStore`, cada turno en espera vuelve a leer el historial, el agente activo y la metadata confirmados por el turno anterior antes de empezar.
  * **Alcance:** la garantía vale dentro de un proceso, una instancia del store y un event loop en ejecución (por ejemplo, un worker de Uvicorn, un bot de Telegram o llamadas sucesivas a `Runner.run()`). **No** se serializan los turnos de una misma sesión que provienen de varias instancias del store sobre el mismo archivo, de varios procesos o de varios threads con su propio event loop.
  * **Turnos iniciados desde un turno:** el código que inicia un turno (una tool, o una tarea o thread que la tool inicia) no puede iniciar otro turno de la misma sesión mientras ese turno se ejecuta: `runner.run_async()` / `runner.run()` lanzan `ReentrantTurnError` (un `RuntimeError`) en lugar de esperar un cerrojo que el turno actual podría estar esperando. Para encolar un turno posterior de la misma sesión, usa `runner.spawn_turn(session_id, mensaje)`: devuelve una `asyncio.Task` que se ejecuta cuando termina el turno actual. Nunca esperes esa tarea dentro del turno que la creó; espérala (o lee su excepción) después. Los turnos de otras sesiones pueden esperarse desde una tool normalmente.
  * **Borrar una sesión:** `delete()` y `clear()` no esperan a los turnos en curso. Un turno que se está ejecutando cuando se borra su sesión falla con `SessionDeletedError` sin dejar nada guardado; un turno que esperaba esa sesión inicia una sesión nueva y vacía. Borrar elimina los datos de la sesión; no bloquea su id, por lo que un mensaje nuevo vuelve a crearla.
  * **SQLite y el event loop:** con un `SQLiteSessionStore` en archivo, las lecturas y escrituras de sesión del `Runner`, del webhook de WhatsApp y del `/reset` de Telegram se ejecutan en un pool de 4 hilos propio del store, por lo que un disco lento o una base bloqueada por otro escritor no detiene otras solicitudes, sesiones ni timeouts. Una llamada SQLite ya iniciada no se puede interrumpir: un turno cancelado o vencido espera su escritura en curso, se revierte y recién entonces lanza la excepción (el timeout se detecta a tiempo, pero se informa cuando el rollback quedó guardado, hasta `busy_timeout`, 10 s, por escritura mientras la base está bloqueada). El webhook de WhatsApp sigue confirmando después de su escritura de deduplicación, y responde HTTP 500 si el bloqueo dura más que `busy_timeout` (Meta reintenta). `SessionStore`, los stores SQLite `:memory:`, las subclases y los stores personalizados siguen ejecutándose en el event loop. Esto no hace seguro un mismo archivo SQLite para varios workers o procesos.
  * **Límite de tiempo del turno:** `await runner.run_async(session_id, mensaje, turn_timeout=30)` limita el turno en sí (solo por nombre, en segundos). El límite empieza cuando el turno tiene la sesión, por lo que el tiempo que espera detrás de turnos anteriores de la misma sesión no cuenta. Al vencer, el turno se cancela, la sesión vuelve al estado previo al turno y se lanza `TimeoutError`. Debe ser un número positivo y finito; el valor por defecto, `None`, no pone límite.
  * **Tools síncronas después de un timeout:** una tool síncrona (`def`) se ejecuta en un thread de trabajo, y Python no puede detener un callable que ya comenzó. El timeout y el rollback conversacional se informan sin esperar indefinidamente ese callable, pero la sesión permanece protegida hasta que realmente retorna o lanza una excepción. Los turnos posteriores de esa sesión no comienzan mientras siga ejecutándose; para los canales, esta espera por trabajo residual cuenta dentro de `queue_timeout_seconds`. Las demás sesiones continúan normalmente. Si el callable nunca termina, la sesión afectada permanece protegida (las solicitudes de canal acotadas reciben el fallback configurado por timeout de cola); configura timeouts en el I/O de la propia tool y diseña efectos secundarios idempotentes. Las tools asíncronas conservan su cancelación normal.
  * **WhatsApp:** cada mensaje tiene dos límites separados. `queue_timeout_seconds` (120 por defecto) limita la espera detrás de turnos anteriores de la misma sesión: si vence, el usuario recibe `fallback_message` y el turno nunca se ejecuta (el mensaje no se agrega al historial y, como antes, un reintento de Meta con el mismo `wamid` se descarta). `llm_timeout_seconds` (120 por defecto) limita el turno en sí y empieza cuando el turno tiene la sesión: al vencer, el turno se cancela y se revierte, y el usuario recibe `fallback_message`. Por eso una respuesta puede tardar hasta `queue_timeout_seconds + llm_timeout_seconds`. Los logs distinguen los casos: `Session busy ...` (límite de espera), `Turn timed out ...` (límite del turno), `Dispatch backstop expired ...` (un `dispatch_async()` personalizado que nunca llegó a `Runner.run_async()`) y `TimeoutError raised while dispatching ...` (un `TimeoutError` del proveedor u otro código, no un límite del canal). `llm_timeout_seconds=None` o `inf` no pone límite al turno.

* **Escudo Anti-500 para Webhooks de Meta:** Si tu endpoint devuelve un error 500, los servidores de Meta reintentan agresivamente la entrega, saturando el servidor. `WhatsAppChannel` captura caídas o límites de cuota de los LLMs, responde **HTTP 200** a Meta para detener la tormenta de reintentos, envía al cliente el `fallback_message` configurado y registra el diagnóstico de forma limpia.

* **Manejo seguro de multimedia y eventos ignorados:** Los audios, fotos y archivos PDF son interceptados con `unsupported_media_message`, mientras que las reacciones (`reaction`) y eventos de sistema (`system`) de WhatsApp se ignoran limpiamente con HTTP 200 sin invocar el `Runner` ni enviar mensajes salientes.

* **Rollback ante fallas en el turno:** Si una solicitud al LLM excede el tiempo de espera o falla a mitad de turno, ChatFlow revierte automáticamente el historial de conversación al último turno consistente en memoria y SQLite (nota: revierte el historial conversacional; no deshace efectos secundarios externos de tools ya ejecutadas), garantizando un contexto limpio sin turnos huérfanos. Consulta [Idempotencia de tools y efectos secundarios](#idempotencia-de-tools-y-efectos-secundarios) para diseñar tools seguras ante reintentos.
  * Con `SQLiteSessionStore`, el rollback restaura el historial, el agente activo y la metadata con los que empezó el turno, incluidos los mensajes que la ventana deslizante podó durante el turno. Si algo más modificó la sesión durante el turno (otro contexto que cambió su metadata o su agente, un reset, un delete o un escritor fuera del alcance indicado arriba), solo se revierten los cambios del turno fallido, se conservan los mensajes, el agente y la metadata de los otros escritores, y se registra una advertencia.

---

## Idempotencia de tools y efectos secundarios

Las tools que interactúan con sistemas externos (procesan pagos, crean registros en base de datos, envían emails o mensajes, llaman a APIs externas, crean recursos remotos o disparan operaciones de negocio persistentes) pueden ejecutarse nuevamente después de un turno fallido o de un reintento del usuario. Esta sección explica cómo diseñarlas para que una llamada repetida no repita el efecto real.

### Rollback conversacional vs. efectos externos

Cuando un turno falla, ChatFlow restaura lo siguiente al último turno consistente, en memoria y en SQLite:

* el historial de la sesión;
* el agente activo;
* los metadatos de la sesión.

Esto **no** deshace operaciones realizadas fuera de `chatflow-agent`. Una tool puede ejecutarse con éxito y producir un efecto secundario externo, y el mismo turno puede fallar después por un error del proveedor o de red. Como el historial se revierte, el reintento ya no contiene evidencia de que la tool se ejecutó, por lo que la misma operación de negocio puede solicitarse otra vez.

Este es un comportamiento esperado en sistemas distribuidos. Se resuelve con un diseño idempotente de las tools, no con deduplicación automática en el `Runner`.

### Diseño de tools idempotentes

Identifica cada operación de negocio con un identificador de negocio estable, como:

```text
order_id
payment_id
request_id
operation_id
```

Úsalo para construir una clave de idempotencia (un ID de operación de negocio) que se mantenga igual entre reintentos:

```python
operation_id = f"refund:{order_id}"
```

Una clave de idempotencia identifica una operación de negocio. Reutilízala para los reintentos de esa misma operación, con los mismos parámetros semánticos (por ejemplo, el mismo pedido y el mismo monto); una operación distinta necesita una clave distinta.

> Una tool solo recibe los argumentos que entrega el modelo, por lo que no dejes que el modelo genere la clave. Usa un identificador que ya exista en tu sistema.

### Diseño de tools seguras ante reintentos

Reserva la clave de forma atómica, ejecuta el efecto secundario y luego registra el resultado. `payment_gateway` es un marcador de posición para tu sistema externo.

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


@agente_facturacion.tool
def create_refund(order_id: str, amount: float) -> dict:
    """Reembolsa un pedido. Es seguro llamarla más de una vez para el mismo pedido."""
    operation_id = f"refund:{order_id}"

    with sqlite3.connect(DB_PATH) as conn:
        try:
            # 1. Reservar la clave de forma atómica.
            conn.execute(
                "INSERT INTO operations (operation_id, status) VALUES (?, 'pending')",
                (operation_id,),
            )
        except sqlite3.IntegrityError:
            # Ya existe: devolver el resultado guardado en vez de repetir el efecto.
            status, result = conn.execute(
                "SELECT status, result FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if status == "done":
                return {**json.loads(result), "already_processed": True}
            return {"status": "in_progress", "operation_id": operation_id}

    # 2. Ejecutar el efecto secundario, enviando la clave al sistema externo.
    result = payment_gateway.refund(
        order_id=order_id, amount=amount, idempotency_key=operation_id
    )

    # 3. Registrar el resultado.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE operations SET status = 'done', result = ? WHERE operation_id = ?",
            (json.dumps(result), operation_id),
        )
    return result
```

* Devuelve el resultado guardado cuando la operación ya se ejecutó, para que el modelo responda de forma consistente en lugar de repetir la acción.
* Envía la clave al sistema externo cuando lo soporte.
* **Este ejemplo no recupera operaciones pendientes.** Una fila `pending` significa que la operación sigue en ejecución o que la ejecución se interrumpió antes de completarse, y el ejemplo no puede distinguir entre ambos casos. Llamar de nuevo a la tool no reintenta una operación pendiente; solo devuelve `in_progress`. Tu aplicación debe reconciliar o reintentar las operaciones pendientes de forma deliberada, usando la misma clave de idempotencia.

### Protección contra repeticiones (Replay)

Una tool puede llamarse otra vez porque un turno se reintenta después de un rollback, el usuario reenvía un mensaje o un canal entrega nuevamente un webhook. Las tools idempotentes hacen seguros todos estos casos: la segunda llamada encuentra la clave registrada y devuelve el resultado guardado.

### Seguimiento de transacciones externas

Registra lo que hizo cada tool fuera del historial de la conversación (la tabla `operations` de arriba es un ejemplo mínimo): la clave de idempotencia, el ID del sistema externo (ID de pago, de mensaje o del recurso remoto), el estado y el resultado devuelto al modelo. Un rollback del historial no afecta este registro.

### Patrones recomendados

| Tipo de tool | Enfoque recomendado |
| --- | --- |
| **Pagos** | Clave basada en `payment_id` u `order_id` más la operación; enviar la clave al proveedor de pagos. |
| **Registros en base de datos** | Restricciones `UNIQUE` o upserts (`INSERT ... ON CONFLICT`) sobre una clave de negocio. |
| **Emails / mensajes** | Reservar una entrega pendiente por `(destinatario, tipo_de_mensaje, id_de_negocio)`, enviarla y luego confirmar el resultado (ver la nota siguiente). |
| **Llamadas a APIs externas** | Enviar la clave de idempotencia del proveedor si existe; si no, consultar el estado previo primero. |
| **Recursos remotos** | Usar nombres o etiquetas deterministas derivados de un ID de negocio y buscarlos antes de crear. |
| **Operaciones de negocio persistentes** | Asignar un `operation_id` por acción de negocio y registrarlo antes de ejecutar. |

**Emails y mensajes.** Trata la entrega como tres pasos separados: reservar una entrega pendiente, enviar el mensaje y luego confirmar el resultado. Registrar una marca de "enviado" *antes* de enviar puede hacer que se pierda un mensaje si el proceso se detiene antes de la entrega. Si la entrega tiene éxito pero falla el registro del resultado, una marca local por sí sola no puede garantizar a la vez que no haya duplicados y que la entrega ocurra; eso requiere un proveedor externo que soporte idempotencia o reconciliación.

---

## Canales disponibles

### Canal de WhatsApp (Meta Cloud API)

Ejecuta un servidor de webhook listo para producción, compatible con la API oficial de WhatsApp Business en Meta Cloud (requiere `pip install "chatflow-agent[whatsapp]"`):

```python
from chatflow_agent import Agent, Runner, WhatsAppChannel

agente_ventas = Agent(name="ConciergeWhatsApp", instructions="Atiende consultas comerciales.")
runner = Runner(starting_agent=agente_ventas)

canal_whatsapp = WhatsAppChannel(
    verify_token="tu_token_verificacion_meta",  # Configurado en tu app de Meta Developers
    access_token="EAA...",                      # Token permanente de System User en Meta
    app_secret="a1b2c3d4...",                   # Meta App Secret (obligatorio con verify_signature=True)
    phone_number_id="109876543210987",          # WhatsApp Phone Number ID del panel de Meta
    fallback_message="Disculpa, estamos experimentando una demora temporal. Por favor intenta en unos momentos.",
    unsupported_media_message="Por el momento solo puedo procesar mensajes de texto.",
    llm_timeout_seconds=120.0,
    queue_timeout_seconds=120.0,
)

canal_whatsapp.attach(runner)

if __name__ == "__main__":
    # Expone GET /webhook (verificación), POST /webhook (mensajes) y GET /health
    canal_whatsapp.run(host="0.0.0.0", port=8000)
```

* **Verificación HMAC fail-closed (`verify_signature=True` por defecto):** Valida `X-Hub-Signature-256` usando `app_secret` en cada solicitud `POST /webhook` antes de procesar el JSON.
* **Protección de payloads:** Aplica `max_body_size` (1 MB por defecto, respondiendo HTTP 413 si se excede o HTTP 411 si falta `Content-Length`) y devuelve HTTP 400 ante JSON malformado o cuya raíz no sea un objeto.
* **Fragmentación automática y reintentos de salida:** Las respuestas que superan los 4.096 caracteres se dividen automáticamente mediante `split_message()`, y los errores transitorios de red o cuota al enviar mensajes se reintentan con backoff exponencial (`max_outbound_retries=3`).

> **¿Probando localmente?** Usa un túnel como [ngrok](https://ngrok.com) (`ngrok http 8000`) o Cloudflare Tunnels para brindarle a Meta una URL pública HTTPS (`https://tu-dominio.ngrok-free.app/webhook`).

> [!NOTE]
> **Documentación interactiva en desarrollo:** Por motivos de seguridad, la documentación interactiva de FastAPI (`/docs`, `/redoc` y `/openapi.json`) está deshabilitada por defecto en producción. Para habilitar Swagger UI y los esquemas OpenAPI durante el desarrollo local o debugging, pasa `enable_docs=True` al instanciar `WhatsAppChannel`:
> ```python
> canal_whatsapp = WhatsAppChannel(
>     ...,
>     enable_docs=True,  # Habilita /docs, /redoc y /openapi.json para desarrollo local
> )
> ```

> [!TIP]
> **Rate Limiting y Reverse Proxy en Producción:** Para proteger los endpoints `/webhook` y `/health` contra ataques de inundación, denegación de servicio (DoS) y agotamiento de CPU por peticiones no autenticadas sin riesgo de descartar ráfagas legítimas de webhooks de Meta a nivel aplicación, ubica siempre Uvicorn detrás de un reverse proxy en producción (como Nginx con `limit_req_zone` o Cloudflare Rate Limiting):
> ```nginx
> limit_req_zone $binary_remote_addr zone=whatsapp_limit:10m rate=10r/s;
> location /webhook {
>     limit_req zone=whatsapp_limit burst=20 nodelay;
>     proxy_pass http://127.0.0.1:8000;
>     proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
> }
> ```

### Canal de Telegram

Conecta tu enjambre de agentes a Telegram con polling asíncrono, indicadores de escritura, fragmentación automática de mensajes a 4.096 caracteres y soporte nativo de los comandos `/start` y `/reset` (requiere `pip install "chatflow-agent[telegram]"`):

```python
from chatflow_agent import Agent, Runner, TelegramChannel

agente_bot = Agent(name="TelegramBot", instructions="Asistes a los usuarios en Telegram.")
runner = Runner(starting_agent=agente_bot)

canal_telegram = TelegramChannel(
    token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",  # Token provisto por @BotFather
    fallback_message="Disculpa, ocurrió un inconveniente temporal. Por favor reintenta en breve.",
    start_message="Bienvenido al bot. ¿En qué podemos ayudarte hoy?",
)

canal_telegram.attach(runner)

if __name__ == "__main__":
    # Ejecuta polling asíncrono; en chats grupales, /reset está restringido a administradores del grupo
    canal_telegram.run()
```

### Terminal interactiva (CLI)

Prueba y depura enjambres multi-agente directamente en tu terminal:

```python
from chatflow_agent import Agent, CLIChannel, Runner

agente_terminal = Agent(name="AsistenteCLI", instructions="Responde de manera concisa y técnica.")
runner = Runner(starting_agent=agente_terminal)

canal_cli = CLIChannel(session_id="prueba_desarrollo")
canal_cli.attach(runner)

if __name__ == "__main__":
    # Soporta /reset para limpiar la memoria de la sesión, y 'exit' / 'quit' / 'salir' o Ctrl+C para salir
    canal_cli.run()
```

---

## Ejemplos de uso

En el directorio [`examples/`](examples/) encontrarás scripts de referencia listos para ejecutar:

* **[`examples/whatsapp_persistent_bot.py`](examples/whatsapp_persistent_bot.py):** Bot de WhatsApp Business (Meta Cloud API) con persistencia SQLite, candados de concurrencia por sesión y jerarquía de 3 agentes con handoffs.
* **[`examples/whatsapp_service.py`](examples/whatsapp_service.py):** Servicio base de webhook para WhatsApp Business Cloud API con herramientas personalizadas y handoffs a especialistas.
* **[`examples/telegram_persistent_bot.py`](examples/telegram_persistent_bot.py):** Bot de Telegram con persistencia de estado en SQLite entre reinicios de polling.
* **[`examples/telegram_triage.py`](examples/telegram_triage.py):** Bot multi-agente de clasificación (triage) para Telegram con herramientas de seguimiento de pedidos y soporte técnico.
* **[`examples/webhook_api_service.py`](examples/webhook_api_service.py):** Servicio REST API con FastAPI que expone `/api/chat`, `/api/sessions/{session_id}` y `/health` con persistencia SQLite.
* **[`examples/local_gemma_cli.py`](examples/local_gemma_cli.py):** Asistente de terminal 100% offline e impulsado por Google Gemma 2 vía Ollama con sesiones persistentes en SQLite.
* **[`examples/multi_provider_swarm.py`](examples/multi_provider_swarm.py):** Enjambre multi-proveedor que coordina Gemini, Gemma local, Grok y Claude en un único flujo conversacional.
* **[`examples/cli_multiagent.py`](examples/cli_multiagent.py):** Enjambre interactivo en consola que demuestra handoffs autónomos entre recepción, soporte técnico y facturación.

---

## Documentación técnica

* **[Guía de Arquitectura](docs/es/ARCHITECTURE.md):** Explicación detallada del bucle de ejecución interno, esquemas de mensajes, adaptadores de motores y persistencia de sesiones.
* **[Guía de Migración (0.3.4 → 0.4.0)](docs/es/MIGRATION_0.4.md):** Instrucciones de actualización, cambios incompatibles y nuevas funcionalidades en `0.4.0`.
* **[Changelog](CHANGELOG.md):** Historial de versiones, nuevas funcionalidades, avisos de seguridad y cambios incompatibles (breaking changes).
* **[Política de Seguridad](SECURITY.md):** Versiones soportadas e instrucciones para reportar vulnerabilidades.
* **[Código de Conducta](CODE_OF_CONDUCT.md):** Estándares de convivencia de la comunidad.
* **[English Documentation](README.md):** Documentación completa en inglés (`README.md`).

---

<a id="contribuciones"></a>
## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Ya sea que quieras corregir un error, mejorar la documentación, agregar pruebas o proponer nuevos conectores de canales, tu ayuda es valorada.

Si quieres ayudar a mejorar `chatflow-agent`, puedes revisar los Issues abiertos:

* 🟢 **[Good First Issues](https://github.com/GustavoBaranda/chatflow-agent/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** — Tareas bien delimitadas, ideales para realizar una primera contribución.
* 🟡 **[Help Wanted](https://github.com/GustavoBaranda/chatflow-agent/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)** — Tareas intermedias o avanzadas que buscan colaboradores.
* 📋 **[Todos los Issues abiertos](https://github.com/GustavoBaranda/chatflow-agent/issues)** — Consulta errores, mejoras y oportunidades de contribución disponibles.

Por favor, revisa nuestra **[Guía de contribución](CONTRIBUTING.md)** antes de enviar un Pull Request. Allí se encuentran las instrucciones de configuración del entorno, estilo de código (`ruff`, `mypy --strict`) y requisitos de pruebas (`pytest`).

---

<a id="apoyo-al-proyecto"></a>
## 💖 Apoyo al proyecto

Si encuentras útil `chatflow-agent` para tus proyectos u organización, considera apoyar su desarrollo continuo de código abierto a través de **[GitHub Sponsors](https://github.com/sponsors/GustavoBaranda)**.

El patrocinio ayuda al mantenimiento del proyecto, la mejora de la documentación, la ampliación de la cobertura de pruebas, el fortalecimiento de la confiabilidad en los canales de mensajería y la continuidad en el desarrollo de nuevas capacidades.

El patrocinio es completamente voluntario y no condiciona ni limita el acceso a ninguna funcionalidad, versión o documentación del proyecto de código abierto.

---

## Licencia

Distribuido bajo la **[Licencia MIT](LICENSE)**.

Creado y mantenido por **[Gustavo Baranda](https://github.com/GustavoBaranda)**, con contribuciones de la [comunidad](#nuestros-colaboradores).
