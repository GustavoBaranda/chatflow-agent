🌐 Idioma: [English](../MIGRATION_0.4.md) | **Español**

# Guía de Migración: `v0.3.4` → `0.4.0rc1`

Esta guía describe todo lo que necesitas saber al actualizar **`chatflow-agent`** desde **`v0.3.4`** hacia **`0.4.0rc1`** (y la serie `0.4.0`).

* **[1. ¿Quiénes deben realizar cambios?](#1-quiénes-deben-realizar-cambios)**
* **[2. Cambios incompatibles confirmados (`v0.3.4` → `0.4.0rc1`)](#2-cambios-incompatibles-confirmados-v034--040rc1)**
  * [2.1. `GeminiEngine` utiliza la API REST de Gemini mediante `httpx` en lugar de `google-genai`](#21-geminiengine-utiliza-la-api-rest-de-gemini-mediante-httpx-en-lugar-de-google-genai)
  * [2.2. Eliminación del atributo `GeminiEngine.client`](#22-eliminación-del-atributo-geminiengineclient)
  * [2.3. Rechazo de nombres de `Agent` duplicados entre instancias distintas](#23-rechazo-de-nombres-de-agent-duplicados-entre-instancias-distintas)
* **[3. Otros cambios relevantes y mejoras de `0.4.0`](#3-otros-cambios-relevantes-y-mejoras-de-040)**
  * [3.1. Nuevas funcionalidades](#31-nuevas-funcionalidades)
  * [3.2. Cambios de comportamiento compatibles y validaciones fail-closed](#32-cambios-de-comportamiento-compatibles-y-validaciones-fail-closed)
  * [3.3. Correcciones de bugs](#33-correcciones-de-bugs)
  * [3.4. Mejoras internas que no requieren migración](#34-mejoras-internas-que-no-requieren-migración)
* **[4. Documentación relacionada](#4-documentación-relacionada)**

---

## 1. ¿Quiénes deben realizar cambios?

Para la mayoría de las aplicaciones, actualizar de `v0.3.4` a `0.4.0rc1` es una **actualización directa que no requiere cambios en el código**. Utiliza la siguiente tabla de resumen para verificar si tu proyecto se ve afectado por alguno de los tres cambios incompatibles de `0.4.0rc1`.

| Patrón de uso | ¿Requiere acción? | Detalles |
| :--- | :--- | :--- |
| Uso estándar de `Agent`, `Runner`, `@tool`, `SessionContext` o `SQLiteSessionStore` | **No requiere cambios** | Los archivos de base de datos existentes, definiciones de agentes, herramientas y APIs de sesión mantienen compatibilidad. |
| `GeminiEngine` / `provider="gemini"` con `GEMINI_API_KEY`, `GOOGLE_API_KEY` o `api_key=...` | **No requiere cambios de código** | Ahora se ejecuta sobre `httpx` sin dependencias adicionales de SDK (`google-genai` ya no es necesario). |
| `OpenAIEngine`, `AnthropicEngine`, Groq, xAI Grok, DeepSeek, Ollama o vLLM | **No requiere cambios** | Todos los motores compatibles con OpenAI y Anthropic mantienen compatibilidad de código fuente. |
| `TelegramChannel`, `WhatsAppChannel` o `CLIChannel` | **No requiere cambios** | Los constructores y manejadores de webhooks existentes continúan funcionando sin modificaciones. |
| `GeminiEngine` basado en **Google Cloud Vertex AI / Application Default Credentials (ADC)** sin clave de API | **Sí — Requiere migración** | Consulta la [Sección 2.1](#21-geminiengine-utiliza-la-api-rest-de-gemini-mediante-httpx-en-lugar-de-google-genai). |
| Código o pruebas que acceden directamente a **`GeminiEngine.client`** | **Sí — Requiere migración** | Consulta la [Sección 2.2](#22-eliminación-del-atributo-geminiengineclient). |
| Grafos multiagente que registran **instancias distintas de `Agent` con el mismo `name`** | **Sí — Requiere migración** | Consulta la [Sección 2.3](#23-rechazo-de-nombres-de-agent-duplicados-entre-instancias-distintas). |

> [!NOTE]
> Los cambios de seguridad y aislamiento ya publicados en **`v0.3.3`** (validación estricta de host local para `OpenAIEngine(base_url=...)` sin clave de API, PRs #74/#75) y **`v0.3.4`** (aislamiento estricto de variables de entorno de claves de API por proveedor, PR #126 / `GHSA-mcgm-q29p-3984`) ya forman parte de `v0.3.4` y no constituyen nuevos cambios incompatibles en `0.4.0rc1`.

---

## 2. Cambios incompatibles confirmados (`v0.3.4` → `0.4.0rc1`)

### 2.1. `GeminiEngine` utiliza la API REST de Gemini mediante `httpx` en lugar de `google-genai`

* **Origen:** PR #130 (`308fbfc`, Issue #116)
* **Comportamiento anterior (`v0.3.4`):**
  * `google-genai>=0.3.0` era una dependencia principal del paquete en `pyproject.toml`.
  * `GeminiEngine` inicializaba `genai.Client(api_key=resolved_key)`. Cuando no se configuraba `api_key`, `GEMINI_API_KEY` ni `GOOGLE_API_KEY`, `GeminiEngine` emitía una advertencia en el log y permitía que `genai.Client()` recurriera a Google Cloud Vertex AI / Application Default Credentials (`GOOGLE_GENAI_USE_VERTEXAI` / ADC).
* **Comportamiento actual (`0.4.0rc1`):**
  * `google-genai` fue eliminado de `pyproject.toml`. Las dependencias principales son ahora únicamente `pydantic>=2.7.0` y `httpx>=0.27.0`.
  * `GeminiEngine` se comunica directamente con la API REST de Google Gemini (`https://generativelanguage.googleapis.com/v1beta` por defecto, o reemplazada mediante `GOOGLE_GEMINI_BASE_URL`) mediante `httpx.AsyncClient` usando el encabezado `x-goog-api-key` (o `api_key_in_query=True`), y acepta `http_client: httpx.AsyncClient | None = None`.
  * Si no se proporciona una clave mediante `api_key=...` y tampoco están definidas `GEMINI_API_KEY` ni `GOOGLE_API_KEY` en el entorno, `GeminiEngine.generate_turn_async()` lanza `ProviderError` en el primer turno (fail-closed) en lugar de intentar autenticación mediante Vertex AI ADC.
* **Usuarios afectados:**
  * Aplicaciones que utilizaban `GeminiEngine` sin clave de API apoyándose en Application Default Credentials de Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true`).
  * Entornos cuyos scripts de dependencias esperaban que `chatflow-agent` instalara `google-genai` de forma transitiva para código externo al framework.
* **Acción de migración requerida:**
  1. Para el uso estándar de la API de Gemini, proporciona una clave de API mediante `GEMINI_API_KEY` (o `GOOGLE_API_KEY`, o `Agent(..., api_key="...")` / `GeminiEngine(api_key="...")`).
  2. Para cargas de trabajo en Google Cloud Vertex AI, utiliza el endpoint compatible con OpenAI de Vertex AI mediante `OpenAIEngine(base_url=..., api_key=...)`, o bien implementa una subclase personalizada de `BaseEngine` que encapsule `google-genai` (declarando `google-genai` directamente en las dependencias de tu proyecto).

**Antes (`v0.3.4` — fallback implícito a Vertex AI ADC sin clave de API):**

```python
import os
from chatflow_agent import Agent

# En v0.3.4, omitir GEMINI_API_KEY/GOOGLE_API_KEY registraba un warning
# y dependía de las variables de entorno ADC de google-genai para Vertex AI:
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    provider="gemini",
    model="gemini-2.5-flash",
)
```

**Después (`0.4.0rc1` — clave explícita de Gemini API o `http_client` personalizado):**

```python
from chatflow_agent import Agent, GeminiEngine

# Opción A: Clave estándar de Gemini API (vía variable GEMINI_API_KEY o parámetro api_key)
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    provider="gemini",
    model="gemini-2.5-flash",
    api_key="AIzaSyYourGeminiKey",
)

# Opción B: Instancia directa de GeminiEngine
engine = GeminiEngine(
    api_key="AIzaSyYourGeminiKey",
    timeout_seconds=30.0,
)
assert engine.api_key == "AIzaSyYourGeminiKey"
```

---

### 2.2. Eliminación del atributo `GeminiEngine.client`

* **Origen:** PR #130 (`308fbfc`, Issue #116)
* **Comportamiento anterior (`v0.3.4`):**
  * `GeminiEngine.__init__()` creaba `self.client = genai.Client(...)`. Algunas suites de pruebas o extensiones personalizadas aplicaban monkeypatching sobre `engine.client.aio.models.generate_content` para simular respuestas de Gemini.
* **Comportamiento actual (`0.4.0rc1`):**
  * `GeminiEngine` ya no posee el atributo `.client` (acceder a `engine.client` lanza `AttributeError`).
  * En su lugar, `GeminiEngine` acepta `http_client: httpx.AsyncClient | None = None` en su constructor (y además soporta el `_ClientPool` por bucle de eventos de `Runner` al entrar mediante `async with runner:`).
* **Usuarios afectados:**
  * Código o pruebas unitarias que leen, simulan (mock) o aplican monkeypatching sobre `engine.client` o `engine.client.aio.models.generate_content`.
* **Acción de migración requerida:**
  * Reemplaza los mocks que apuntaban a `engine.client` por una de las siguientes alternativas:
    1. Un stub personalizado de `BaseEngine` pasado a `Runner(starting_agent=..., engine=...)`, o
    2. Un `httpx.AsyncClient(transport=httpx.MockTransport(...))` inyectado mediante `GeminiEngine(..., http_client=client)`.

**Antes (`v0.3.4` — monkeypatching sobre `engine.client`):**

```python
from unittest.mock import AsyncMock
from chatflow_agent import GeminiEngine

# Patrón legado de v0.3.4 (lanza AttributeError en 0.4.0rc1):
# engine = GeminiEngine(api_key="test-key")
# engine.client.aio.models.generate_content = AsyncMock(...)
```

**Después (`0.4.0rc1` — inyección de `http_client` con `httpx.MockTransport`):**

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

### 2.3. Rechazo de nombres de `Agent` duplicados entre instancias distintas

* **Origen:** PR #128 (`a3bcd02`, Issue #123)
* **Comportamiento anterior (`v0.3.4`):**
  * `Runner._register_agent_tree()` indexaba los agentes por `agent.name` en `self._agents_registry` usando `if agent.name in self._agents_registry: return`.
  * Si dos instancias **distintas** de `Agent` compartían el mismo `name` dentro de un grafo de handoffs, la segunda instancia era **ignorada silenciosamente**, redirigiendo los handoffs hacia la primera instancia e ignorando las instrucciones, herramientas y sub-handoffs de la segunda.
* **Comportamiento actual (`0.4.0rc1`):**
  * `Runner` compara la identidad de objeto (`existing is agent`):
    * Las **referencias cíclicas o compartidas** a la **misma** instancia de `Agent` (`a1 -> a2 -> a1`, o grafos en diamante que comparten un único objeto `Agent`) siguen estando completamente soportadas.
    * **Dos instancias distintas de `Agent`** (`existing is not agent`) con el mismo `name` en el grafo de handoffs ahora lanzan `ValueError: Duplicate agent name '<name>' detected in handoff graph. Each distinct Agent instance must have a unique name.` durante la inicialización de `Runner(...)` o el registro dinámico posterior.
* **Usuarios afectados:**
  * Aplicaciones multiagente que instancian múltiples objetos `Agent(name="...")` separados con nombres idénticos dentro del mismo grafo de `Runner`.
* **Acción de migración requerida:**
  * Asigna un `name` único a cada instancia distinta de `Agent`, o reutiliza la misma instancia de `Agent` cuando múltiples agentes padres deriven hacia el mismo especialista.

**Antes (`v0.3.4` — dos objetos `Agent` distintos con el mismo `name`):**

```python
from chatflow_agent import Agent, Runner

specialist_v1 = Agent(name="Support", instructions="Tier 1 support")
specialist_v2 = Agent(name="Support", instructions="Tier 2 support")
triage = Agent(
    name="Triage",
    instructions="Route requests to support.",
    handoffs=[specialist_v1, specialist_v2],
)

# En v0.3.4, specialist_v2 era descartado silenciosamente.
# En 0.4.0rc1, esto lanza ValueError inmediatamente:
try:
    Runner(starting_agent=triage)
except ValueError as exc:
    assert "Duplicate agent name 'Support'" in str(exc)
```

**Después (`0.4.0rc1` — nombres únicos para agentes distintos, o instancia compartida):**

```python
from chatflow_agent import Agent, Runner

# Solución 1: Asignar nombres únicos a instancias distintas de Agent
tier1_support = Agent(name="Tier1Support", instructions="Tier 1 support")
tier2_support = Agent(name="Tier2Support", instructions="Tier 2 support")
triage = Agent(
    name="Triage",
    instructions="Route requests to support.",
    handoffs=[tier1_support, tier2_support],
)
runner = Runner(starting_agent=triage)

# Solución 2: Reutilizar exactamente la misma instancia de Agent en grafos cíclicos/diamante
shared_billing = Agent(name="Billing", instructions="Shared billing specialist")
tier1_support.add_handoff(shared_billing)
tier2_support.add_handoff(shared_billing)
runner_shared = Runner(starting_agent=triage)
```

---

## 3. Otros cambios relevantes y mejoras de `0.4.0`

Las siguientes mejoras, validaciones y correcciones de errores se incorporaron entre `v0.3.4` y `0.4.0rc1`. Ninguna de ellas rompe configuraciones válidas de `v0.3.4`.

### 3.1. Nuevas funcionalidades

1. **Context Manager asíncrono en `Runner` para pooling compartido de conexiones HTTP (`async with runner:`):**
   * Entrar en `async with runner:` crea un `_ClientPool` asociado al bucle de eventos (`contextvars.ContextVar`) que reutiliza conexiones `httpx.AsyncClient` entre turnos para `GeminiEngine` en ese bucle, cerrando limpiamente el cliente compartido al salir (PR #130).
   * Llamar a `runner.run_async()` sin `async with runner:` sigue siendo 100% compatible (cada intento HTTP crea y cierra su propio cliente como fallback).
2. **Timeout de turno completo (`turn_timeout`) y `Runner.spawn_turn()`:**
   * `Runner.run_async(..., turn_timeout: float | None = None)` acota la ejecución activa del turno (comenzando una vez adquirido el lock de la sesión), cancelando y revirtiendo el turno y lanzando `TimeoutError` al expirar (PRs #120, #131).
   * `Runner.spawn_turn(session_id, user_message, metadata=None) -> asyncio.Task[AgentResponse]` encola un turno de seguimiento independiente como tarea en segundo plano en el bucle de eventos activo (PR #120).
3. **Timeout de espera en cola independiente en `WhatsAppChannel` (`queue_timeout_seconds`):**
   * `WhatsAppChannel.__init__(..., queue_timeout_seconds: float | None = 120.0)` limita cuánto tiempo espera un mensaje entrante de webhook por el lock de sesión de manera independiente al presupuesto de ejecución activa `llm_timeout_seconds` (PR #131).
4. **`SQLiteSessionStore(db_path=":memory:")` en memoria y métodos de ciclo de vida:**
   * `SQLiteSessionStore` ahora soporta `db_path=":memory:"` utilizando una única conexión persistente compartida protegida por lock de hilos, además de `store.close()` explícito y soporte de context manager (`with SQLiteSessionStore(...) as store:`) (PRs #107, #120, #133).
5. **Método público `Tool.to_openai_parameters()`:**
   * `Tool.to_openai_parameters()` forma ahora parte de la API pública de `Tool`, devolviendo el objeto `parameters` de JSON Schema estricto compartido entre los adaptadores de OpenAI, Anthropic y Gemini (PR #118).
6. **Comando interactivo `/reset` en `CLIChannel`:**
   * Los usuarios de `CLIChannel` pueden escribir `/reset` (o `/clear`) durante una sesión interactiva para limpiar el historial de conversación y restablecer `active_agent` y `metadata` sin reiniciar el proceso (PR #119).
7. **Autodetección del prefijo de modelo `deepseek-*` en `resolve_engine()`:**
   * Configurar `model="deepseek-chat"` o `model="deepseek-reasoner"` sin un `provider` explícito selecciona automáticamente `provider="deepseek"` (`https://api.deepseek.com`, `DEEPSEEK_API_KEY`) (PR #55).
8. **Nuevas excepciones públicas exportadas en `chatflow_agent` y `chatflow_agent.exceptions`:**
   * `ToolSchemaError` (subclase de `ChatFlowError`, PRs #118, #136)
   * `ReentrantTurnError` (subclase de `ChatFlowError` y `RuntimeError`, PR #120)
   * `SessionDeletedError` (subclase de `ChatFlowError`, PR #120)

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

### 3.2. Cambios de comportamiento compatibles y validaciones fail-closed

* **Detección fail-fast de reentrada en la misma sesión (`ReentrantTurnError`):** Llamar a `runner.run_async(session, ...)` recursivamente desde dentro de una herramienta ejecutándose sobre el **mismo** `session_id` ahora lanza inmediatamente `ReentrantTurnError` en lugar de producir un deadlock infinito sobre el lock de la sesión (PR #120).
* **Validación fail-closed de esquemas de herramientas (`ToolSchemaError`):** Registrar una función `@tool` con una anotación de tipo de parámetro no representable en JSON Schema (como una clase arbitraria personalizada, `object`, `callable` o `tuple` sin parámetros) ahora lanza `ToolSchemaError` en el momento de decoración/registro en lugar de convertir silenciosamente el parámetro a `"type": "string"` (PRs #118, #136).
* **Protección ante sesiones eliminadas (`SessionDeletedError`):** Si una `SQLiteSessionContext` es eliminada mediante `store.delete_session(session_id)` mientras un turno está en cola o en ejecución, las operaciones de persistencia subsiguientes sobre esa sesión lanzan `SessionDeletedError` en lugar de recrear filas parciales silenciosamente (PR #120).
* **Logs estructurados de timeouts en `WhatsAppChannel`:** Los mensajes de advertencia de timeout en `WhatsAppChannel` fueron refinados para distinguir entre contención en la cola del lock (`"Session busy for %s after %ss; skipping turn."`), timeouts de ejecución del turno (`"Turn timed out for %s after %ss."`) y expiración del backstop (`"Dispatch backstop expired for %s after %ss"` / `"TimeoutError raised while dispatching message from %s"`) (PR #131).
* **`Runner.run()` dentro de un bucle de eventos activo:** Llamar al helper síncrono `Runner.run()` desde dentro de un bucle de eventos `asyncio` ya en ejecución ahora lanza un error claro `RuntimeError("Runner.run() cannot be called from a running event loop. Use `await runner.run_async(...)` instead.")` (PR #57).
* **Validación de argumentos en constructores (`ValueError`):**
  * `Agent(name="", instructions="...")` o un `Agent.name` compuesto solo por espacios en blanco lanza `ValueError` (PR #86).
  * `SQLiteSessionStore(..., default_max_turns=0)` o `SessionContext(..., max_turns=0)` (o `< 0`) lanza `ValueError` (PR #63).
  * `split_message(text, max_len=0)` (o `< 0`) lanza `ValueError` (PR #46).

---

### 3.3. Correcciones de bugs

* **Concurrencia de sesiones SQLite y diario transaccional de turno (`SQLiteSessionStore`):**
  * Los turnos concurrentes dirigidos al mismo `session_id` a través de distintas instancias envoltorio de `SQLiteSessionContext` ahora se serializan mediante un registro débil de locks por `(event_loop, session_id)` y recargan automáticamente el último estado confirmado en SQLite (`session.refresh()`) al adquirir el lock (PR #120).
  * `SQLiteSessionContext.restore_history()` se ejecuta ahora dentro de una única transacción `BEGIN IMMEDIATE` utilizando una tabla temporal `turn_journal`, evitando condiciones de carrera y preservando filas insertadas concurrentemente (PR #120).
* **Rollback atómico del turno ante errores y cancelación:**
  * Cuando un turno falla con una excepción no controlada, expira su timeout (`turn_timeout`) o es cancelado (`asyncio.CancelledError`), `Runner` restaura atómicamente el checkpoint completo previo al turno: historial de mensajes (incluyendo mensajes antiguos podados por `SessionContext.max_turns`), `session.active_agent` y una copia profunda de `session.metadata` (PRs #34, #52, #69).
* **Normalización determinista de IDs de llamadas a herramientas entre proveedores:**
  * `replacement_tool_call_id(index, seed)` genera IDs alfanuméricos deterministas al normalizar llamadas a herramientas entre proveedores, evitando colisiones durante handoffs multi-proveedor (PR #130).
* **Reenvío de mensajes `Role.SYSTEM` del historial:**
  * Los mensajes `Role.SYSTEM` almacenados en mitad del historial de sesión ahora se fusionan y reenvían junto con `system_prompt` en `GeminiEngine`, `OpenAIEngine` y `AnthropicEngine` en lugar de ser descartados silenciosamente (PR #100).
* **Encapsulamiento resiliente en `ProviderError` ante respuestas malformadas del proveedor:**
  * Respuestas HTTP que no contienen JSON válido, ausencia de `choices`/`candidates` y argumentos JSON de herramientas malformados provenientes del proveedor ahora se encapsulan consistentemente en `ProviderError` en lugar de propagar `JSONDecodeError`, `IndexError` o `KeyError` crudos (PR #124).
* **Endurecimiento del webhook de `WhatsAppChannel`:**
  * Las notificaciones de webhook de tipo `reaction` y `system` ignoradas y las estructuras anidadas malformadas ya no provocan excepciones no controladas, y los cuerpos JSON que no son objetos (`[]`, `"str"`, `123`) ahora devuelven HTTP `400 Bad Request` (PRs #110, #117, #129).
* **Desambiguación de nombres de handoff y registro dinámico:**
  * Los nombres de agentes no ASCII (por ejemplo, `"Política"`, `"Müller"`) se transliteran limpiamente a nombres de herramientas de handoff válidos `[a-z0-9_]`, las colisiones tras la sanitización se desambiguan automáticamente (`transfer_to_support_2`), y llamar a `agent.add_handoff()` después de inicializar `Runner` registra dinámicamente el subárbol de destino (PRs #49, #108, #112).
* **Precedencia de `resolve_engine()` con `prefer_default_engine=True`:**
  * Cuando un `Agent` dentro de un enjambre multiagente hereda los valores por defecto `provider="gemini"` y `model="gemini-2.5-flash"` sin sobrescrituras explícitas, `Runner` ahora respeta el motor fallback `Runner(engine=...)` antes de recurrir a la configuración por defecto del entorno (PR #39).
* **Introspección de `Tool` y casos borde de JSON Schema:**
  * Se corrigió la generación de esquemas para módulos que usan `from __future__ import annotations` (PR #72), uniones PEP 604 (`str | int`, `str | None`, PRs #26, #118), `typing.Annotated[T, ...]` (PR #62), contenedores tipados (`list[int]`, `set[str]`, `dict[str, int]`, PRs #40, #87), callables `functools.partial` síncronos y asíncronos (PRs #43, #111), parámetros variádicos `*args`/`**kwargs` (PR #35) y parámetros de herramienta llamados literalmente `default` (PR #68).

---

### 3.4. Mejoras internas que no requieren migración

* **I/O no bloqueante para SQLite basado en archivo:** Las instancias de `SQLiteSessionStore` basadas en archivo delegan las operaciones bloqueantes de SQLite a un `ThreadPoolExecutor` dedicado de 4 workers mediante `_run_blocking()` cuando se invocan desde un contexto asíncrono, evitando bloqueos del bucle de eventos bajo carga (PR #133).
* **Deduplicación atómica de `wamid` en una sola sentencia:** `SQLiteSessionStore.record_processed_message()` utiliza ahora una única sentencia atómica `INSERT OR IGNORE INTO processed_messages` con purga en línea de registros expirados (PR #120).
* **Gestión automática del pool en canales y `Runner.run()`:** `Runner.run()`, `TelegramChannel.run()` y el ciclo de vida (`lifespan`) de `WhatsAppChannel` entran automáticamente en `async with runner:` para que el pooling de conexiones esté activo sin configuración adicional (PR #130).
* **Compatibilidad total hacia atrás del esquema SQLite:** Los archivos de base de datos SQLite existentes creados en `v0.3.x` funcionan en `0.4.0rc1` sin necesidad de migración manual de esquema.

---

## 4. Documentación relacionada

* **[Guía de Arquitectura](ARCHITECTURE.md):** Desglose detallado del bucle de ejecución de `Runner`, resolución de motores, introspección de herramientas, handoffs, concurrencia en SQLite y canales.
* **[README en Español](../../README.es.md):** Inicio rápido, tabla de referencia de proveedores y descripción completa de funcionalidades.
* **[Changelog](../../CHANGELOG.md):** Notas de versión cronológicas.
