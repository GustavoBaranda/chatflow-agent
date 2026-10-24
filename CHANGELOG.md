# Changelog / Registro de Cambios



All notable changes to this project will be documented in this file.  
*Todos los cambios notables en este proyecto serán documentados en este archivo.*

Format / Formato: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning / Versionado: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

[English](#english) | [Español](#español)

---

## English

## [Unreleased]

### ⚠ BREAKING CHANGES
1. **Vertex AI Is No Longer Supported by `GeminiEngine` (Issue #116):** `GeminiEngine` now calls the Gemini Developer API (Google AI Studio) over REST with an API key, instead of using the `google-genai` SDK. Application Default Credentials and the `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` variables, which `google-genai` could use, are ignored; such setups now fail with `ProviderError("Gemini API client is not configured. ...")`. Use a Gemini API key (`GEMINI_API_KEY`), or provide your own `BaseEngine` through `Runner(engine=...)`.
2. **`GeminiEngine.client` Removed (v0.4.0, Issue #116):** the attribute held the `google.genai.Client` (or `None` if it could not be created) and no longer exists. Reading it raises `AttributeError`; assigning it (e.g. a fake client in a test) raises nothing but has no effect, so requests then go to the real API. Migration:
   - **Calling the SDK directly** (e.g. listing models or uploading files): install `google-genai` yourself and create your own `genai.Client`; it is no longer a chatflow-agent dependency.
   - **Mocking Gemini in tests or customizing HTTP** (proxies, TLS, transports): pass an `httpx.AsyncClient` with `GeminiEngine(http_client=...)`, for example `httpx.AsyncClient(transport=httpx.MockTransport(handler))`, and use the engine through `Runner(engine=...)` (pass the key to `GeminiEngine(api_key=...)` or set `GEMINI_API_KEY`; `Runner(api_key=...)` is then not used). Responses follow the Gemini REST API (`generateContent`), not the SDK's objects. The client belongs to the caller: chatflow-agent never closes it, and it must be used on one event loop only, so it does not suit `Runner.run()` or `CLIChannel`, which start a new event loop per call.

### Fixed
- **Tool-Call IDs Across Provider Handoffs (Issue #116):** after a handoff between providers, `OpenAIEngine` and `AnthropicEngine` no longer send tool-call IDs the destination rejects. An ID Anthropic does not accept (anything but letters, digits, `_` and `-`), or one that is empty or longer than 40 characters for `OpenAIEngine(provider="openai")`, is sent as a stable replacement (`call_` and 24 hex digits), the same in the call and in its result. Other IDs are sent unchanged, and the session history keeps the originals, so the provider that issued an ID gets it back. `GeminiEngine` keeps the `functionCall` IDs Gemini returns and sends them back in the matching `functionResponse`.
- **WhatsApp Messages No Longer Time Out While Queued:** `WhatsAppChannel`'s `llm_timeout_seconds` used to cover both the wait behind earlier turns of the same session and the turn itself, so valid messages sent in a burst, or queued behind a slow turn, timed out without running. The wait and the turn now have separate limits: the new `queue_timeout_seconds` bounds the wait, and `llm_timeout_seconds` bounds only the turn, starting once it holds the session. A message whose wait expires gets `fallback_message` and its turn never runs, even if the session becomes free later. Queue and turn timeouts are logged differently.
- **Duplicate Agent Names Fail Closed (Issue #123):** Building a `Runner` whose handoff tree contains two distinct `Agent` instances with the same name now raises `ValueError`. `_resolve_handoff_target()` returns the peer the current agent declares (not the first registered object under that name), so same-named specialists on different branches no longer steal each other's handoffs. Late `add_handoff()` replacements update the registry atomically after validation; a replacement that would leave two distinct same-named agents reachable, collide in the replacement tree, or overwrite a `register_agent()` registration still raises without partially mutating the registry. Failed `register_agent()` trees roll back. Shared DAG instances keep working.
- **WhatsApp Reactions And Malformed Nested Shapes (Issue #121):** `reaction` and `system` webhook events are acknowledged with no outbound reply. Meta Cloud API payloads that include `entry` never fall through to the generic extractor, so ignored events cannot be revived by top-level `from` / `text` fields. Signed payloads whose nested `entry` / `changes` / `value` / `messages` / `text` nodes have an unexpected type are skipped with a warning and HTTP 200, so Meta does not retry them as failures.
- **SQLite Session Integrity Under Concurrent Requests (Issue #31):** Turns for the same `session_id` on a `SQLiteSessionStore` now share one lock per event loop and run one at a time, in the order they start waiting for it; previously every request got its own lock and turns ran in parallel. Each turn reloads the history, active agent and metadata committed by earlier turns after acquiring the lock. A failed or cancelled turn restores the state it started from, including messages pruned during the turn (#34). It no longer deletes or overwrites what someone else committed meanwhile (another turn's messages, or agent and metadata changed through another context): it then reverts only its own changes, in a single `BEGIN IMMEDIATE` transaction, and logs a warning. The lock registry holds locks weakly, so it does not grow with the number of sessions, and it creates separate locks per event loop, so successive `Runner.run()` calls keep working.
- **Deleted Sessions No Longer Get Orphan Messages:** a turn that is running when its SQLite session is deleted fails with the new `SessionDeletedError` instead of storing its reply without a session (rows that later reappeared when the id was reused). A turn waiting for a deleted session starts a new, empty one; deleted content is never restored. Deleting a session erases its data but does not block its id.

### Added
- **`Runner.run_async(..., turn_timeout=...)`:** optional keyword-only limit, in seconds, for the turn itself. It starts once the turn holds the session lock, so waiting behind earlier turns does not count. On expiry the turn is cancelled, the session is restored to its state before the turn, and `TimeoutError` (`asyncio.TimeoutError`) is raised. It must be a positive finite number; the default, `None`, keeps the previous behaviour.
- **`WhatsAppChannel(queue_timeout_seconds=120.0)`:** limit for the wait behind earlier turns of the same session. Must be a positive finite number.
- **`Runner.spawn_turn()`:** starts a turn as an independent task, outside any running turn. Tools use it to queue a follow-up turn for their own session; it runs after the current turn ends and must not be awaited inside it. The runner keeps the task alive until it finishes.
- **`ReentrantTurnError`** (subclass of `ChatFlowError` and `RuntimeError`) and **`SessionDeletedError`** (subclass of `ChatFlowError`), exported from `chatflow_agent`.

### Changed
- **`GeminiEngine` Retry Delays Are Bounded by `max_delay` (Issue #116):** on a 429, 500, 502, 503 or 504 response, a `Retry-After` header (seconds, or an HTTP-date in any of the three formats HTTP allows) replaces the exponential backoff, but the wait never exceeds `max_delay` (10 s by default), however long the server asks for, even a value too large for a number. A malformed value (for example `soon`, `-5`, `nan` or an invalid date) is ignored and the exponential backoff applies.
- **`llm_timeout_seconds` Bounds Only the Turn:** in `WhatsAppChannel` it no longer includes the wait for the session. A reply can now take up to `queue_timeout_seconds + llm_timeout_seconds` (240 s with the defaults). A custom `dispatch_async()` that does not call `Runner.run_async()` is bounded by that sum plus one second instead of `llm_timeout_seconds`. `None` and `inf` still set no turn limit and zero or negative values still time every turn out; `nan` now sets no limit (it used to time turns out at once or, on Python 3.13, stop the event loop). Zero, negative and `nan` values log a warning.
- **WhatsApp Timeout Logs:** the single `LLM dispatch timed out ...` error is replaced by `Session busy ...` (queue limit), `Turn timed out ...` (turn limit), `Dispatch backstop expired ...` (a `dispatch_async()` that never reached `Runner.run_async()`) and `TimeoutError raised while dispatching ...` (a `TimeoutError` raised by the provider or other code). Update alerts that match the old text.
- **Queued Turns Wait:** Concurrent requests for the same SQLite session are now processed sequentially, so a queued request waits for the previous turn.
- **Turns Started From Inside a Turn Raise:** Calling `Runner.run_async()` or `Runner.run()` for a session from code started by a running turn of that same session (a tool, or any task or thread it starts: `create_task`, `gather`, `wait_for`, `TaskGroup`, `asyncio.to_thread`) now raises `ReentrantTurnError` immediately. Previously these calls waited forever (in-memory store), or ran unserialized alongside the outer turn (SQLite). Use `Runner.spawn_turn()` for follow-up turns. This applies to every session store, including custom ones.
- **New Method:** `SessionContext.refresh()` reloads committed state; it is a no-op for in-memory sessions and is called by `Runner` after acquiring the session lock.

### Known Issues
- **Concurrency Scope:** Serialization is not provided across store instances on the same file, across processes, or across threads running their own event loops.
- **Nested Turns Outside the Turn's Tasks:** a tool that waits for work done by a task that was not started from the turn (for example, a worker created earlier) can still deadlock if that work runs a turn of the same session. Awaiting a task returned by `spawn_turn()` inside the turn that spawned it deadlocks as well.
- **Reset and Delete During a Turn:** `clear_history()` and `delete()` are not coordinated with running turns beyond the rollback and `SessionDeletedError` guarantees above (tracked separately).

## [0.3.4] - 2026-10-23

### Security
- **Stricter API key scoping for custom endpoints:** API keys read from environment variables are now sent only to the official endpoint of the provider they belong to. Configurations that use a custom `base_url` (proxies, gateways, other providers) must pass `api_key` explicitly; `api_key=""` sends no credential. `OpenAIEngine` and `AnthropicEngine` no longer follow HTTP redirects, even with an injected client configured to. See GHSA-mcgm-q29p-3984.

## [0.3.3] - 2026-10-19

### Security
- **Strict Local Endpoint Hostname Validation:** `OpenAIEngine` now validates `base_url` structure, HTTP/HTTPS scheme, port range, and parsed hostname via `urllib.parse.urlsplit` when identifying local endpoints (`localhost`, `127.0.0.1`, `::1`, and local bind compatibility address `0.0.0.0`), replacing substring matching (Issue #74).

## [0.3.2] - 2026-09-29

### Security
- **Tool Execution Error Sanitization (SEC-02):** Sanitized tool execution errors in `Runner` before exposing them to the LLM, returning a generic error with a unique `trace_id` UUID and logging the original traceback internally to prevent leakage of internal stack traces or environment variables (Issue #3).
- **Mask Phone Numbers in Exception Logs (SEC-03):** Automatically mask phone numbers (`_mask_phone`) in exception and fallback delivery logs in `WhatsAppChannel` for privacy compliance (Issue #5).
- **Admin-Only /reset in Telegram Groups (SEC-04):** Restricted the `/reset` command in Telegram group and supergroup chats strictly to chat creators and administrators (Issue #7).
- **Disabled FastAPI OpenAPI / Docs Endpoints by Default (SEC-05):** `WhatsAppChannel` now sets `docs_url=None`, `redoc_url=None`, and `openapi_url=None` by default to prevent internal schema and route exposure on public webhook endpoints. Interactive documentation can be optionally enabled for development via `enable_docs=True` (Issue #9).
- **Reverse Proxy Rate Limiting Recommendation (SEC-06):** Added production deployment guidance for placing Uvicorn behind a reverse proxy (e.g., Nginx with `limit_req_zone` or Cloudflare Rate Limiting) to protect `/webhook` and `/health` endpoints against flood attacks and CPU exhaustion from unauthenticated requests, avoiding in-app rate limiting that could drop legitimate bursts of Meta webhooks (Issue #10).
- **API Key Leakage Prevention on Custom Base URLs (SEC-07):** `OpenAIEngine` now prevents forwarding global environment credentials (`OPENAI_API_KEY`) to untrusted or custom remote endpoints. When a custom `base_url` (not matching official presets or localhost/127.0.0.1) is provided without an explicit `api_key`, a `ValueError` is raised, requiring an explicit `api_key` (or `api_key=""` if unauthenticated) (Issue #11).
- **Outbound Message Splitting in Telegram (SEC-08):** `TelegramChannel` now splits outbound responses exceeding Telegram's 4,096-character limit into progressive chunks using `split_message()`, preventing systematic delivery failures and error fallbacks on long LLM responses. `split_message()` now resides in `chatflow_agent.channels.base` and is re-exported from `whatsapp` for backwards compatibility (Issue #12).
- **Async vs Sync Tool Execution Clarification and Guard (SEC-09):** Documented that asynchronous tools must always be invoked via `await tool.execute_async(**kwargs)` in asynchronous environments (FastAPI, Runners, Telegram handlers). Calling `tool.execute()` on an async tool from within an active running event loop now immediately raises a clear `ToolExecutionError` explaining why synchronous re-entry of the running event loop is unsupported and directing the caller to `execute_async()` (Issue #13).
- **Retry and Backoff in AnthropicEngine (SEC-10):** `AnthropicEngine` now includes exponential backoff with jitter and retry handling for transient API errors (HTTP 429, 500, 502, 503, 504, 529 Overloaded, and connection errors `httpx.ConnectError`, `httpx.ConnectTimeout`). Honors the `Retry-After` header when provided (Issue #14).

## [0.3.1] - 2026-09-25

### Security
- **WhatsApp Webhook Request Body Size Limit (SEC-01):** Fixed vulnerability where inbound `POST /webhook` requests did not validate the payload size prior to buffering, enabling potential Denial of Service (DoS via Out-of-Memory / OOM). The endpoint now validates the `Content-Length` header before reading the body, enforcing a configurable limit (default: 1 MB via `DEFAULT_MAX_BODY_SIZE`) and rejecting oversized payloads with HTTP 413, missing headers with HTTP 411, and malformed headers with HTTP 400.

## [0.3.0] - 2026-09-21

### Security
- **WhatsApp Webhook Signature Verification Enforced (SEC-01):** Fixed vulnerability where inbound POST `/webhook` requests were processed without verifying Meta's `X-Hub-Signature-256` HMAC header, allowing unauthenticated attackers to forge messages. `WhatsAppChannel` now enforces a strict fail-closed policy: `app_secret` is required by default, validating requests with constant-time `hmac.compare_digest` and rejecting invalid or missing signatures with HTTP 403.
- **Timing Attack Mitigation (SEC-02):** GET `/webhook` challenge verification now uses constant-time `hmac.compare_digest` for verify tokens.
- **Credential Masking in Logs:** Sensitive tokens, app secrets, and credentials are automatically redacted across all logging handlers.
- **Fail-Closed Starter Templates:** Example scripts (`whatsapp_service.py`, `whatsapp_persistent_bot.py`) raise `RuntimeError` if `WHATSAPP_APP_SECRET` is unset rather than falling back to an insecure development secret.

### ⚠ BREAKING CHANGES
1. **`WhatsAppChannel` requires `app_secret` by default (SEC-01):** Omitting `app_secret` raises `ValueError` on startup unless `verify_signature=False` is explicitly set (local testing only).
2. **Webhook response payload changed (BUS-01):** POST `/webhook` returns `{"status":"success","sender":"...","messages_queued": N}`. Inference and messaging are now dispatched asynchronously in background tasks.
3. **`max_turns` counts complete user conversation turns (DATA-01):** `max_turns` now enforces complete user dialogue turns (user message + model responses + tool results) rather than individual message objects.

### Added / Fixed
- **Background Webhook Dispatch (BUS-01):** Webhooks respond HTTP 200 to Meta immediately, preventing timeout retries on slow LLM calls.
- **At-Most-Once Deduplication (ARCH-01):** Atomic message ID tracking in SQLite with an 8-day TTL (`DEFAULT_DEDUP_TTL_HOURS = 192`) safely discards Meta retries.
- **Multi-Message Batch Extraction:** Webhooks with multiple message payloads are unpacked and dispatched independently.
- **Selective Outbound Retry Backoff with Jitter:** Outbound HTTP retries are strictly limited to `ConnectError`, `ConnectTimeout`, and HTTP 429, honoring the `Retry-After` header.
- **Meta 24h Window Expiration Handling:** Detects Meta error code `131047`, executes optional `on_24h_window_expired` callback safely (try/except), and logs masked recipient numbers.
- **Outbound Message Splitting:** Messages exceeding Meta's 4,096-character limit are automatically split across paragraph and sentence boundaries into sequential chunks.
- **Mid-Turn Failure Rollback:** Automatically reverts dialogue history to the last consistent turn in memory and SQLite on mid-turn engine failure and guarantees fallback delivery to WhatsApp.
- **Turn-Based Pruning & Memory Integrity (DATA-01):** FIFO pruning preserves system prompts and atomic tool call/result pairs.
- **SQLite Concurrency & WAL Mode:** Configured `PRAGMA busy_timeout = 10000` and `PRAGMA journal_mode = WAL` for atomic multi-reader concurrency. Existing databases transition automatically without schema recreation.
- **Documentation Parity:** 100% bilingual parity across README and CHANGELOG.

### Migration Guide
If you are upgrading from v0.2.0:
1. **Set `WHATSAPP_APP_SECRET`:** Add your Meta App Secret to your environment variables and pass it to `WhatsAppChannel(..., app_secret=os.environ["WHATSAPP_APP_SECRET"])`. For local testing without signature validation, explicitly pass `verify_signature=False`.
2. **Update Webhook Payload Parsing:** If any internal client monitored the POST `/webhook` response body, update it to expect `{"status": "success", "messages_queued": N}` instead of the synchronous `reply` field.
3. **Reduce `max_turns` Setting:** If your configuration previously used inflated `max_turns` values to account for individual messages (e.g. `max_turns=40` which held ~10 real turns), reduce it to the number of full turns you actually want to retain (e.g., lower to `max_turns=10`). Keeping the old number will retain roughly 4x more context than before.

---

## Español

## [Unreleased]

### Corregido
- **Los Mensajes de WhatsApp Ya No Vencen Mientras Esperan:** `llm_timeout_seconds` de `WhatsAppChannel` cubría tanto la espera detrás de turnos anteriores de la misma sesión como el turno en sí, por lo que mensajes válidos enviados en ráfaga, o encolados detrás de un turno lento, vencían sin ejecutarse. La espera y el turno ahora tienen límites separados: el nuevo `queue_timeout_seconds` limita la espera, y `llm_timeout_seconds` limita solo el turno, desde que este tiene la sesión. Un mensaje cuya espera vence recibe `fallback_message` y su turno nunca se ejecuta, aunque la sesión se libere después. Los vencimientos de espera y de turno se registran con mensajes distintos.
- **Integridad de Sesiones SQLite ante Solicitudes Concurrentes (Issue #31):** Los turnos de un mismo `session_id` en un `SQLiteSessionStore` ahora comparten un candado por event loop y se ejecutan de a uno, en el orden en que empiezan a esperarlo; antes cada solicitud obtenía su propio candado y los turnos corrían en paralelo. Cada turno vuelve a leer el historial, el agente activo y la metadata confirmados por los turnos anteriores después de adquirir el candado. Un turno fallido o cancelado restaura el estado con el que empezó, incluidos los mensajes podados durante el turno (#34). Ya no borra ni sobrescribe lo que otro confirmó mientras tanto (mensajes de otro turno, o agente y metadata cambiados desde otro contexto): en ese caso solo revierte sus propios cambios, en una única transacción `BEGIN IMMEDIATE`, y registra una advertencia. El registro de candados los mantiene con referencias débiles, por lo que no crece con la cantidad de sesiones, y crea candados separados por event loop, por lo que las llamadas sucesivas a `Runner.run()` siguen funcionando.
- **Las Sesiones Borradas Ya No Reciben Mensajes Huérfanos:** un turno que se está ejecutando cuando se borra su sesión SQLite falla con la nueva `SessionDeletedError` en lugar de guardar su respuesta sin sesión (filas que reaparecían al reutilizar el id). Un turno que esperaba una sesión borrada inicia una sesión nueva y vacía; el contenido borrado nunca se restaura. Borrar una sesión elimina sus datos, pero no bloquea su id.

### Agregado
- **`Runner.run_async(..., turn_timeout=...)`:** límite opcional, solo por nombre y en segundos, para el turno en sí. Empieza cuando el turno tiene el candado de la sesión, por lo que la espera detrás de turnos anteriores no cuenta. Al vencer, el turno se cancela, la sesión vuelve al estado previo al turno y se lanza `TimeoutError` (`asyncio.TimeoutError`). Debe ser un número positivo y finito; el valor por defecto, `None`, conserva el comportamiento anterior.
- **`WhatsAppChannel(queue_timeout_seconds=120.0)`:** límite para la espera detrás de turnos anteriores de la misma sesión. Debe ser un número positivo y finito.
- **`Runner.spawn_turn()`:** inicia un turno como tarea independiente, fuera de cualquier turno en curso. Las tools lo usan para encolar un turno posterior de su propia sesión; se ejecuta cuando termina el turno actual y no debe esperarse dentro de él. El runner mantiene viva la tarea hasta que termina.
- **`ReentrantTurnError`** (subclase de `ChatFlowError` y `RuntimeError`) y **`SessionDeletedError`** (subclase de `ChatFlowError`), exportadas desde `chatflow_agent`.

### Cambiado
- **`llm_timeout_seconds` Limita Solo el Turno:** en `WhatsAppChannel` ya no incluye la espera de la sesión. Una respuesta ahora puede tardar hasta `queue_timeout_seconds + llm_timeout_seconds` (240 s con los valores por defecto). Un `dispatch_async()` personalizado que no llama a `Runner.run_async()` queda limitado por esa suma más un segundo, en lugar de `llm_timeout_seconds`. `None` e `inf` siguen sin poner límite al turno y los valores cero o negativos siguen haciendo vencer cada turno; `nan` ahora no pone límite (antes podía hacer vencer los turnos de inmediato o, en Python 3.13, detener el event loop). Los valores cero, negativos y `nan` registran una advertencia.
- **Logs de Timeout de WhatsApp:** el error único `LLM dispatch timed out ...` se reemplaza por `Session busy ...` (límite de espera), `Turn timed out ...` (límite del turno), `Dispatch backstop expired ...` (un `dispatch_async()` que nunca llegó a `Runner.run_async()`) y `TimeoutError raised while dispatching ...` (un `TimeoutError` del proveedor u otro código). Actualizá las alertas que busquen el texto anterior.
- **Los Turnos en Espera Esperan:** Las solicitudes concurrentes para una misma sesión SQLite ahora se procesan en secuencia, por lo que una solicitud encolada espera al turno anterior.
- **Los Turnos Iniciados Desde un Turno Fallan:** Llamar `Runner.run_async()` o `Runner.run()` para una sesión desde código iniciado por un turno en curso de esa misma sesión (una tool, o cualquier tarea o thread que inicie: `create_task`, `gather`, `wait_for`, `TaskGroup`, `asyncio.to_thread`) ahora lanza `ReentrantTurnError` de inmediato. Antes esas llamadas esperaban para siempre (store en memoria) o corrían sin serializar junto al turno externo (SQLite). Usa `Runner.spawn_turn()` para turnos posteriores. Aplica a todos los stores de sesión, incluidos los personalizados.
- **Nuevo Método:** `SessionContext.refresh()` recarga el estado confirmado; no hace nada en sesiones en memoria y `Runner` lo llama después de adquirir el candado de la sesión.

### Problemas Conocidos
- **Alcance de la Concurrencia:** No se serializan turnos entre instancias del store sobre el mismo archivo, entre procesos ni entre threads con su propio event loop.
- **Turnos Anidados Fuera de las Tareas del Turno:** una tool que espera trabajo hecho por una tarea que no fue iniciada desde el turno (por ejemplo, un worker creado antes) todavía puede quedar bloqueada si ese trabajo ejecuta un turno de la misma sesión. Esperar, dentro del turno que la creó, una tarea devuelta por `spawn_turn()` también la bloquea.
- **Reset y Delete Durante un Turno:** `clear_history()` y `delete()` no se coordinan con los turnos en curso más allá de las garantías de rollback y de `SessionDeletedError` descritas arriba (se trata por separado).

## [0.3.4] - 2026-10-23

### Seguridad
- **Alcance más estricto de las API keys en endpoints personalizados:** las API keys leídas de variables de entorno ahora se envían solo al endpoint oficial del proveedor al que pertenecen. Las configuraciones con un `base_url` personalizado (proxies, gateways, otros proveedores) deben pasar `api_key` explícitamente; `api_key=""` no envía ninguna credencial. `OpenAIEngine` y `AnthropicEngine` ya no siguen redirects HTTP, aunque el cliente inyectado esté configurado para hacerlo. Ver GHSA-mcgm-q29p-3984.

## [0.3.3] - 2026-10-19

### Seguridad
- **Validación Estricta de Hostname en Endpoints Locales:** `OpenAIEngine` ahora valida la estructura de `base_url`, el esquema HTTP/HTTPS, el rango de puerto y el hostname real mediante `urllib.parse.urlsplit` al identificar endpoints locales (`localhost`, `127.0.0.1`, `::1` y la dirección de compatibilidad local `0.0.0.0`), reemplazando la coincidencia por subcadenas (Issue #74).

## [0.3.2] - 2026-09-29

### Seguridad
- **Sanitización de Errores de Tools (SEC-02):** Se sanitizaron los errores de ejecución de herramientas en `Runner` antes de exponerlos al LLM, retornando un mensaje genérico con un UUID `trace_id` único y registrando el traceback internamente para prevenir la fuga de stack traces o variables de entorno (Issue #3).
- **Enmascaramiento de Teléfonos en Logs (SEC-03):** Se enmascaran automáticamente los números telefónicos (`_mask_phone`) en logs de excepciones y fallbacks en `WhatsAppChannel` para cumplimiento de privacidad (Issue #5).
- **Comando /reset Exclusivo para Administradores en Telegram (SEC-04):** Se restringió la ejecución del comando `/reset` en grupos y supergrupos de Telegram exclusivamente a creadores y administradores (Issue #7).
- **Desactivación de Endpoints OpenAPI / Docs de FastAPI por Defecto (SEC-05):** `WhatsAppChannel` ahora configura `docs_url=None`, `redoc_url=None` y `openapi_url=None` por defecto para prevenir la exposición de rutas y esquemas internos en endpoints públicos de webhook. La documentación interactiva puede habilitarse opcionalmente para desarrollo mediante `enable_docs=True` (Issue #9).
- **Recomendación de Rate Limiting con Reverse Proxy (SEC-06):** Se añadió documentación de despliegue en producción recomendando ubicar Uvicorn detrás de un reverse proxy (como Nginx con `limit_req_zone` o Cloudflare Rate Limiting) para proteger los endpoints `/webhook` y `/health` contra ataques de inundación y agotamiento de CPU por peticiones no autenticadas, evitando rate limiting a nivel aplicación que pudiera descartar ráfagas legítimas de webhooks de Meta (Issue #10).
- **Prevención de Fuga de Claves API en Base URLs Personalizadas (SEC-07):** `OpenAIEngine` ahora previene el reenvío de credenciales globales de entorno (`OPENAI_API_KEY`) a endpoints remotos personalizados o no oficiales. Cuando se proporciona un `base_url` personalizado (que no coincide con los presets oficiales ni con localhost/127.0.0.1) sin una `api_key` explícita, se levanta un `ValueError`, exigiendo una `api_key` explícita (o `api_key=""` si no requiere autenticación) (Issue #11).
- **División de Mensajes Salientes en Telegram (SEC-08):** `TelegramChannel` ahora divide las respuestas salientes que superan el límite de 4.096 caracteres de Telegram en fragmentos progresivos usando `split_message()`, previniendo fallos sistemáticos de entrega y respuestas de error en respuestas extensas de LLM. `split_message()` ahora reside en `chatflow_agent.channels.base` y se reexporta desde `whatsapp` para retrocompatibilidad (Issue #12).
- **Clarificación y Validación de Ejecución de Tools Asíncronas vs Síncronas (SEC-09):** Se documentó que las herramientas asíncronas deben invocarse siempre mediante `await tool.execute_async(**kwargs)` en entornos asíncronos (FastAPI, Runners, handlers de Telegram). Ejecutar `tool.execute()` sobre una herramienta asíncrona dentro de un bucle de eventos activo ahora levanta inmediatamente un `ToolExecutionError` claro explicando que la reentrada síncrona al bucle de eventos no está soportada e indicando el uso de `execute_async()` (Issue #13).
- **Reintentos y Backoff Exponencial en AnthropicEngine (SEC-10):** `AnthropicEngine` ahora incluye reintentos automáticos con backoff exponencial y jitter ante errores transitorios de la API (HTTP 429, 500, 502, 503, 504, 529 Sobrecargado, y errores de conexión `httpx.ConnectError`, `httpx.ConnectTimeout`). Respeta la cabecera `Retry-After` cuando está presente (Issue #14).

## [0.3.1] - 2026-09-25

### Seguridad
- **Límite de Tamaño en Request Body del Webhook de WhatsApp (SEC-01):** Se corrigió una vulnerabilidad donde las peticiones entrantes `POST /webhook` no validaban el tamaño del payload antes de cargarlo en memoria, permitiendo ataques de Denegación de Servicio por agotamiento de memoria (DoS / OOM). Ahora se valida el encabezado `Content-Length` antes de leer el cuerpo, aplicando un límite configurable (por defecto 1 MB mediante `DEFAULT_MAX_BODY_SIZE`) y rechazando payloads excesivos con HTTP 413, encabezados ausentes con HTTP 411 y encabezados malformados con HTTP 400.

## [0.3.0] - 2026-09-21

### Seguridad
- **Verificación Obligatoria de Firma en Webhook de WhatsApp (SEC-01):** Se corrigió una vulnerabilidad en `WhatsAppChannel` donde las peticiones POST `/webhook` entrantes se procesaban sin validar la cabecera HMAC `X-Hub-Signature-256` de Meta, permitiendo a un atacante enviar mensajes falsificados suplantando números de usuario. `WhatsAppChannel` ahora aplica una política estricta *fail-closed*: exige `app_secret` por defecto, valida la firma con `hmac.compare_digest` en tiempo constante y rechaza cualquier petición sin firma válida con HTTP 403.
- **Mitigación de Ataques de Temporización (SEC-02):** La verificación del handshake GET `/webhook` utiliza `hmac.compare_digest` en tiempo constante para validar el token.
- **Enmascaramiento de Credenciales en Logs:** Tokens, secretos de aplicación y números telefónicos se enmascaran automáticamente en toda la salida de depuración y logs.
- **Plantillas con Validación Estricta (Fail-Closed):** Los ejemplos (`whatsapp_service.py` y `whatsapp_persistent_bot.py`) lanzan `RuntimeError` si falta `WHATSAPP_APP_SECRET` en vez de usar un secreto de desarrollo inseguro por defecto.

### ⚠ CAMBIOS CON RUPTURA DE COMPATIBILIDAD (BREAKING CHANGES)
1. **`WhatsAppChannel` exige `app_secret` por defecto (SEC-01):** Si se omite `app_secret`, se lanza `ValueError` al arrancar el servidor a menos que se configure explícitamente `verify_signature=False` (solo para pruebas locales).
2. **Cambio en el cuerpo de respuesta del Webhook (BUS-01):** El POST `/webhook` ahora responde `{"status":"success","sender":"...","messages_queued": N}`. La inferencia y el envío saliente se procesan asíncronamente en segundo plano.
3. **`max_turns` ahora cuenta turnos completos de usuario (DATA-01):** `max_turns` mide ciclos conversacionales completos (mensaje del usuario + respuestas del modelo + herramientas asociadas) en vez de objetos de mensaje individuales.

### Agregado / Corregido
- **Despacho Asíncrono de Webhooks (BUS-01):** Responde HTTP 200 a Meta inmediatamente, evitando tormentas de reintentos por latencia del LLM.
- **Deduplicación At-Most-Once (ARCH-01):** Registro atómico por `wamid` en SQLite con TTL de 8 días (`DEFAULT_DEDUP_TTL_HOURS = 192`) que descarta reintentos duplicados de Meta.
- **Extracción de Lotes Multi-Mensaje:** Procesamiento y despacho independiente para webhooks que contienen múltiples mensajes simultáneos.
- **Reintentos Salientes Selectivos con Jitter:** Limitados estrictamente a errores de conexión de red y HTTP 429 con jitter, respetando el encabezado `Retry-After`.
- **Manejo de Ventana de 24h de Meta:** Detección del código de error `131047`, ejecución protegida con try/except del hook `on_24h_window_expired` y log con teléfono enmascarado.
- **División Automática de Mensajes Salientes:** Fragmentación automática de mensajes que exceden el límite de 4.096 caracteres de Meta respetando párrafos y oraciones.
- **Rollback ante Fallas a Mitad de Turno:** Reversión automática del historial conversacional al último turno consistente en memoria y SQLite ante fallas del LLM, garantizando la entrega del mensaje de fallback.
- **Integridad de Memoria y Poda por Turnos (DATA-01):** Poda FIFO que preserva el prompt del sistema y mantiene indivisibles los pares de llamadas y resultados de herramientas.
- **Concurrencia SQLite y Modo WAL:** `PRAGMA busy_timeout = 10000` y `PRAGMA journal_mode = WAL` para concurrencia multi-lector sin contención. Bases existentes transicionan automáticamente sin recreación de tablas.
- **Paridad Bilingüe en Documentación:** Paridad 100% en inglés y español en README y CHANGELOG.

### Guía de Migración
Si ya instalaste o integraste la versión v0.2.0:
1. **Configurá `WHATSAPP_APP_SECRET`:** Agregá tu App Secret de Meta a las variables de entorno y pasalo a `WhatsAppChannel(..., app_secret=os.environ["WHATSAPP_APP_SECRET"])`. Si estás testeando localmente sin firma, pasá explícitamente `verify_signature=False`.
2. **Actualizá el parseo de respuesta del Webhook:** Si tenías clientes o scripts esperando el campo sincrónico `reply` en el JSON de respuesta de POST `/webhook`, actualizalos para esperar `{"status": "success", "messages_queued": N}`.
3. **Reducí el valor de `max_turns`:** Si tu configuración anterior inflaba `max_turns` para contar mensajes sueltos (por ejemplo, `max_turns=40` para retener unos ~10 turnos reales), reducilo al número real de turnos deseados (por ejemplo, bajalo a `max_turns=10`). Mantener el número anterior retendrá aproximadamente 4 veces más contexto que antes.
