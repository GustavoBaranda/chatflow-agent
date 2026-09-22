# Changelog / Registro de Cambios

All notable changes to this project will be documented in this file.  
*Todos los cambios notables en este proyecto seran documentados en este archivo.*

Format / Formato: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning / Versionado: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

[English](#english) | [Español](#español)

---

## English

## [Unreleased] — branch: fix/production-hardening

### ⚠ BREAKING CHANGES

#### `WhatsAppChannel` requires `app_secret` by default (SEC-01)

`WhatsAppChannel` now raises `ValueError` at startup if `app_secret` is not
provided and `verify_signature` is not explicitly set to `False`.

**Before (v0.1.x):**
```python
# app_secret was optional and silently ignored
channel = WhatsAppChannel(verify_token="...", access_token="...")
```

**After:**
```python
# app_secret is required; omitting it raises ValueError
channel = WhatsAppChannel(
    verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"],
    access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
    app_secret=os.environ["WHATSAPP_APP_SECRET"],   # <-- now required
    phone_number_id=os.environ["WHATSAPP_PHONE_ID"],
)
# For local development/testing only:
channel = WhatsAppChannel(
    verify_token="...",
    verify_signature=False,   # emits SECURITY WARNING in logs
)
```

#### Webhook response body changed (BUS-01)

The POST `/webhook` response body no longer contains `reply` or `active_agent`
fields. Processing is now asynchronous.

**Before:** `{"status":"success","sender":"...","reply":"...","active_agent":"..."}`  
**After:**  `{"status":"success","sender":"...","messages_queued": N}`

#### `max_turns` now counts user conversation turns, not individual messages (DATA-01)

Previously `max_turns` counted individual `Message` objects (user + model + tool
messages). Now it counts complete user conversation turns (each turn = one user
message plus its associated model/tool responses). A setting of `max_turns=20`
now means 20 user-initiated turns, not 20 messages.

---

### Security

- **HMAC-SHA256 Webhook Verification (SEC-01):** Reads raw request body before JSON
  decoding; validates `X-Hub-Signature-256` with constant-time `hmac.compare_digest`.
  Fails with HTTP 403 if the header is missing or invalid.
- **Constant-Time Verify Token Check (SEC-02):** GET `/webhook` challenge handler uses
  `hmac.compare_digest` to prevent timing attacks on the verify token.
- **Credential Masking in Logs:** Sensitive tokens, app secrets, and credentials are
  automatically masked in logging output.
- **Fail-Closed Starter Examples:** Starter templates (`examples/whatsapp_service.py` and
  `examples/whatsapp_persistent_bot.py`) raise `RuntimeError` if `WHATSAPP_APP_SECRET`
  is missing instead of defaulting to an insecure dev secret.

### Performance & Reliability

- **Async Webhook Processing via `BackgroundTasks` (BUS-01):** Returns HTTP 200 to Meta
  immediately; agent inference and outbound messaging run in a background task.
  Prevents Meta retry storms caused by slow LLM responses (Meta times out at 15 s).
- **At-Most-Once Dedup by `wamid` (ARCH-01):** Message IDs are atomically recorded in
  `chatflow_processed_messages` before the background task is enqueued. Duplicate
  deliveries from Meta retries (up to 7 days per Meta docs) are safely discarded.
  TTL defaults to 8 days (`DEFAULT_DEDUP_TTL_HOURS = 192`) to prevent unbounded growth.
- **Multi-Message Batch Extraction:** Webhook payloads with multiple messages in
  `entry[].changes[].value.messages` are fully extracted; each message is individually
  deduplicated and independently dispatched.
- **Anti-500 Shield & Fallback Delivery:** Background task wraps the full flow in resilient
  error handling; delivers a user-facing fallback message on failures without HTTP 500 crashes.
- **Selective Outbound Retry Backoff with Jitter:** Outbound HTTP retries to Meta Cloud API
  are strictly limited to network connection errors (`ConnectError`, `ConnectTimeout`) and
  HTTP `429` rate limits with exponential backoff and jitter. Reads and write timeouts
  do not retry to prevent duplicate user deliveries. Respects `Retry-After` response header.
- **Meta 24h Window Expiration Handling:** Detects Meta error code `131047` ("message
  sent outside 24h window"), safely executes optional `on_24h_window_expired` callback
  wrapped in try/except, and logs masked recipient telephone numbers.
- **Outbound Message Splitting:** Messages exceeding Meta's 4,096-character limit are
  automatically split across paragraph and sentence boundaries into sequential chunks.
- **Mid-Turn Failure Rollback:** Automatically rolls back dialogue history to the last
  consistent turn in memory and SQLite on mid-turn engine failure, ensuring fallback
  dispatch without orphaned partial turns.

### Storage & Memory Integrity

- **Turn-Based Pruning (DATA-01):** `max_turns` now counts complete user conversation
  turns. The pruning algorithm preserves the system prompt and guarantees that
  `tool_calls` / `tool_results` pairs are never severed.
- **SQLite Schema Versioning (MAINT-01):** `PRAGMA user_version = 1` and table
  `chatflow_processed_messages` for persistent deduplication with TTL purge.
- **SQLite Concurrency & Seamless WAL:** `PRAGMA busy_timeout = 10000` + `PRAGMA journal_mode=WAL`
  to prevent lock contention. Existing SQLite databases transition automatically to WAL
  mode without requiring manual schema migrations or data recreation.
- **Turn Rollback Semantics:** Documented that turn rollback reverts conversational dialogue
  history; external side effects of already executed tools are not rolled back.

### Documentation

- **100% Bilingual Parity:** Comprehensive English and Spanish documentation across
  README.md and CHANGELOG.md.

---

## Español

## [No publicado] — rama: fix/production-hardening

### ⚠ CAMBIOS CON RUPTURA DE COMPATIBILIDAD (BREAKING CHANGES)

#### `WhatsAppChannel` requiere `app_secret` por defecto (SEC-01)

`WhatsAppChannel` ahora lanza `ValueError` al inicializarse si no se proporciona
`app_secret` y no se ha configurado explicitamente `verify_signature=False`.

**Antes (v0.1.x):**
```python
# app_secret era opcional y se ignoraba en silencio
channel = WhatsAppChannel(verify_token="...", access_token="...")
```

**Despues:**
```python
# app_secret es obligatorio; omitirlo lanza ValueError
channel = WhatsAppChannel(
    verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"],
    access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
    app_secret=os.environ["WHATSAPP_APP_SECRET"],   # <-- ahora obligatorio
    phone_number_id=os.environ["WHATSAPP_PHONE_ID"],
)
# Solo para desarrollo local y pruebas:
channel = WhatsAppChannel(
    verify_token="...",
    verify_signature=False,   # emite ADVERTENCIA DE SEGURIDAD en logs
)
```

#### Cambio en el cuerpo de respuesta del Webhook (BUS-01)

El cuerpo de respuesta de POST `/webhook` ya no contiene los campos `reply` ni
`active_agent`. El procesamiento ahora es completamente asincrono.

**Antes:** `{"status":"success","sender":"...","reply":"...","active_agent":"..."}`  
**Despues:** `{"status":"success","sender":"...","messages_queued": N}`

#### `max_turns` ahora cuenta turnos completos de usuario, no mensajes individuales (DATA-01)

Anteriormente `max_turns` contaba objetos `Message` individuales (mensajes de usuario + modelo + tools).
Ahora cuenta turnos completos de conversacion del usuario (cada turno = un mensaje de usuario mas
sus respuestas asociadas de modelo y herramientas). Una configuracion de `max_turns=20` ahora
significa 20 turnos iniciados por el usuario, no 20 mensajes sueltos.

---

### Seguridad

- **Verificacion de Webhooks con HMAC-SHA256 (SEC-01):** Lee el cuerpo binario crudo antes de la
  deserializacion JSON; valida la firma `X-Hub-Signature-256` con `hmac.compare_digest` en tiempo constante.
  Rechaza con HTTP 403 si la cabecera falta o no coincide.
- **Validacion de Verify Token en Tiempo Constante (SEC-02):** El endpoint GET `/webhook` utiliza
  `hmac.compare_digest` para neutralizar ataques de temporizacion sobre el token de verificacion.
- **Enmascaramiento de Credenciales en Logs:** Tokens sensibles, secretos y credenciales son
  enmascarados automaticamente en los registros de auditoria y logs de depuracion.
- **Ejemplos con Validacion Estricta (Fail-Closed):** Las plantillas de ejemplo (`examples/whatsapp_service.py`
  y `examples/whatsapp_persistent_bot.py`) lanzan `RuntimeError` si falta `WHATSAPP_APP_SECRET`, evitando
  secretos de prueba inseguros por defecto.

### Rendimiento y Confiabilidad

- **Procesamiento Asincrono de Webhooks con `BackgroundTasks` (BUS-01):** Retorna HTTP 200 a Meta
  de forma inmediata; la inferencia del agente y el envio saliente se ejecutan en segundo plano.
  Evita tormentas de reintentos causadas por latencia en el LLM (Meta aplica timeout a los 15 s).
- **Deduplicacion At-Most-Once por `wamid` (ARCH-01):** Los identificadores de mensaje se registran
  atomicamente en `chatflow_processed_messages` antes de encolar la tarea. Los reintentos duplicados
  de Meta (hasta 7 dias segun la documentacion oficial) se descartan limpiamente.
  El TTL por defecto es de 8 dias (`DEFAULT_DEDUP_TTL_HOURS = 192`) para evitar crecimiento indefinido.
- **Extraccion por Lotes Multi-Mensaje:** Cargas de webhook con multiples mensajes en
  `entry[].changes[].value.messages` son procesadas individualmente, deduplicadas y despachadas de forma aislada.
- **Escudo Anti-500 y Entrega de Fallback:** Tarea en segundo plano envuelta en manejo resiliente de errores;
  garantiza la entrega de un mensaje de fallback amigable al usuario sin provocar caidas con error HTTP 500.
- **Reintento Saliente Selectivo con Jitter:** Los reintentos hacia la Meta Cloud API se limitan
  estrictamente a errores de conexion de red (`ConnectError`, `ConnectTimeout`) y codigos HTTP `429` (rate limit)
  con backoff exponencial y jitter. Los timeouts de lectura o escritura no reintentan para evitar duplicacion.
  Respeta el encabezado `Retry-After`.
- **Manejo de Ventana de 24h de Meta:** Detecta el codigo de error `131047` ("mensaje enviado fuera de la ventana
  de 24 horas"), ejecuta el callback seguro `on_24h_window_expired` protegido con try/except y registra el log
  con el numero de telefono enmascarado.
- **Division Automatica de Mensajes Salientes:** Mensajes que superan el limite de 4.096 caracteres de Meta
  son segmentados automaticamente respetando saltos de parrafo y limites de oraciones.
- **Rollback ante Fallas a Mitad de Turno:** Revierte automaticamente el historial conversacional al ultimo
  turno consistente en memoria y SQLite ante excepciones del motor LLM, garantizando un estado limpio y
  asegurando el envio del mensaje de contingencia al usuario por WhatsApp.

### Almacenamiento e Integridad de Memoria

- **Poda por Turnos Completos (DATA-01):** `max_turns` respeta los turnos de conversacion completos. El algoritmo
  de poda preserva el prompt de sistema y garantiza que los pares atomicos `tool_calls` / `tool_results` nunca
  queden huerfanos.
- **Versionado de Esquema SQLite (MAINT-01):** `PRAGMA user_version = 1` y tabla
  `chatflow_processed_messages` para deduplicacion persistente con purga automatica por TTL.
- **Concurrencia SQLite y Transicion Transparente a WAL:** `PRAGMA busy_timeout = 10000` + `PRAGMA journal_mode=WAL`
  para eliminar contencion de bloqueos. Las bases de datos existentes pasan automaticamente a modo WAL sin
  requerir migracion manual ni recreacion de tablas.
- **Semantica de Rollback de Turno:** Se documenta expresamente que el rollback revierte el historial conversacional
  y no los efectos secundarios externos de herramientas que ya fueron ejecutadas.

### Documentacion

- **Paridad Bilingue al 100%:** Documentacion integral en español e inglés en README.md y CHANGELOG.md.
