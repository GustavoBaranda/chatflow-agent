# Changelog / Registro de Cambios

All notable changes to this project will be documented in this file.  
*Todos los cambios notables en este proyecto serán documentados en este archivo.*

Format / Formato: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning / Versionado: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

[English](#english) | [Español](#español)

---

## English

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
