# Solución de Problemas

🌐 **Idioma:** [English](../TROUBLESHOOTING.md) | **Español**

Problemas frecuentes al instalar y ejecutar `chatflow-agent`, con los mensajes exactos que produce la biblioteca y cómo resolverlos. Para los detalles de diseño, consultá la [Guía de Arquitectura](ARCHITECTURE.md); para actualizar desde `0.3.x`, la [Guía de Migración](MIGRATION_0.4.md).

- [Instalación](#instalación)
- [API keys y configuración de proveedores](#api-keys-y-configuración-de-proveedores)
- [Webhook de WhatsApp](#webhook-de-whatsapp)
- [Timeouts](#timeouts)
- [Almacenes de sesión SQLite](#almacenes-de-sesión-sqlite)

---

## Instalación

### Una API documentada no existe después de `pip install chatflow-agent`

**Causa:** la documentación describe la prerelease `0.4.0`. Un `pip install chatflow-agent` sin opciones instala la última versión estable `0.3.x`, que no incluye todas las APIs de `0.4.0`.

**Solución:** instalá la prerelease de forma explícita, como se indica en la sección [Instalación y requisitos](../../README.es.md#instalación-y-requisitos) del README:

```bash
pip install --pre chatflow-agent
pip install --pre "chatflow-agent[whatsapp]"   # con un extra opcional
```

### `DependencyError` al usar un canal

```text
The feature '<feature>' requires optional dependencies from '<extra>'. Please install them using: pip install "chatflow-agent[<extra>]"
```

**Causa:** el paquete base no instala las dependencias de los canales. `WhatsAppChannel` necesita el extra `whatsapp` (FastAPI, Uvicorn) y `TelegramChannel` el extra `telegram` (`python-telegram-bot`).

**Solución:** ejecutá el comando que indica el mensaje, agregando `--pre` mientras `0.4.0` sea una prerelease. `pip install --pre "chatflow-agent[all]"` instala todos los extras.

---

## API keys y configuración de proveedores

### `ProviderError: Gemini API client is not configured`

```text
Gemini API client is not configured. Set GEMINI_API_KEY environment variable or pass api_key to Runner.
```

**Causa:** no se encontró ninguna clave en `api_key` (del `Runner` o del `Agent`), `GEMINI_API_KEY` ni `GOOGLE_API_KEY`. El error se produce cuando el primer turno llama a Gemini, no al crear el `Runner`.

**Solución:** definí `GEMINI_API_KEY` (o `GOOGLE_API_KEY`), o pasá `api_key` al `Runner` o al `Agent`. Vertex AI y Application Default Credentials no están soportados.

### `ProviderError: Gemini API key is invalid`

**Causa:** la clave contiene espacios, caracteres de control o caracteres no ASCII. Los espacios en los extremos, como el salto de línea final de un archivo de secretos, ya se eliminan.

**Solución:** revisá el valor de `GEMINI_API_KEY` o de la `api_key` que pasás.

### `ProviderError: Anthropic API key is not configured`

**Causa:** `AnthropicEngine` no encontró ninguna clave en `api_key` ni en `ANTHROPIC_API_KEY`.

**Solución:** definí `ANTHROPIC_API_KEY` o pasá `api_key`. Pasar `api_key=""` indica a propósito que no se envíe ninguna credencial (por ejemplo, a un gateway sin autenticación).

### `ConfigurationError` al crear un `Agent` o el `Runner`

Una configuración de proveedor inválida falla al crear el `Agent`. El `Runner` además valida la configuración de motor de cada agente al crearse. Mensajes frecuentes:

| Mensaje (inicio) | Lo lanza | Causa | Solución |
|---|---|---|---|
| `Unsupported provider '...'. Supported providers are: ...` | `Agent(...)` | `provider` no es un nombre de proveedor soportado. | Usá uno de los nombres listados, u omití `provider` y dejá que decidan `base_url` o el nombre del modelo. |
| `Provider '...' does not support a custom 'base_url'.` | `Agent(...)` | Se combinó `provider="gemini"` (o `"google"`) con `base_url`. | Quitá `base_url`. Para un endpoint personalizado de Gemini, definí la variable de entorno `GOOGLE_GEMINI_BASE_URL`. |
| `A custom or unofficial base_url was provided (...) without an explicit api_key.` | `Runner(...)` | Una clave del entorno (por ejemplo `OPENAI_API_KEY`) solo se envía al endpoint oficial de su proveedor, nunca a un `base_url` personalizado. | Pasá `api_key` explícitamente, o `api_key=""` si el endpoint no requiere autenticación. |

`ConfigurationError` también es subclase de `ValueError`.

---

## Webhook de WhatsApp

### `ValueError` al crear `WhatsAppChannel`

```text
WhatsAppChannel requires 'app_secret' when verify_signature=True (the default). ...
```

**Causa:** la verificación de firmas del webhook está activada por defecto y falla de forma cerrada si no hay secreto de la app de Meta.

**Solución:** pasá `app_secret` (por ejemplo desde `WHATSAPP_APP_SECRET`). Solo para desarrollo local o pruebas, pasá `verify_signature=False`, que registra:

```text
SECURITY WARNING: Webhook signature verification is DISABLED (verify_signature=False). Do NOT use this in production.
```

### Se rechazan las solicitudes al webhook

| Estado | `detail` | Causa |
|---|---|---|
| 403 | `Verification token mismatch` | El `hub.verify_token` que envía Meta al verificar el webhook no coincide con `verify_token`. |
| 403 | `Missing or malformed X-Hub-Signature-256 header` | La solicitud no tiene el encabezado `X-Hub-Signature-256: sha256=...`. |
| 403 | `Invalid X-Hub-Signature-256 signature` | El HMAC-SHA256 del cuerpo original no coincide; suele ser un `app_secret` incorrecto o un proxy que modifica el cuerpo antes de que llegue a la aplicación. |
| 411 | `Missing Content-Length header` | La solicitud no tiene `Content-Length`. |
| 400 | `Invalid Content-Length header` | `Content-Length` no es un número válido. |
| 413 | `Payload size exceeds maximum limit of ... bytes` / `Payload body exceeds maximum limit of ... bytes` | El cuerpo supera `max_body_size` (1 MiB por defecto). |
| 400 | `Invalid JSON payload` / `JSON payload must be an object` | El cuerpo no es JSON válido o no es un objeto JSON. |

### Los turnos se ejecutan pero no se envían respuestas

```text
WhatsAppChannel created without outbound credentials (access_token and/or phone_number_id missing); turns will run but no WhatsApp replies will be sent.
```

**Solución:** pasá tanto `access_token` como `phone_number_id`.

---

## Timeouts

### Mensajes de WhatsApp respondidos con `fallback_message`

Cada mensaje de WhatsApp tiene dos límites separados, y cada uno registra un error distinto:

| Log (inicio) | Límite | Significado |
|---|---|---|
| `Session busy: wamid=... waited ...s (queue_timeout_seconds) behind earlier turns of the same session; its turn was not run.` | `queue_timeout_seconds` (120 por defecto) | El mensaje esperó demasiado detrás de turnos anteriores del mismo usuario. Su turno nunca se ejecutó y no se agregó nada al historial. |
| `Turn timed out after ...s (llm_timeout_seconds) ...; the turn was rolled back.` | `llm_timeout_seconds` (120 por defecto) | El turno en sí (llamadas al modelo, herramientas, handoffs) tardó demasiado. Se canceló y se revirtieron sus cambios en la sesión. |
| `Dispatch backstop expired after ...s ...` | `queue_timeout_seconds + llm_timeout_seconds` más un margen | `dispatch_async()` no terminó a tiempo; suele deberse a una sobrescritura que no llama a `Runner.run_async()`. |

La espera en la cola no consume `llm_timeout_seconds`, así que un mensaje puede tardar hasta `queue_timeout_seconds + llm_timeout_seconds` antes de su respuesta. Aumentá el límite que corresponda al log. `llm_timeout_seconds=None` o `inf` no pone límite al turno.

### `TimeoutError` desde `Runner.run_async(..., turn_timeout=...)`

**Causa:** el turno superó `turn_timeout` segundos y lanzó `asyncio.TimeoutError` (el `TimeoutError` integrado en Python 3.11+). El límite empieza cuando el turno obtiene su sesión, así que la espera detrás de otros turnos de la misma sesión no cuenta. El turno se cancela y la sesión (historial, agente activo, metadatos) se restaura antes de lanzar el error.

**Solución:** aumentá `turn_timeout`, o pasá `None` (el valor por defecto) para no tener límite. Una herramienta síncrona (`def`) que ya estaba en ejecución sigue corriendo en su hilo de trabajo después del timeout; consultá [Resiliencia y concurrencia en producción](../../README.es.md#resiliencia-y-concurrencia-en-producción) en el README.

---

## Almacenes de sesión SQLite

### `sqlite3.OperationalError: database is locked`

**Causa:** cada conexión SQLite espera hasta 10 segundos (`busy_timeout`) a que termine otro escritor, como otro proceso o una herramienta externa con una transacción de escritura abierta. Pasado ese tiempo, la operación falla.

- En un turno del `Runner`, el error se propaga desde `run_async()`.
- En el webhook de WhatsApp, falla la escritura de deduplicación, la solicitud termina con HTTP 500 y no se registra nada, así que el reintento de Meta se procesa con normalidad.

**Solución:** evitá transacciones de escritura largas de otras herramientas sobre el mismo archivo de base de datos. `SQLiteSessionStore` serializa los turnos de una sesión dentro de un mismo almacén y bucle de eventos; no coordina varios procesos. Los despliegues con varios procesos o nodos necesitan un almacén externo provisto por la aplicación.

### Las sesiones desaparecen después de reiniciar

**Causa:** `SQLiteSessionStore(":memory:")` guarda los datos en una base en memoria con caché compartida. No usa WAL, y sus datos se pierden al cerrar el almacén o al terminar el proceso.

**Solución:** usá una ruta de archivo, como `SQLiteSessionStore("sessions.db")`, para tener persistencia. Los almacenes en archivo usan WAL.

### Quedan hilos o archivos abiertos después de cerrar la aplicación

**Solución:** cerrá los almacenes de forma determinista con `with SQLiteSessionStore(...) as store:` o `store.close()`.
