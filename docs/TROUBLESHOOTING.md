# Troubleshooting

🌐 **Language:** **English** | [Español](es/TROUBLESHOOTING.md)

Common problems when installing and running `chatflow-agent`, with the exact messages the library produces and how to fix them. For design details see the [Architecture Guide](ARCHITECTURE.md); for upgrading from `0.3.x` see the [Migration Guide](MIGRATION_0.4.md).

- [Installation](#installation)
- [API keys and provider configuration](#api-keys-and-provider-configuration)
- [WhatsApp webhook](#whatsapp-webhook)
- [Timeouts](#timeouts)
- [SQLite session stores](#sqlite-session-stores)

---

## Installation

### A documented API does not exist after `pip install chatflow-agent`

**Cause:** the documentation describes the `0.4.0` prerelease. Plain `pip install chatflow-agent` installs the latest stable `0.3.x` release, which does not include every `0.4.0` API.

**Fix:** install the prerelease explicitly, as shown in the README's [Installation & Requirements](../README.md#installation--requirements) section:

```bash
pip install --pre chatflow-agent
pip install --pre "chatflow-agent[whatsapp]"   # with an optional extra
```

### `DependencyError` when using a channel

```text
The feature '<feature>' requires optional dependencies from '<extra>'. Please install them using: pip install "chatflow-agent[<extra>]"
```

**Cause:** the core package does not install channel dependencies. `WhatsAppChannel` needs the `whatsapp` extra (FastAPI, Uvicorn) and `TelegramChannel` needs the `telegram` extra (`python-telegram-bot`).

**Fix:** run the command shown in the message, adding `--pre` while `0.4.0` is a prerelease. `pip install --pre "chatflow-agent[all]"` installs every extra.

---

## API keys and provider configuration

### `ProviderError: Gemini API client is not configured`

```text
Gemini API client is not configured. Set GEMINI_API_KEY environment variable or pass api_key to Runner.
```

**Cause:** no key was found in `api_key` (on the `Runner` or the `Agent`), `GEMINI_API_KEY` or `GOOGLE_API_KEY`. The error is raised when the first turn calls Gemini, not when the `Runner` is created.

**Fix:** set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), or pass `api_key` to `Runner` or to the `Agent`. Vertex AI and Application Default Credentials are not supported.

### `ProviderError: Gemini API key is invalid`

**Cause:** the key contains spaces, control characters or non-ASCII characters. Surrounding whitespace, such as the trailing newline of a secrets file, is already stripped.

**Fix:** check the value of `GEMINI_API_KEY` or of the `api_key` you pass.

### `ProviderError: Anthropic API key is not configured`

**Cause:** `AnthropicEngine` found no key in `api_key` or `ANTHROPIC_API_KEY`.

**Fix:** set `ANTHROPIC_API_KEY` or pass `api_key`. Passing `api_key=""` deliberately sends no credential (for example, to an unauthenticated gateway).

### `ConfigurationError` when creating an `Agent` or the `Runner`

Invalid provider settings fail when the `Agent` is created. The `Runner` also validates every agent's engine configuration when it is created. Common messages:

| Message (beginning) | Raised by | Cause | Fix |
|---|---|---|---|
| `Unsupported provider '...'. Supported providers are: ...` | `Agent(...)` | `provider` is not a supported provider name. | Use one of the listed names, or omit `provider` and rely on `base_url` or the model name. |
| `Provider '...' does not support a custom 'base_url'.` | `Agent(...)` | `provider="gemini"` (or `"google"`) was combined with `base_url`. | Remove `base_url`. For a custom Gemini endpoint, set the `GOOGLE_GEMINI_BASE_URL` environment variable. |
| `A custom or unofficial base_url was provided (...) without an explicit api_key.` | `Runner(...)` | An environment key (for example `OPENAI_API_KEY`) is only sent to its provider's official endpoint, never to a custom `base_url`. | Pass `api_key` explicitly, or `api_key=""` if the endpoint needs no authentication. |

`ConfigurationError` also subclasses `ValueError`.

---

## WhatsApp webhook

### `ValueError` when creating `WhatsAppChannel`

```text
WhatsAppChannel requires 'app_secret' when verify_signature=True (the default). ...
```

**Cause:** webhook signature verification is on by default and fails closed without the Meta app secret.

**Fix:** pass `app_secret` (for example from `WHATSAPP_APP_SECRET`). Only for local development or tests, pass `verify_signature=False`, which logs:

```text
SECURITY WARNING: Webhook signature verification is DISABLED (verify_signature=False). Do NOT use this in production.
```

### Webhook requests are rejected

| Status | `detail` | Cause |
|---|---|---|
| 403 | `Verification token mismatch` | The `hub.verify_token` sent by Meta during webhook verification does not match `verify_token`. |
| 403 | `Missing or malformed X-Hub-Signature-256 header` | The request has no `X-Hub-Signature-256: sha256=...` header. |
| 403 | `Invalid X-Hub-Signature-256 signature` | The HMAC-SHA256 of the raw body does not match; usually a wrong `app_secret`, or a proxy that changes the body before it reaches the app. |
| 411 | `Missing Content-Length header` | The request has no `Content-Length`. |
| 400 | `Invalid Content-Length header` | `Content-Length` is not a valid number. |
| 413 | `Payload size exceeds maximum limit of ... bytes` / `Payload body exceeds maximum limit of ... bytes` | The body is larger than `max_body_size` (1 MiB by default). |
| 400 | `Invalid JSON payload` / `JSON payload must be an object` | The body is not valid JSON, or is not a JSON object. |

### Turns run but no replies are sent

```text
WhatsAppChannel created without outbound credentials (access_token and/or phone_number_id missing); turns will run but no WhatsApp replies will be sent.
```

**Fix:** pass both `access_token` and `phone_number_id`.

---

## Timeouts

### WhatsApp messages answered with `fallback_message`

Each WhatsApp message has two separate limits, and each logs a distinct error:

| Log (beginning) | Limit | Meaning |
|---|---|---|
| `Session busy: wamid=... waited ...s (queue_timeout_seconds) behind earlier turns of the same session; its turn was not run.` | `queue_timeout_seconds` (default 120) | The message waited too long behind earlier turns of the same user. Its turn never ran and nothing was added to the history. |
| `Turn timed out after ...s (llm_timeout_seconds) ...; the turn was rolled back.` | `llm_timeout_seconds` (default 120) | The turn itself (model calls, tools, handoffs) took too long. It was cancelled and its changes to the session were rolled back. |
| `Dispatch backstop expired after ...s ...` | `queue_timeout_seconds + llm_timeout_seconds` plus a grace period | `dispatch_async()` did not finish in time; usually an override that does not call `Runner.run_async()`. |

Waiting in the queue does not consume `llm_timeout_seconds`, so a message can take up to `queue_timeout_seconds + llm_timeout_seconds` before its reply. Raise the limit that matches the log. `llm_timeout_seconds=None` or `inf` sets no turn limit.

### `TimeoutError` from `Runner.run_async(..., turn_timeout=...)`

**Cause:** the turn exceeded `turn_timeout` seconds and raised `asyncio.TimeoutError` (the built-in `TimeoutError` on Python 3.11+). The limit starts once the turn holds its session, so waiting behind other turns of the same session does not count. The turn is cancelled and the session (history, active agent, metadata) is restored before the error is raised.

**Fix:** raise `turn_timeout`, or pass `None` (the default) for no limit. A synchronous (`def`) tool that was already running keeps running in its worker thread after the timeout; see [Production Resilience & Concurrency](../README.md#production-resilience--concurrency) in the README.

---

## SQLite session stores

### `sqlite3.OperationalError: database is locked`

**Cause:** every SQLite connection waits up to 10 seconds (`busy_timeout`) for another writer, such as another process or an external tool holding a write transaction. After that, the operation fails.

- In a `Runner` turn, the error propagates from `run_async()`.
- In the WhatsApp webhook, the deduplication write fails, the request ends with HTTP 500 and nothing is recorded, so Meta's retry is processed normally.

**Fix:** avoid long write transactions from other tools on the same database file. `SQLiteSessionStore` serializes turns of a session within one store and event loop; it does not coordinate several processes. Multi-process or multi-node deployments need an external store provided by the application.

### Sessions disappear after a restart

**Cause:** `SQLiteSessionStore(":memory:")` keeps data in a shared-cache in-memory database. It does not use WAL, and its data is lost when the store is closed or the process exits.

**Fix:** use a file path, such as `SQLiteSessionStore("sessions.db")`, for persistence. File-backed stores use WAL.

### Worker threads or files stay open after shutdown

**Fix:** close stores deterministically with `with SQLiteSessionStore(...) as store:` or `store.close()`.
