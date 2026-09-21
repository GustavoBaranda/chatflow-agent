# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) ·
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

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

### Performance & Reliability

- **Async Webhook Processing via `BackgroundTasks` (BUS-01):** Returns HTTP 200 to Meta
  immediately; agent inference and outbound messaging run in a background task.
  Prevents Meta retry storms caused by slow LLM responses (Meta times out at 15 s).
- **At-Most-Once Dedup by `wamid` (ARCH-01):** Message IDs are atomically recorded in
  `chatflow_processed_messages` before the background task is enqueued. Duplicate
  deliveries from Meta retries (up to 7 days per Meta docs) are safely discarded.
  TTL defaults to 8 days (> Meta's 7-day retry window) to prevent unbounded growth.
- **Multi-Message Batch Extraction:** Webhook payloads with multiple messages in
  `entry[].changes[].value.messages` are fully extracted; each message is individually
  deduplicated and independently dispatched.
- **Anti-500 Shield:** Background task wraps the full flow in resilient error handling;
  delivers a user-facing fallback message on failures without HTTP 500 crashes.

### Storage & Memory Integrity

- **Turn-Based Pruning (DATA-01):** `max_turns` now counts complete user conversation
  turns. The pruning algorithm preserves the system prompt and guarantees that
  `tool_calls` / `tool_results` pairs are never severed.
- **SQLite Schema Versioning (MAINT-01):** `PRAGMA user_version = 1` and table
  `chatflow_processed_messages` for persistent deduplication with TTL purge.
- **SQLite Concurrency (MAINT-01):** `PRAGMA busy_timeout` + `PRAGMA journal_mode=WAL`
  to prevent lock contention. WAL requires a local filesystem (not NFS/EFS) and a
  single Uvicorn worker process.
