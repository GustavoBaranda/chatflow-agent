# SQLite fixture from chatflow-agent 0.3.4

`chatflow-0.3.4.db` is a session database written by chatflow-agent 0.3.4.
Regression tests copy it with `shutil.copy` and open the copy with the current
`SQLiteSessionStore`. Pytest does not install 0.3.4.

## Provenance

The package is `chatflow-agent==0.3.4`.
Generate the file with `python scripts/generate_sqlite_034_fixture.py`.
This copy was generated at 2026-10-29 02:42:31 UTC with Python 3.13.5.

`PRAGMA user_version` is 1. `PRAGMA journal_mode` is wal.
`PRAGMA integrity_check` returned `ok`.

The generator installs `chatflow-agent==0.3.4` into a temporary virtualenv and runs
`scripts/sqlite_034_fixture_writer.py` with that interpreter. The writer exits
if the imported package version is not 0.3.4. The writer inserts
rows with `SQLiteSessionStore.get_or_create`, `set_active_agent`, `add_message`,
and `record_processed_message`, then runs `PRAGMA wal_checkpoint(TRUNCATE)`.
The generator copies the single database file.
Timestamps stored inside `message_json` are fixed.
`created_at`, `updated_at`, and `processed_at` use SQLite `CURRENT_TIMESTAMP`
at generation time.

## Schema

These are the `sqlite_master` SQL texts in the file after chatflow-agent
0.3.4 `_init_db` and the inserts. SQLite added `sqlite_sequence`
because `chatflow_messages.id` is `AUTOINCREMENT`. The compat tests compare
`sqlite_master` on a copy before and after the current store opens it.

```sql
CREATE INDEX idx_chatflow_messages_session ON chatflow_messages(session_id, id)

CREATE INDEX idx_chatflow_processed_at ON chatflow_processed_messages(processed_at)

CREATE TABLE chatflow_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    message_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chatflow_sessions(session_id) ON DELETE CASCADE
                )

CREATE TABLE chatflow_processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )

CREATE TABLE chatflow_sessions (
                    session_id TEXT PRIMARY KEY,
                    active_agent_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )

CREATE TABLE sqlite_sequence(name,seq)
```

## Rows

`legacy-billing` was created with agent Triage, then `set_active_agent`
switched it to Billing. `active_agent_name` is `Billing`. `metadata_json` is
`{"channel": "whatsapp", "locale": "en"}`. History has two messages.

1. Role `user`, content `Need invoice 1042`, timestamp `2026-10-23T12:00:00Z`.
2. Role `model`, content `Invoice 1042 is open.`, timestamp `2026-10-23T12:00:01Z`.

`legacy-triage` keeps agent Triage. `metadata_json` is `{"channel": "cli"}`.
History is one `user` message, `Hello`, timestamp `2026-10-23T12:00:02Z`.

`chatflow_processed_messages` contains `wamid.legacy.1`.

## Regenerate

From the repository root, with network access so pip can install `chatflow-agent==0.3.4`:

```
python scripts/generate_sqlite_034_fixture.py
```

The command replaces `chatflow-0.3.4.db` and rewrites this README from the new file.
