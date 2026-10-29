"""Write the 0.3.4 session fixture. Run only via generate_sqlite_034_fixture.py.

The generator installs chatflow-agent 0.3.4 in a temporary virtualenv and executes
this file with that interpreter. Importing it from the checkout uses the current
package and refuses to write.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import chatflow_agent
from chatflow_agent import Agent, Message, Role, SQLiteSessionStore

REQUIRED_VERSION = "0.3.4"

USER_TS = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
MODEL_TS = datetime(2026, 10, 23, 12, 0, 1, tzinfo=timezone.utc)
HELLO_TS = datetime(2026, 10, 23, 12, 0, 2, tzinfo=timezone.utc)

BILLING_METADATA = '{"channel": "whatsapp", "locale": "en"}'
TRIAGE_METADATA = '{"channel": "cli"}'
USER_JSON = (
    '{"role":"user","content":"Need invoice 1042","tool_calls":null,'
    '"tool_results":null,"timestamp":"2026-10-23T12:00:00Z"}'
)
MODEL_JSON = (
    '{"role":"model","content":"Invoice 1042 is open.","tool_calls":null,'
    '"tool_results":null,"timestamp":"2026-10-23T12:00:01Z"}'
)
HELLO_JSON = (
    '{"role":"user","content":"Hello","tool_calls":null,'
    '"tool_results":null,"timestamp":"2026-10-23T12:00:02Z"}'
)


def _fail(message: str) -> None:
    raise SystemExit(message)


def write_fixture(db_path: Path) -> None:
    """Insert the legacy rows through the 0.3.4 session store."""
    if chatflow_agent.__version__ != REQUIRED_VERSION:
        _fail(
            "sqlite_034_fixture_writer.py must run on chatflow-agent "
            f"{REQUIRED_VERSION}, found {chatflow_agent.__version__} at {chatflow_agent.__file__}"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    triage = Agent(name="Triage", instructions="Route the user.")
    billing = Agent(name="Billing", instructions="Handle invoices.")
    store = SQLiteSessionStore(db_path=db_path, default_max_turns=40)

    billing_ctx = store.get_or_create(
        "legacy-billing",
        default_agent=triage,
        metadata={"channel": "whatsapp", "locale": "en"},
    )
    billing_ctx.set_active_agent(billing)
    billing_ctx.add_message(Message(role=Role.USER, content="Need invoice 1042", timestamp=USER_TS))
    billing_ctx.add_message(
        Message(role=Role.MODEL, content="Invoice 1042 is open.", timestamp=MODEL_TS)
    )

    triage_ctx = store.get_or_create(
        "legacy-triage",
        default_agent=triage,
        metadata={"channel": "cli"},
    )
    triage_ctx.add_message(Message(role=Role.USER, content="Hello", timestamp=HELLO_TS))

    if store.record_processed_message("wamid.legacy.1") is not True:
        _fail("record_processed_message did not accept wamid.legacy.1")
    if store.record_processed_message("wamid.legacy.1") is not False:
        _fail("record_processed_message treated wamid.legacy.1 as new on the second call")

    del store
    _checkpoint_and_check(db_path)


def _checkpoint_and_check(db_path: Path) -> None:
    """Fold the WAL into the main file and refuse a database that is not the expected rows."""
    conn = sqlite3.connect(db_path)
    try:
        busy, _log, _checkpointed = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            _fail(f"wal_checkpoint left the database busy: {db_path}")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            _fail(f"integrity_check returned {integrity!r}")
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if user_version != 1:
            _fail(f"0.3.4 user_version is {user_version}, expected 1")

        sessions = conn.execute(
            """
            SELECT session_id, active_agent_name, metadata_json
            FROM chatflow_sessions
            ORDER BY session_id
            """
        ).fetchall()
        if sessions != [
            ("legacy-billing", "Billing", BILLING_METADATA),
            ("legacy-triage", "Triage", TRIAGE_METADATA),
        ]:
            _fail(f"unexpected sessions: {sessions!r}")

        messages = conn.execute(
            """
            SELECT session_id, role, content, message_json
            FROM chatflow_messages
            ORDER BY id
            """
        ).fetchall()
        if messages != [
            ("legacy-billing", "user", "Need invoice 1042", USER_JSON),
            ("legacy-billing", "model", "Invoice 1042 is open.", MODEL_JSON),
            ("legacy-triage", "user", "Hello", HELLO_JSON),
        ]:
            _fail(f"unexpected messages: {messages!r}")

        processed = conn.execute(
            "SELECT message_id FROM chatflow_processed_messages ORDER BY message_id"
        ).fetchall()
        if processed != [("wamid.legacy.1",)]:
            _fail(f"unexpected processed messages: {processed!r}")
    finally:
        conn.close()

    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if not sidecar.exists():
            continue
        if sidecar.stat().st_size > 0:
            _fail(f"non-empty sidecar remains after checkpoint: {sidecar}")
        sidecar.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        _fail(f"usage: {Path(__file__).name} OUTPUT.db")
    write_fixture(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
