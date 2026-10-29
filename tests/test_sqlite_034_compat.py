"""Open a chatflow-agent 0.3.4 SQLite file with the current SQLiteSessionStore.

The database under tests/fixtures/sqlite_0_3_4 is produced by
scripts/generate_sqlite_034_fixture.py. These tests copy it before any write.
They do not install 0.3.4.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from chatflow_agent import Agent, SQLiteSessionStore
from chatflow_agent.types import Message, Role

FIXTURE_DB = Path(__file__).resolve().parent / "fixtures" / "sqlite_0_3_4" / "chatflow-0.3.4.db"
LEGACY_USER_VERSION = 1
# The committed processed_at is the generation clock. A long TTL keeps that row
# inside the dedup window without rewriting it to "now".
LEGACY_DEDUP_TTL_HOURS = 24 * 365 * 100


def copy_legacy_fixture(tmp_path: Path) -> Path:
    """Copy the committed 0.3.4 database so the test mutates only the copy."""
    destination = tmp_path / "chatflow-0.3.4.db"
    shutil.copy(FIXTURE_DB, destination)
    return destination


def _master_rows(db_path: Path) -> list[tuple[str, str, str | None]]:
    conn = sqlite3.connect(db_path)
    try:
        return list(
            conn.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
        )
    finally:
        conn.close()


def _pragma(db_path: Path, name: str) -> str | int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()


def _agents() -> tuple[Agent, Agent]:
    triage = Agent(name="Triage", instructions="Route the user.")
    billing = Agent(name="Billing", instructions="Handle invoices.")
    return triage, billing


def _open(db_path: Path, triage: Agent, billing: Agent) -> SQLiteSessionStore:
    store = SQLiteSessionStore(db_path=db_path)
    store.set_agent_resolver(lambda name: {"Triage": triage, "Billing": billing}.get(name))
    return store


def test_sqlite_034_opens_without_migration(tmp_path: Path) -> None:
    original = FIXTURE_DB.read_bytes()
    db_path = copy_legacy_fixture(tmp_path)
    triage, billing = _agents()
    schema_before = _master_rows(db_path)
    assert _pragma(db_path, "user_version") == LEGACY_USER_VERSION
    assert _pragma(db_path, "integrity_check") == "ok"

    with _open(db_path, triage, billing) as store:
        assert set(store.list_sessions()) == {"legacy-billing", "legacy-triage"}
        loaded = store.get("legacy-billing")
        assert loaded is not None
        assert loaded.active_agent.name == "Billing"

    assert _master_rows(db_path) == schema_before
    assert _pragma(db_path, "user_version") == LEGACY_USER_VERSION
    assert _pragma(db_path, "integrity_check") == "ok"
    assert FIXTURE_DB.read_bytes() == original


def test_sqlite_034_reads_legacy_sessions_history_agent_and_metadata(tmp_path: Path) -> None:
    triage, billing = _agents()
    db_path = copy_legacy_fixture(tmp_path)

    with _open(db_path, triage, billing) as store:
        loaded = store.get_or_create(
            "legacy-billing",
            default_agent=triage,
            metadata={"channel": "should-not-overwrite"},
        )
        assert loaded.active_agent is billing
        assert loaded.metadata == {"channel": "whatsapp", "locale": "en"}
        assert [(message.role, message.content) for message in loaded.history] == [
            (Role.USER, "Need invoice 1042"),
            (Role.MODEL, "Invoice 1042 is open."),
        ]
        assert loaded.history[0].timestamp == datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        assert loaded.history[1].timestamp == datetime(2026, 10, 23, 12, 0, 1, tzinfo=timezone.utc)
        assert loaded.history[0].tool_calls is None
        assert loaded.history[0].tool_results is None

        other = store.get("legacy-triage")
        assert other is not None
        assert other.active_agent.name == "Triage"
        assert other.metadata == {"channel": "cli"}
        assert [(message.role, message.content) for message in other.history] == [
            (Role.USER, "Hello"),
        ]
        assert other.history[0].timestamp == datetime(2026, 10, 23, 12, 0, 2, tzinfo=timezone.utc)


def test_sqlite_034_appends_turn_and_reloads(tmp_path: Path) -> None:
    triage, _billing = _agents()
    db_path = copy_legacy_fixture(tmp_path)
    appended_at = datetime(2026, 10, 23, 12, 1, tzinfo=timezone.utc)

    with _open(db_path, triage, _billing) as store:
        ctx = store.get_or_create("legacy-billing", default_agent=triage)
        ctx.add_message(Message(role=Role.USER, content="Send it to accounts.", timestamp=appended_at))

    with _open(db_path, triage, _billing) as store:
        reloaded = store.get("legacy-billing")
        assert reloaded is not None
        assert [(message.role, message.content) for message in reloaded.history] == [
            (Role.USER, "Need invoice 1042"),
            (Role.MODEL, "Invoice 1042 is open."),
            (Role.USER, "Send it to accounts."),
        ]
        assert reloaded.history[2].timestamp == appended_at
        assert reloaded.active_agent.name == "Billing"
        assert reloaded.metadata == {"channel": "whatsapp", "locale": "en"}

        untouched = store.get("legacy-triage")
        assert untouched is not None
        assert [message.content for message in untouched.history] == ["Hello"]

    assert _pragma(db_path, "integrity_check") == "ok"
    assert _pragma(db_path, "user_version") == LEGACY_USER_VERSION


def test_sqlite_034_delete_and_clear_history(tmp_path: Path) -> None:
    triage, _billing = _agents()
    db_path = copy_legacy_fixture(tmp_path)

    with _open(db_path, triage, _billing) as store:
        billing_ctx = store.get_or_create("legacy-billing", default_agent=triage)
        billing_ctx.clear_history()
        assert store.delete("legacy-triage") is True
        assert store.delete("legacy-triage") is False

    with _open(db_path, triage, _billing) as store:
        billing_ctx = store.get("legacy-billing")
        assert billing_ctx is not None
        assert billing_ctx.history == []
        assert billing_ctx.active_agent.name == "Billing"
        assert billing_ctx.metadata == {"channel": "whatsapp", "locale": "en"}
        assert store.get("legacy-triage") is None
        assert store.list_sessions() == ["legacy-billing"]

    conn = sqlite3.connect(db_path)
    try:
        leftover = conn.execute(
            "SELECT COUNT(*) FROM chatflow_messages WHERE session_id = ?",
            ("legacy-triage",),
        ).fetchone()[0]
        billing_messages = conn.execute(
            "SELECT COUNT(*) FROM chatflow_messages WHERE session_id = ?",
            ("legacy-billing",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert leftover == 0
    assert billing_messages == 0
    assert _pragma(db_path, "integrity_check") == "ok"


def test_sqlite_034_dedup_uses_existing_processed_messages(tmp_path: Path) -> None:
    db_path = copy_legacy_fixture(tmp_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "chatflow_processed_messages" in tables

    with SQLiteSessionStore(db_path=db_path, dedup_ttl_hours=LEGACY_DEDUP_TTL_HOURS) as store:
        assert store.is_message_processed("wamid.legacy.1") is True
        assert store.record_processed_message("wamid.legacy.1") is False
        assert store.record_processed_message("wamid.legacy.2") is True
        assert store.is_message_processed("wamid.legacy.2") is True
        assert store.record_processed_message("wamid.legacy.2") is False

    assert _pragma(db_path, "integrity_check") == "ok"
    assert _pragma(db_path, "user_version") == LEGACY_USER_VERSION
