"""Smoke tests for the public APIs the 0.4 migration guide documents (issue #149).

The guide previously described signatures and commands that do not exist
(`queue_timeout_seconds: float | None`, `/clear`, `store.delete_session`), and a
deduplication statement that was never used. These checks pin the documented
claims to the implementation, so the two cannot drift apart silently again.
"""
import inspect

import pytest

from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.memory import SQLiteSessionContext, SQLiteSessionStore


def test_queue_timeout_seconds_signature_matches_the_guide() -> None:
    param = inspect.signature(WhatsAppChannel.__init__).parameters["queue_timeout_seconds"]
    assert param.default == 120.0
    assert param.annotation is float
    with pytest.raises(TypeError):
        WhatsAppChannel(verify_token="t", queue_timeout_seconds=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WhatsAppChannel(verify_token="t", queue_timeout_seconds=0.0)


def test_cli_channel_supports_reset_not_clear() -> None:
    source = inspect.getsource(CLIChannel)
    assert '"/reset"' in source
    assert '"/clear"' not in source


def test_session_store_api_is_delete_not_delete_session() -> None:
    assert hasattr(SQLiteSessionStore, "delete")
    assert not hasattr(SQLiteSessionStore, "delete_session")


def test_restore_history_uses_the_in_memory_journal_and_begin_immediate() -> None:
    source = inspect.getsource(SQLiteSessionContext.restore_history)
    assert "BEGIN IMMEDIATE" in source
    assert "CREATE" not in source  # no temporary turn_journal table


def test_record_processed_message_is_a_conditional_upsert() -> None:
    source = inspect.getsource(SQLiteSessionStore.record_processed_message)
    assert "ON CONFLICT(message_id) DO UPDATE" in source
    assert "INSERT OR IGNORE" not in source
