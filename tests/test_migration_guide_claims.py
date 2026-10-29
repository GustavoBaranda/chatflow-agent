"""Smoke tests for the public APIs the 0.4 migration guide documents (issue #149).

The guide previously described signatures and commands that do not exist
(`queue_timeout_seconds: float | None`, `/clear`, `store.delete_session`), and a
deduplication statement that was never used. These checks pin the documented
surface to the implementation; the behaviour behind the remaining claims is
covered by the existing tests rather than by reading source text:

- `/reset` handling: `tests/test_cli_channel.py`
- SQLite dedup / TTL renewal: `tests/test_memory_sqlite.py::test_sqlite_deduplication_by_wamid_with_ttl`
"""
import inspect

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.memory import SQLiteSessionStore


def test_queue_timeout_seconds_is_optional_but_validated() -> None:
    param = inspect.signature(WhatsAppChannel.__init__).parameters["queue_timeout_seconds"]
    assert param.default == 120.0
    assert param.annotation is float
    with pytest.raises(TypeError):
        WhatsAppChannel(verify_token="t", queue_timeout_seconds=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WhatsAppChannel(verify_token="t", queue_timeout_seconds=0.0)


def test_session_store_api_is_delete_not_delete_session() -> None:
    assert hasattr(SQLiteSessionStore, "delete")
    assert not hasattr(SQLiteSessionStore, "delete_session")


def test_record_processed_message_reports_duplicates(tmp_path) -> None:
    """The documented conditional upsert returns True once and False afterwards."""
    store = SQLiteSessionStore(db_path=tmp_path / "dedup.db", dedup_ttl_hours=1)
    try:
        assert store.record_processed_message("wamid_guide_claim") is True
        assert store.record_processed_message("wamid_guide_claim") is False
    finally:
        store.close()
