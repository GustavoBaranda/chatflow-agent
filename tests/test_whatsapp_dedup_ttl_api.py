"""WhatsAppChannel no longer accepts an unused dedup_ttl_hours knob."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from tests.test_sqlite_dedup_atomic import _age
from tests.test_whatsapp_background import CountingEngine, _meta_payload


def test_whatsapp_channel_rejects_dedup_ttl_hours_kwarg() -> None:
    """Passing dedup_ttl_hours to WhatsAppChannel must fail loudly.

    Before the fix the constructor stored the value and never used it; the
    store's TTL kept controlling webhook dedup. Rejecting the kwarg makes the
    single source of truth unmistakable.
    """
    with pytest.raises(TypeError, match="dedup_ttl_hours"):
        WhatsAppChannel(
            verify_token="tok",
            verify_signature=False,
            dedup_ttl_hours=0.001,
        )


def test_whatsapp_channel_signature_omits_dedup_ttl_hours() -> None:
    params = inspect.signature(WhatsAppChannel.__init__).parameters
    assert "dedup_ttl_hours" not in params
    assert not hasattr(
        WhatsAppChannel(verify_token="tok", verify_signature=False),
        "dedup_ttl_hours",
    )


def test_session_store_still_owns_dedup_ttl() -> None:
    store = SessionStore(dedup_ttl_hours=3)
    assert store.dedup_ttl_hours == 3


def test_extra_positional_after_on_24h_window_expired_is_rejected() -> None:
    through_window_hook = (
        "tok",
        None,
        None,
        None,
        False,
        "v21.0",
        "fallback",
        "media",
        120.0,
        3,
        1.0,
        30.0,
        None,
    )
    with pytest.raises(TypeError, match="positional"):
        WhatsAppChannel(*through_window_hook, 12345)


def _channel(store: SessionStore | SQLiteSessionStore) -> tuple[WhatsAppChannel, CountingEngine]:
    engine = CountingEngine()
    runner = Runner(
        starting_agent=Agent(name="Bot", instructions="test"),
        engine=engine,
        session_store=store,
    )
    channel = WhatsAppChannel(verify_token="tok", verify_signature=False)
    channel.attach(runner)
    return channel, engine


def _post_wamid(client: TestClient, wamid: str) -> int:
    response = client.post("/webhook", json=_meta_payload(wamid))
    assert response.status_code == 200
    queued = response.json()["messages_queued"]
    assert isinstance(queued, int)
    return queued


def test_webhook_respects_session_store_dedup_ttl() -> None:
    store = SessionStore(dedup_ttl_hours=24)
    channel, engine = _channel(store)
    client = TestClient(channel.app)
    wamid = "wamid.ttl_memory"

    assert _post_wamid(client, wamid) == 1
    assert _post_wamid(client, wamid) == 0
    assert engine.call_count == 1

    with store._dedup_lock:
        store._processed_ids[wamid] -= store._dedup_ttl_seconds + 1.0

    assert _post_wamid(client, wamid) == 1
    assert engine.call_count == 2


def test_webhook_respects_sqlite_session_store_dedup_ttl(tmp_path: Path) -> None:
    db = tmp_path / "dedup.db"
    store = SQLiteSessionStore(db, dedup_ttl_hours=24)
    try:
        channel, engine = _channel(store)
        client = TestClient(channel.app)
        wamid = "wamid.ttl_sqlite"

        assert _post_wamid(client, wamid) == 1
        assert _post_wamid(client, wamid) == 0
        assert engine.call_count == 1

        _age(db, wamid, 25)

        assert _post_wamid(client, wamid) == 1
        assert engine.call_count == 2
    finally:
        store.close()
