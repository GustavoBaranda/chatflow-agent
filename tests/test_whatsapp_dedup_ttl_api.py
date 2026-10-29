"""WhatsAppChannel no longer accepts an unused dedup_ttl_hours knob."""

from __future__ import annotations

import inspect

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.memory import SessionStore


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
