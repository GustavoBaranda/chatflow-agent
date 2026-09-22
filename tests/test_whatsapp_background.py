"""Tests for BackgroundTasks dispatch, dedup by wamid, and HTTP 200 isolation."""

import json
from unittest.mock import patch

from starlette.testclient import TestClient

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner


class CountingEngine(GeminiEngine):
    """Engine that counts invocations and returns a canned reply."""

    def __init__(self, reply: str = "ok") -> None:
        super().__init__()
        self.reply = reply
        self.call_count = 0

    async def generate_turn_async(self, agent, history):
        self.call_count += 1
        return EngineTurnResult(text=self.reply)


def _make_channel(engine: GeminiEngine) -> WhatsAppChannel:
    agent = Agent(name="Bot", instructions="test")
    runner = Runner(starting_agent=agent, engine=engine)
    ch = WhatsAppChannel(verify_token="tok", verify_signature=False)
    ch.attach(runner)
    return ch


def _meta_payload(wamid: str, phone: str = "5491100001111", text: str = "Hola") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"value": {
            "messages": [{"id": wamid, "from": phone, "type": "text",
                          "text": {"body": text}}]
        }}]}],
    }


# ── HTTP 200 returned before engine executes ──────────────────────────────────

def test_webhook_returns_200_before_engine_executes() -> None:
    """HTTP 200 must be returned immediately; engine must not run before the response.

    Mocks BackgroundTasks.add_task to capture without executing — verifies decoupling.
    """
    engine = CountingEngine()
    ch = _make_channel(engine)
    client = TestClient(ch.app, raise_server_exceptions=True)

    test_wamid = "wamid.bg_isolation_001"
    captured: list = []

    def capturing_add_task(_self_or_func, func=None, **kwargs):
        # add_task is called as an instance method: (self, func, **kwargs)
        actual_func = func if func is not None else _self_or_func
        captured.append((actual_func, kwargs))  # capture but do NOT run

    with patch("fastapi.BackgroundTasks.add_task", capturing_add_task):
        body = json.dumps(_meta_payload(test_wamid)).encode()
        resp = client.post(
            "/webhook", content=body, headers={"Content-Type": "application/json"}
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert engine.call_count == 0, "Engine must not have run before 200 was returned"
    assert len(captured) == 1, "Exactly one BG task must be enqueued"
    store = ch.runner.session_store
    assert store is not None
    assert store.is_message_processed(test_wamid), "wamid must be recorded before BG task runs"


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_whatsapp_deduplication_by_wamid() -> None:
    """Second delivery of the same wamid must be discarded (messages_queued=0)."""
    engine = CountingEngine()
    ch = _make_channel(engine)
    client = TestClient(ch.app)

    test_wamid = "wamid.dedup_001"
    r1 = client.post("/webhook", json=_meta_payload(test_wamid))
    assert r1.status_code == 200
    assert r1.json()["messages_queued"] == 1

    r2 = client.post("/webhook", json=_meta_payload(test_wamid))  # Meta retry
    assert r2.status_code == 200
    assert r2.json()["messages_queued"] == 0  # deduplicated

    assert engine.call_count == 1


# ── Multi-message batch ───────────────────────────────────────────────────────

def test_whatsapp_processes_multiple_messages_in_single_batch() -> None:
    """Payload with two distinct wamids must enqueue two BG tasks."""
    engine = CountingEngine()
    ch = _make_channel(engine)
    client = TestClient(ch.app)

    payload = {
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"value": {
            "messages": [
                {"id": "wamid.batch001", "from": "549111", "type": "text",
                 "text": {"body": "msg1"}},
                {"id": "wamid.batch002", "from": "549111", "type": "text",
                 "text": {"body": "msg2"}},
            ]
        }}]}],
    }
    r = client.post("/webhook", json=payload)
    assert r.status_code == 200
    assert r.json()["messages_queued"] == 2
    assert engine.call_count == 2


# ── Status notifications ignored ──────────────────────────────────────────────

def test_status_notification_returns_ignored() -> None:
    """Status-update webhooks return ignored_or_status_update without calling the engine."""
    engine = CountingEngine()
    ch = _make_channel(engine)
    client = TestClient(ch.app)

    r = client.post("/webhook", json={
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"value": {
            "statuses": [{"id": "wamid.s001", "status": "delivered"}]
        }}]}],
    })
    assert r.status_code == 200
    assert r.json()["status"] == "ignored_or_status_update"
    assert engine.call_count == 0
