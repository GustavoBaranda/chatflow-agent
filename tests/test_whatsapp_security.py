"""Tests for WhatsApp webhook security: HMAC signature enforcement and fail-closed behavior."""

import hashlib
import hmac
import json

import pytest
from starlette.testclient import TestClient

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner

APP_SECRET = "test_app_secret_xyz"
VERIFY_TOKEN = "test_verify_token"


class MockEngine(GeminiEngine):
    def __init__(self, reply: str = "ok") -> None:
        super().__init__()
        self.reply = reply

    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text=self.reply)


def _make_channel(app_secret: str = APP_SECRET) -> WhatsAppChannel:
    agent = Agent(name="Bot", instructions="test")
    runner = Runner(starting_agent=agent, engine=MockEngine())
    ch = WhatsAppChannel(
        verify_token=VERIFY_TOKEN,
        app_secret=app_secret,
        verify_signature=True,
    )
    ch.attach(runner)
    return ch


def _sign(payload_bytes: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


SAMPLE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "changes": [
                {
                    "value": {
                        "messages": [
                            {"id": "wamid.test001", "from": "5491100001111",
                             "type": "text", "text": {"body": "Hola"}}
                        ]
                    }
                }
            ]
        }
    ],
}


# ── SEC-01: Fail-closed: no app_secret raises ValueError ──────────────────────

def test_fail_closed_raises_without_app_secret() -> None:
    """WhatsAppChannel MUST raise ValueError when app_secret is omitted and verify_signature=True."""
    with pytest.raises(ValueError, match="app_secret"):
        WhatsAppChannel(verify_token=VERIFY_TOKEN)


def test_explicit_verify_signature_false_does_not_raise() -> None:
    """verify_signature=False must not raise even without app_secret."""
    # Should not raise
    agent = Agent(name="Bot", instructions="test")
    runner = Runner(starting_agent=agent, engine=MockEngine())
    ch = WhatsAppChannel(verify_token=VERIFY_TOKEN, verify_signature=False)
    ch.attach(runner)
    assert ch.verify_signature is False


# ── SEC-01: HMAC validation ────────────────────────────────────────────────────

def test_valid_signature_accepted() -> None:
    """A POST with a correct HMAC-SHA256 signature is accepted (200)."""
    ch = _make_channel()
    client = TestClient(ch.app)
    body = json.dumps(SAMPLE_PAYLOAD).encode()
    sig = _sign(body, APP_SECRET)
    resp = client.post("/webhook", content=body,
                       headers={"Content-Type": "application/json",
                                "X-Hub-Signature-256": sig})
    assert resp.status_code == 200


def test_missing_signature_rejected() -> None:
    """A POST without X-Hub-Signature-256 must return 403."""
    ch = _make_channel()
    client = TestClient(ch.app)
    body = json.dumps(SAMPLE_PAYLOAD).encode()
    resp = client.post("/webhook", content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 403


def test_wrong_signature_rejected() -> None:
    """A POST with a tampered signature must return 403."""
    ch = _make_channel()
    client = TestClient(ch.app)
    body = json.dumps(SAMPLE_PAYLOAD).encode()
    resp = client.post("/webhook", content=body,
                       headers={"Content-Type": "application/json",
                                "X-Hub-Signature-256": "sha256=deadbeef"})
    assert resp.status_code == 403


def test_wrong_secret_rejected() -> None:
    """A POST signed with a different secret must return 403."""
    ch = _make_channel()
    client = TestClient(ch.app)
    body = json.dumps(SAMPLE_PAYLOAD).encode()
    sig = _sign(body, "wrong_secret")
    resp = client.post("/webhook", content=body,
                       headers={"Content-Type": "application/json",
                                "X-Hub-Signature-256": sig})
    assert resp.status_code == 403


# ── SEC-02: Constant-time GET /webhook ────────────────────────────────────────

def test_valid_verify_token_accepted() -> None:
    """GET /webhook with correct verify_token must return 200 and the challenge."""
    ch = _make_channel()
    client = TestClient(ch.app)
    resp = client.get(
        "/webhook",
        params={"hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "challenge_abc"},
    )
    assert resp.status_code == 200
    assert resp.text == "challenge_abc"


def test_wrong_verify_token_rejected() -> None:
    """GET /webhook with wrong verify_token must return 403."""
    ch = _make_channel()
    client = TestClient(ch.app)
    resp = client.get(
        "/webhook",
        params={"hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "x"},
    )
    assert resp.status_code == 403
