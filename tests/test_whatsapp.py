"""Tests for WhatsAppChannel webhook endpoints and Meta Cloud API integration."""

import pytest
from starlette.testclient import TestClient

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner


class MockEngine(GeminiEngine):
    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply

    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text=self.reply)


@pytest.fixture
def configured_channel() -> WhatsAppChannel:
    agent = Agent(name="WhatsApp Bot", instructions="WhatsApp customer assistant.")
    runner = Runner(
        starting_agent=agent,
        engine=MockEngine("Hello from ChatFlow WhatsApp!"),
    )
    channel = WhatsAppChannel(verify_token="test_secret_token_123", verify_signature=False)
    channel.attach(runner)
    return channel


def test_whatsapp_health_endpoint(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy", "channel": "whatsapp"}


def test_meta_webhook_challenge_success(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "test_secret_token_123",
        "hub.challenge": "1155998822",
    }
    res = client.get("/webhook", params=params)
    assert res.status_code == 200
    assert res.text == "1155998822"


def test_meta_webhook_challenge_forbidden(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "wrong_token",
        "hub.challenge": "1155998822",
    }
    res = client.get("/webhook", params=params)
    assert res.status_code == 403


def test_meta_cloud_api_message_post(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123456",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "5491112345678",
                                    "id": "wamid.123",
                                    "timestamp": "1710000000",
                                    "text": {"body": "Need info on pricing"},
                                    "type": "text",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    res = client.post("/webhook", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["sender"] == "5491112345678"
    assert data["reply"] == "Hello from ChatFlow WhatsApp!"
    assert data["active_agent"] == "WhatsApp Bot"


def test_generic_webhook_format_post(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    payload = {"phone": "5491199887766", "message": "Can I return an item?"}
    res = client.post("/webhook", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["sender"] == "5491199887766"
    assert data["reply"] == "Hello from ChatFlow WhatsApp!"


def test_status_update_webhook_ignored(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    # Status receipt without messages array
    payload = {
        "entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]
    }
    res = client.post("/webhook", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "ignored_or_status_update"
