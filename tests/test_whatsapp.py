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


def test_whatsapp_default_background_timeout() -> None:
    channel = WhatsAppChannel(verify_token="fixture", app_secret="fixture")
    assert channel.llm_timeout_seconds == 120.0


@pytest.mark.parametrize("timeout", [0.05, 30.0, 90.0, 300.0])
def test_whatsapp_explicit_background_timeout(timeout: float) -> None:
    channel = WhatsAppChannel(
        verify_token="fixture", app_secret="fixture", llm_timeout_seconds=timeout
    )
    assert channel.llm_timeout_seconds == timeout


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
    assert data["sender"] == "5491***678"
    assert data["messages_queued"] == 1
    # BG task ran synchronously inside TestClient — verify session state
    session = configured_channel.runner.get_session("5491112345678")
    assert session is not None
    assert session.history[-1].content == "Hello from ChatFlow WhatsApp!"
    assert session.active_agent.name == "WhatsApp Bot"


def test_generic_webhook_format_post(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    payload = {"phone": "5491199887766", "message": "Can I return an item?"}
    res = client.post("/webhook", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["sender"] == "5491***766"
    assert data["messages_queued"] == 1
    session = configured_channel.runner.get_session("5491199887766")
    assert session is not None
    assert session.history[-1].content == "Hello from ChatFlow WhatsApp!"
    assert session.active_agent.name == "WhatsApp Bot"

def test_identical_generic_webhooks_are_processed_independently(
    configured_channel: WhatsAppChannel,
) -> None:
    """Repeated generic messages without IDs must not be deduplicated."""
    client = TestClient(configured_channel.app)
    payload = {
        "phone": "5491199887766",
        "message": "hola",
    }

    first = client.post("/webhook", json=payload)
    second = client.post("/webhook", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["messages_queued"] == 1
    assert second.json()["messages_queued"] == 1


def test_generic_webhooks_with_explicit_ids_are_deduplicated(
    configured_channel: WhatsAppChannel,
) -> None:
    """Repeated provider-supplied IDs must be processed only once."""
    client = TestClient(configured_channel.app)
    payload = {
        "phone": "5491199887766",
        "message": "hola",
        "id": "wamid.test123",
    }

    first = client.post("/webhook", json=payload)
    second = client.post("/webhook", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["messages_queued"] == 1
    assert second.json()["messages_queued"] == 0


def test_status_update_webhook_ignored(configured_channel: WhatsAppChannel) -> None:
    client = TestClient(configured_channel.app)
    # Status receipt without messages array
    payload = {
        "entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]
    }
    res = client.post("/webhook", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "ignored_or_status_update"


@pytest.mark.parametrize(
    "body",
    [
        b"[1, 2, 3]",
        b"[]",
        b'"hello"',
        b'""',
        b"42",
        b"0",
        b"3.14",
        b"true",
        b"false",
        b"null",
    ],
)
def test_webhook_rejects_non_object_json_payload(
    body: bytes, configured_channel: WhatsAppChannel
) -> None:
    """Valid JSON whose root is not an object must yield 400, never 500."""
    client = TestClient(configured_channel.app, raise_server_exceptions=False)
    res = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 400
    assert res.json() == {"detail": "JSON payload must be an object"}
    assert configured_channel.runner.get_session("5491112345678") is None


def test_webhook_malformed_json_still_reports_invalid_json(
    configured_channel: WhatsAppChannel,
) -> None:
    """Unparseable bodies keep the original 400 / "Invalid JSON payload" path."""
    client = TestClient(configured_channel.app, raise_server_exceptions=False)
    res = client.post(
        "/webhook",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 400
    assert res.json() == {"detail": "Invalid JSON payload"}


def _meta_message_payload(msg: dict[str, object]) -> dict[str, object]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123456",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [msg],
                        },
                    }
                ],
            }
        ],
    }


def test_meta_fallback_wamid_is_key_order_independent_and_collision_free(
    configured_channel: WhatsAppChannel,
) -> None:
    """Fallback wamid for Meta messages without id is canonical-JSON SHA-256, independent of key order (#175)."""
    msg_order_a = {
        "from": "5491112345678",
        "timestamp": "1710000000",
        "type": "text",
        "text": {"body": "Need info on pricing", "preview_url": False},
    }
    msg_order_b = {
        "text": {"preview_url": False, "body": "Need info on pricing"},
        "type": "text",
        "timestamp": "1710000000",
        "from": "5491112345678",
    }

    extracted_a = configured_channel._extract_messages(_meta_message_payload(msg_order_a))
    extracted_b = configured_channel._extract_messages(_meta_message_payload(msg_order_b))

    assert len(extracted_a) == 1
    assert len(extracted_b) == 1
    wamid_a = extracted_a[0][0]
    wamid_b = extracted_b[0][0]
    assert wamid_a == wamid_b
    assert wamid_a.startswith("gen_")
    assert len(wamid_a) == 4 + 64
    # Sensitive phone number and message text must not appear in the generated identifier
    assert "5491112345678" not in wamid_a
    assert "Need info on pricing" not in wamid_a

    # Distinct messages produce distinct fallback identifiers
    variants = [
        {
            "from": "5491112345679",
            "timestamp": "1710000000",
            "type": "text",
            "text": {"body": "Need info on pricing", "preview_url": False},
        },
        {
            "from": "5491112345678",
            "timestamp": "1710000001",
            "type": "text",
            "text": {"body": "Need info on pricing", "preview_url": False},
        },
        {
            "from": "5491112345678",
            "timestamp": "1710000000",
            "type": "text",
            "text": {"body": "Different body", "preview_url": False},
        },
        {
            "from": "5491112345678",
            "timestamp": "1710000000",
            "type": "image",
            "image": {"id": "media_1"},
        },
    ]
    seen_ids = {wamid_a}
    for variant in variants:
        [(variant_wamid, _, _)] = configured_channel._extract_messages(
            _meta_message_payload(variant)
        )
        assert variant_wamid.startswith("gen_")
        assert variant_wamid not in seen_ids
        seen_ids.add(variant_wamid)


def test_meta_real_wamid_is_preserved_when_present(
    configured_channel: WhatsAppChannel,
) -> None:
    """When a Meta message includes an id, the real wamid is preserved untouched (#175)."""
    msg_with_id = {
        "id": "wamid.HBgNNTQ5MTExMjM0NTY3OBUCABIYFjNFQjA",
        "from": "5491112345678",
        "timestamp": "1710000000",
        "type": "text",
        "text": {"body": "Hello"},
    }
    [(wamid, sender, body)] = configured_channel._extract_messages(
        _meta_message_payload(msg_with_id)
    )
    assert wamid == "wamid.HBgNNTQ5MTExMjM0NTY3OBUCABIYFjNFQjA"
    assert sender == "5491112345678"
    assert body == "Hello"


def test_meta_fallback_wamid_is_stable_across_processes(
    configured_channel: WhatsAppChannel,
) -> None:
    """Fallback wamid remains identical across processes with different PYTHONHASHSEED values (#175)."""
    import os
    import subprocess
    import sys

    msg = {
        "from": "5491112345678",
        "timestamp": "1710000000",
        "type": "text",
        "text": {"body": "Cross-process stability check"},
    }
    [(expected_wamid, _, _)] = configured_channel._extract_messages(_meta_message_payload(msg))

    code = (
        "from chatflow_agent.channels.whatsapp import _fallback_wamid; "
        "msg = {'text': {'body': 'Cross-process stability check'}, "
        "'type': 'text', 'timestamp': '1710000000', 'from': '5491112345678'}; "
        "print(_fallback_wamid(msg))"
    )
    for seed in ("1", "42", "999999"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert proc.stdout.strip() == expected_wamid


def test_meta_fallback_wamid_deduplicates_retries_without_id(
    configured_channel: WhatsAppChannel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Meta webhook retries lacking id are deduplicated even when key order differs, without logging content (#175)."""
    client = TestClient(configured_channel.app)
    first_payload = _meta_message_payload(
        {
            "from": "5491112345678",
            "timestamp": "1710000000",
            "type": "text",
            "text": {"body": "Sensitive secret message body"},
        }
    )
    second_payload = _meta_message_payload(
        {
            "text": {"body": "Sensitive secret message body"},
            "type": "text",
            "timestamp": "1710000000",
            "from": "5491112345678",
        }
    )

    with caplog.at_level("INFO", logger="chatflow_agent.whatsapp"):
        first = client.post("/webhook", json=first_payload)
        second = client.post("/webhook", json=second_payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["messages_queued"] == 1
    assert second.json()["messages_queued"] == 0
    assert "Skipping duplicate wamid: gen_" in caplog.text
    assert "Sensitive secret message body" not in caplog.text
    assert "5491112345678" not in caplog.text

