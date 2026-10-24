"""Regression tests for WhatsApp reaction/system ignore and malformed nested shapes (#121)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging

import pytest
from starlette.testclient import TestClient

from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner

APP_SECRET = "test_app_secret_xyz"
VERIFY_TOKEN = "test_verify_token"


class MockEngine(GeminiEngine):
    def __init__(self, reply: str = "reply") -> None:
        super().__init__()
        self.reply = reply

    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text=self.reply)


def _make_channel() -> WhatsAppChannel:
    agent = Agent(name="Bot", instructions="test")
    runner = Runner(starting_agent=agent, engine=MockEngine())
    ch = WhatsAppChannel(
        verify_token=VERIFY_TOKEN,
        app_secret=APP_SECRET,
        access_token="fake-token",
        phone_number_id="1",
        verify_signature=True,
        unsupported_media_message="unsupported-media",
    )
    ch.attach(runner)
    return ch


def _sign(payload_bytes: bytes, secret: str = APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


def _post_signed(client: TestClient, payload: dict) -> object:
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": _sign(body),
        },
    )


def _meta_payload(message: dict) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"value": {"messages": [message]}}]}],
    }


@pytest.mark.asyncio
async def test_reaction_produces_no_outbound() -> None:
    channel = _make_channel()
    sent: list[str] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append(text)

    channel._send_single_outbound_whatsapp = capture  # type: ignore[method-assign]
    reaction = _meta_payload(
        {
            "from": "5491100000000",
            "id": "wamid.R1",
            "type": "reaction",
            "reaction": {"message_id": "wamid.prev", "emoji": "👍"},
        }
    )
    extracted = channel._extract_messages(reaction)
    assert extracted == []
    for wamid, sender, text in extracted:
        await channel._process_message_background(wamid, sender, text)
    assert sent == []


@pytest.mark.asyncio
async def test_reaction_removal_produces_no_outbound() -> None:
    channel = _make_channel()
    sent: list[str] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append(text)

    channel._send_single_outbound_whatsapp = capture  # type: ignore[method-assign]
    removal = _meta_payload(
        {
            "from": "5491100000000",
            "id": "wamid.R2",
            "type": "reaction",
            "reaction": {"message_id": "wamid.prev"},
        }
    )
    assert channel._extract_messages(removal) == []
    assert sent == []


@pytest.mark.asyncio
async def test_system_event_produces_no_outbound() -> None:
    channel = _make_channel()
    sent: list[str] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append(text)

    channel._send_single_outbound_whatsapp = capture  # type: ignore[method-assign]
    system = _meta_payload(
        {
            "from": "5491100000000",
            "id": "wamid.S1",
            "type": "system",
            "system": {"body": "number changed"},
        }
    )
    assert channel._extract_messages(system) == []
    assert sent == []


@pytest.mark.asyncio
async def test_image_still_gets_unsupported_reply() -> None:
    channel = _make_channel()
    sent: list[str] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append(text)

    channel._send_single_outbound_whatsapp = capture  # type: ignore[method-assign]
    image = _meta_payload(
        {
            "from": "5491100000000",
            "id": "wamid.I1",
            "type": "image",
            "image": {"id": "img"},
        }
    )
    extracted = channel._extract_messages(image)
    assert len(extracted) == 1
    assert extracted[0][2] is None
    await channel._process_message_background(*extracted[0])
    assert sent == ["unsupported-media"]


@pytest.mark.asyncio
async def test_text_still_dispatches() -> None:
    channel = _make_channel()
    sent: list[str] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append(text)

    channel._send_single_outbound_whatsapp = capture  # type: ignore[method-assign]
    text_msg = _meta_payload(
        {
            "from": "5491100000000",
            "id": "wamid.T1",
            "type": "text",
            "text": {"body": "hola"},
        }
    )
    extracted = channel._extract_messages(text_msg)
    assert extracted == [("wamid.T1", "5491100000000", "hola")]
    await channel._process_message_background(*extracted[0])
    assert sent == ["reply"]


@pytest.mark.parametrize(
    "payload,kind",
    [
        ({"entry": "not-a-list"}, "entry"),
        ({"entry": [{"changes": [{"value": None}]}]}, "value"),
        ({"entry": [{"changes": "nope"}]}, "changes"),
        ({"entry": [{"changes": ["string-item"]}]}, "changes item"),
        ({"entry": [{"changes": [{"value": {"messages": "nope"}}]}]}, "messages"),
        ({"entry": [{"changes": [{"value": {"messages": ["string-msg"]}}]}]}, "messages item"),
        (
            {
                "entry": [
                    {
                        "changes": [
                            {
                                "value": {
                                    "messages": [
                                        {
                                            "from": "5491100000000",
                                            "id": "wamid.bad",
                                            "type": "text",
                                            "text": "not-an-object",
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
            "text",
        ),
    ],
)
def test_malformed_shapes_return_200_with_warning(
    payload: dict, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    channel = _make_channel()
    client = TestClient(channel.app)
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.whatsapp"):
        response = _post_signed(client, payload)
    assert response.status_code == 200
    assert any("Ignoring malformed WhatsApp webhook" in rec.message for rec in caplog.records)
    # Extract path itself must not raise.
    assert channel._extract_messages(payload) == []


def test_hmac_still_required_before_parse() -> None:
    channel = _make_channel()
    client = TestClient(channel.app)
    body = json.dumps(
        _meta_payload({"from": "1", "id": "x", "type": "text", "text": {"body": "a"}})
    ).encode()
    response = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert response.status_code == 403


def _attach_spies(channel: WhatsAppChannel) -> tuple[list[str], list[tuple]]:
    sent: list[str] = []
    dispatches: list[tuple] = []

    async def capture(to_phone: str, text: str) -> None:
        sent.append(text)

    original_dispatch = channel.dispatch_async

    async def spy_dispatch(*, session_id: str, message: str, metadata=None):
        dispatches.append((session_id, message, metadata))
        return await original_dispatch(session_id=session_id, message=message, metadata=metadata)

    channel._send_single_outbound_whatsapp = capture  # type: ignore[method-assign]
    channel.dispatch_async = spy_dispatch  # type: ignore[method-assign]
    return sent, dispatches


@pytest.mark.parametrize(
    "message",
    [
        {
            "from": "5491100000000",
            "id": "wamid.R1",
            "type": "reaction",
            "reaction": {"message_id": "wamid.prev", "emoji": "👍"},
        },
        {
            "from": "5491100000000",
            "id": "wamid.R2",
            "type": "reaction",
            "reaction": {"message_id": "wamid.prev"},
        },
        {
            "from": "5491100000000",
            "id": "wamid.S1",
            "type": "system",
            "system": {"body": "number changed"},
        },
    ],
)
def test_http_ignored_meta_events_no_runner_or_outbound(message: dict) -> None:
    """Signed HTTP path: ignored Meta events yield 200, no dispatch, no outbound."""
    channel = _make_channel()
    sent, dispatches = _attach_spies(channel)
    client = TestClient(channel.app)
    response = _post_signed(client, _meta_payload(message))
    assert response.status_code == 200
    assert response.json()["status"] == "ignored_or_status_update"
    assert dispatches == []
    assert sent == []


def test_http_ignored_reaction_with_top_level_from_text_no_fallback() -> None:
    """Meta reaction plus top-level from/text must not revive generic extraction."""
    channel = _make_channel()
    sent, dispatches = _attach_spies(channel)
    client = TestClient(channel.app)
    payload = _meta_payload(
        {
            "from": "5491100000000",
            "id": "wamid.R3",
            "type": "reaction",
            "reaction": {"message_id": "wamid.prev", "emoji": "👍"},
        }
    )
    payload["from"] = "5491100000000"
    payload["text"] = "should-not-run"
    assert channel._extract_messages(payload) == []
    response = _post_signed(client, payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ignored_or_status_update"
    assert dispatches == []
    assert sent == []


def test_http_text_still_dispatches_via_signed_webhook() -> None:
    channel = _make_channel()
    sent, dispatches = _attach_spies(channel)
    client = TestClient(channel.app)
    response = _post_signed(
        client,
        _meta_payload(
            {
                "from": "5491100000000",
                "id": "wamid.T2",
                "type": "text",
                "text": {"body": "hola"},
            }
        ),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(dispatches) == 1
    assert dispatches[0][1] == "hola"
    assert sent == ["reply"]
