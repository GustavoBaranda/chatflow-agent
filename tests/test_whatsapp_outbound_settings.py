"""WhatsAppChannel outbound setting validation (#175 item 2)."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chatflow_agent import ConfigurationError
from chatflow_agent.channels.whatsapp import DEFAULT_MAX_BODY_SIZE, WhatsAppChannel

NAN = float("nan")
INF = float("inf")


def _channel(**kwargs: Any) -> WhatsAppChannel:
    return WhatsAppChannel(verify_token="tok", verify_signature=False, **kwargs)


def test_outbound_defaults_are_accepted() -> None:
    channel = _channel()
    assert channel.max_outbound_retries == 3
    assert channel.outbound_base_delay == 1.0
    assert channel.outbound_max_delay == 30.0
    assert channel.max_body_size == DEFAULT_MAX_BODY_SIZE == 1024 * 1024


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("max_outbound_retries", 0),
        ("max_outbound_retries", 1),
        ("max_outbound_retries", 10),
        ("outbound_base_delay", 0),
        ("outbound_base_delay", 0.0),
        ("outbound_base_delay", 0.01),
        ("outbound_base_delay", 5),
        ("outbound_base_delay", Fraction(1, 2)),
        ("outbound_max_delay", 0.0),
        ("outbound_max_delay", 0.5),
        ("outbound_max_delay", 120),
        ("max_body_size", 1),
        ("max_body_size", 1024),
        ("max_body_size", 10 * 1024 * 1024),
    ],
)
def test_valid_values_are_accepted_and_stored(param: str, value: object) -> None:
    channel = _channel(**{param: value})
    assert getattr(channel, param) == value


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("max_outbound_retries", -1),
        ("max_outbound_retries", 11),
        ("max_outbound_retries", 2000),
        ("max_outbound_retries", True),
        ("max_outbound_retries", False),
        ("max_outbound_retries", 3.0),
        ("max_outbound_retries", NAN),
        ("max_outbound_retries", INF),
        ("max_outbound_retries", "3"),
        ("max_outbound_retries", None),
        ("outbound_base_delay", -0.01),
        ("outbound_base_delay", -1),
        ("outbound_base_delay", NAN),
        ("outbound_base_delay", INF),
        ("outbound_base_delay", -INF),
        ("outbound_base_delay", True),
        ("outbound_base_delay", "1.0"),
        ("outbound_base_delay", None),
        ("outbound_base_delay", Decimal("1.0")),
        ("outbound_base_delay", 1j),
        ("outbound_base_delay", 10**400),
        ("outbound_max_delay", -5.0),
        ("outbound_max_delay", NAN),
        ("outbound_max_delay", INF),
        ("outbound_max_delay", False),
        ("outbound_max_delay", "30"),
        ("outbound_max_delay", None),
        ("max_body_size", 0),
        ("max_body_size", -1),
        ("max_body_size", True),
        ("max_body_size", 1024.0),
        ("max_body_size", 2e6),
        ("max_body_size", INF),
        ("max_body_size", NAN),
        ("max_body_size", "1048576"),
        ("max_body_size", None),
    ],
)
def test_invalid_values_raise_configuration_error_naming_parameter(
    param: str, value: object
) -> None:
    with pytest.raises(ConfigurationError, match=param) as exc_info:
        _channel(**{param: value})
    assert isinstance(exc_info.value, ValueError)
    assert repr(value) in str(exc_info.value)


def test_positional_values_are_validated() -> None:
    positional = ("tok", None, None, None, False, "v21.0", "fb", "media", 120.0)
    assert _positional(positional, 3, 1.0, 30.0).max_outbound_retries == 3
    with pytest.raises(ConfigurationError, match="outbound_max_delay"):
        _positional(positional, 3, 1.0, NAN)


def _positional(head: tuple[Any, ...], *tail: Any) -> WhatsAppChannel:
    return WhatsAppChannel(*head, *tail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_delay", "max_delay", "retry_after"),
    [
        (0.0, 0.0, None),
        (1.0, 30.0, None),
        (30.0, 0.5, None),
        (1.0, 30.0, "100"),
    ],
)
async def test_accepted_settings_produce_bounded_finite_backoff(
    base_delay: float, max_delay: float, retry_after: str | None
) -> None:
    channel = WhatsAppChannel(
        verify_token="tok",
        access_token="test_token",
        phone_number_id="12345",
        verify_signature=False,
        max_outbound_retries=10,
        outbound_base_delay=base_delay,
        outbound_max_delay=max_delay,
    )
    headers = {"Retry-After": retry_after} if retry_after else {}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=httpx.Response(429, headers=headers))
    sleep = AsyncMock()

    with (
        patch("httpx.AsyncClient") as mock_cls,
        patch("chatflow_agent.channels.whatsapp.asyncio.sleep", sleep),
    ):
        mock_cls.return_value.__aenter__.return_value = mock_client
        ok = await channel._send_single_outbound_whatsapp("5491100000000", "Hello")

    assert ok is False
    assert mock_client.post.call_count == 11
    delays = [call.args[0] for call in sleep.call_args_list]
    assert len(delays) == 10
    assert all(math.isfinite(delay) and 0 <= delay <= max_delay for delay in delays)
