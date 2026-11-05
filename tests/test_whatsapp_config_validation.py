"""WhatsAppChannel validates outbound retry and body-size settings at construction (#175).

These settings used to be stored unchecked. A non-positive ``max_body_size`` rejected every
webhook, ``inf`` disabled the body-size guard, a NaN ``outbound_max_delay`` turned the
backoff into ``asyncio.sleep(nan)`` (which never returns on Python 3.12 and raises on 3.13),
negative delays removed the backoff, and a very large ``max_outbound_retries`` overflowed
``base * 2**retries``.
"""

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


def test_defaults_are_unchanged_and_valid() -> None:
    channel = _channel()
    assert channel.max_outbound_retries == 3
    assert channel.outbound_base_delay == 1.0
    assert channel.outbound_max_delay == 30.0
    assert channel.max_body_size == DEFAULT_MAX_BODY_SIZE == 1024 * 1024


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("max_outbound_retries", 0),  # zero retries stays allowed
        ("max_outbound_retries", 1),
        ("max_outbound_retries", 10),  # upper bound, inclusive
        ("outbound_base_delay", 0),  # zero delays stay allowed (used by tests)
        ("outbound_base_delay", 0.0),
        ("outbound_base_delay", 0.01),
        ("outbound_base_delay", 5),  # an int is a real number
        ("outbound_base_delay", Fraction(1, 2)),
        ("outbound_max_delay", 0.0),
        ("outbound_max_delay", 0.5),  # below the base delay is a legitimate cap
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
        # max_outbound_retries: int in [0, 10], not bool
        ("max_outbound_retries", -1),
        ("max_outbound_retries", 11),
        ("max_outbound_retries", 2000),  # would overflow base * 2**retries
        ("max_outbound_retries", True),
        ("max_outbound_retries", False),
        ("max_outbound_retries", 3.0),
        ("max_outbound_retries", NAN),
        ("max_outbound_retries", INF),
        ("max_outbound_retries", "3"),
        ("max_outbound_retries", None),
        # outbound_base_delay: finite real >= 0, not bool
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
        ("outbound_base_delay", 10**400),  # an int too large to convert to float
        # outbound_max_delay: finite real >= 0, not bool
        ("outbound_max_delay", -5.0),
        ("outbound_max_delay", NAN),
        ("outbound_max_delay", INF),
        ("outbound_max_delay", False),
        ("outbound_max_delay", "30"),
        ("outbound_max_delay", None),
        # max_body_size: positive int, not bool
        ("max_body_size", 0),
        ("max_body_size", -1),
        ("max_body_size", True),
        ("max_body_size", 1024.0),
        ("max_body_size", 2e6),
        ("max_body_size", INF),  # would disable the body-size guard
        ("max_body_size", NAN),
        ("max_body_size", "1048576"),
        ("max_body_size", None),
    ],
)
def test_invalid_values_raise_configuration_error_naming_the_parameter(
    param: str, value: object
) -> None:
    with pytest.raises(ConfigurationError, match=param) as exc_info:
        _channel(**{param: value})
    # ConfigurationError subclasses ValueError, so existing `except ValueError` still works.
    assert isinstance(exc_info.value, ValueError)
    assert repr(value) in str(exc_info.value)


def test_positional_values_are_validated_too() -> None:
    """The retry settings are positional-capable; their position must not skip validation."""
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
        (30.0, 0.5, None),  # cap below the base delay
        (1.0, 30.0, "100"),  # Retry-After above the cap
    ],
)
async def test_accepted_settings_only_produce_finite_non_negative_sleeps(
    base_delay: float, max_delay: float, retry_after: str | None
) -> None:
    """With the largest accepted retry count, every backoff sleep is finite and non-negative."""
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

    assert ok is False  # retries exhausted
    assert mock_client.post.call_count == 11  # 1 attempt + 10 retries, no OverflowError
    delays = [call.args[0] for call in sleep.call_args_list]
    assert len(delays) == 10
    assert all(math.isfinite(d) and 0 <= d <= max_delay for d in delays)
