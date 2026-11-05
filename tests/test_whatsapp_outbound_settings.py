"""WhatsAppChannel outbound setting validation (#175 item 2)."""

from __future__ import annotations

import math

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel


def _channel(**kwargs: object) -> WhatsAppChannel:
    return WhatsAppChannel(verify_token="tok", verify_signature=False, **kwargs)  # type: ignore[arg-type]


def test_outbound_defaults_are_accepted() -> None:
    ch = _channel()
    assert ch.max_outbound_retries == 3
    assert ch.outbound_base_delay == 1.0
    assert ch.outbound_max_delay == 30.0
    assert ch.max_body_size > 0


def test_zero_retries_is_allowed() -> None:
    ch = _channel(max_outbound_retries=0)
    assert ch.max_outbound_retries == 0


@pytest.mark.parametrize("value", [-1, -3, True, 1.5, "3"])
def test_max_outbound_retries_must_be_non_negative_int(value: object) -> None:
    with pytest.raises(ValueError, match="max_outbound_retries"):
        _channel(max_outbound_retries=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_outbound_base_delay_must_be_positive_finite(value: float) -> None:
    with pytest.raises(ValueError, match="outbound_base_delay"):
        _channel(outbound_base_delay=value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_outbound_max_delay_must_be_positive_finite(value: float) -> None:
    with pytest.raises(ValueError, match="outbound_max_delay"):
        _channel(outbound_max_delay=value)


def test_outbound_max_delay_must_be_at_least_base() -> None:
    with pytest.raises(ValueError, match="outbound_max_delay must be >="):
        _channel(outbound_base_delay=5.0, outbound_max_delay=1.0)


@pytest.mark.parametrize("value", [0, -1, -1024, True, 1.5, "1024"])
def test_max_body_size_must_be_positive_int(value: object) -> None:
    with pytest.raises(ValueError, match="max_body_size"):
        _channel(max_body_size=value)  # type: ignore[arg-type]


def test_negative_max_body_size_fails_at_construction() -> None:
    """Regression: a negative limit used to accept construction then 413 every body."""
    with pytest.raises(ValueError, match="max_body_size"):
        _channel(max_body_size=-1)
