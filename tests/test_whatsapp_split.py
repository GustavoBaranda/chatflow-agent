"""Tests for outbound message splitting and WhatsApp character limit handling."""

from unittest.mock import AsyncMock, patch

import pytest

from chatflow_agent.channels.whatsapp import WhatsAppChannel, split_message


def test_split_message_under_limit() -> None:
    """Short messages below 4096 characters remain unsplit."""
    short_text = "Hello world! This is a short response."
    result = split_message(short_text, max_len=4096)
    assert result == [short_text]


@pytest.mark.parametrize("max_len", [0, -1, -100])
def test_split_message_rejects_non_positive_max_len(max_len: int) -> None:
    """Non-positive max_len values raise ValueError."""
    with pytest.raises(ValueError, match="max_len must be greater than 0"):
        split_message("Hello world", max_len=max_len)


def test_split_message_prefers_paragraphs() -> None:
    """Messages over limit split at paragraph boundaries (\n\n)."""
    p1 = "A" * 2500
    p2 = "B" * 2500
    text = f"{p1}\n\n{p2}"

    chunks = split_message(text, max_len=4096)
    assert len(chunks) == 2
    assert chunks[0] == p1
    assert chunks[1] == p2
    assert all(len(c) <= 4096 for c in chunks)


def test_split_message_sentence_boundary() -> None:
    """Messages without paragraphs split gracefully at sentence boundaries."""
    s1 = "This is sentence one. " * 150  # ~3300 chars
    s2 = "This is sentence two. " * 150  # ~3300 chars
    text = s1 + s2

    chunks = split_message(text, max_len=4096)
    assert len(chunks) == 2
    assert all(len(c) <= 4096 for c in chunks)


def test_split_message_hard_fallback() -> None:
    """Strings with no whitespace or separators split cleanly without data loss."""
    text = "X" * 9000
    chunks = split_message(text, max_len=4096)
    assert len(chunks) == 3
    assert len(chunks[0]) == 4096
    assert len(chunks[1]) == 4096
    assert len(chunks[2]) == 9000 - 8192
    assert "".join(chunks) == text


def test_split_message_keeps_emoji_heavy_message_within_limit_in_one_chunk() -> None:
    """An emoji-heavy message within the supported character limit stays whole.

    The platforms' message limit is enforced in characters, not UTF-16
    code units: a reply made of 4096 emoji occupies 8192 UTF-16 units,
    yet the Telegram Bot API delivers it as a single message (verified
    2026-10-30), so split_message() must not over-split it.
    """
    text = "🎉" * 4096
    assert len(text) == 4096  # code points: the supported character limit

    chunks = split_message(text, max_len=4096)

    assert chunks == [text]
    assert "".join(chunks) == text


def test_split_message_progresses_on_an_astral_character_bigger_than_max_len() -> None:
    """A single character larger than max_len still makes progress."""
    text = "🎉" * 5
    chunks = split_message(text, max_len=1)
    assert chunks == ["🎉"] * 5
    assert "".join(chunks) == text


@pytest.mark.asyncio
async def test_send_outbound_whatsapp_splits_chunks() -> None:
    """Outbound dispatcher must invoke single-chunk sender once per split chunk."""
    channel = WhatsAppChannel(
        verify_token="test_tok",
        access_token="test_access",
        phone_number_id="12345",
        verify_signature=False,
    )

    long_reply = ("Para 1: " + "A" * 2500) + "\n\n" + ("Para 2: " + "B" * 2500)

    with patch.object(
        channel, "_send_single_outbound_whatsapp", new_callable=AsyncMock
    ) as mock_send_single:
        await channel._send_outbound_whatsapp("5491100000000", long_reply)

        assert mock_send_single.call_count == 2
        calls = mock_send_single.call_args_list
        assert calls[0].kwargs["to_phone"] == "5491100000000"
        assert calls[0].kwargs["text"].startswith("Para 1:")
        assert calls[1].kwargs["to_phone"] == "5491100000000"
        assert calls[1].kwargs["text"].startswith("Para 2:")
