"""Abstract BaseChannel definition for chatflow-agent."""

import bisect
from abc import ABC, abstractmethod
from typing import Any

from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ChatFlowError
from chatflow_agent.types import AgentResponse


class ChannelError(ChatFlowError):
    """Raised when an operation on a channel fails or runner is detached."""


def _utf16_units(text: str) -> int:
    """Length of ``text`` in UTF-16 code units, the unit platforms count text in.

    A code point in the Basic Multilingual Plane is one unit; anything above it —
    most emoji — is two (a surrogate pair). Telegram documents this for the MTProto
    and Bot APIs: https://core.telegram.org/api/entities. The simple form is the
    one Telegram's own docs suggest: encode as UTF-16 and halve the byte count.
    """
    return len(text.encode("utf-16-le")) // 2


def split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split long outbound text into sequential chunks within max_len.

    Splits progressively by paragraph (\n\n), line (\n), sentence (. ), or word ( )
    to preserve formatting and readability across messaging channels.

    ``max_len`` is measured in UTF-16 code units, not Python code points: the
    messaging platforms count a message's length in UTF-16 units, where a
    character above the Basic Multilingual Plane (most emoji) is two. A chunk of
    "🎉" * 4096 is 8192 units — a reply like that went out as one message, and
    Telegram rejects it with "message is too long". Astral characters therefore
    shorten every chunk, which can only keep chunks inside the limit.
    """
    if max_len <= 0:
        raise ValueError("max_len must be greater than 0")

    if _utf16_units(text) <= max_len:
        return [text]

    # UTF-16 unit length of every prefix of text: units[i] is the length of
    # text[:i], so the chunk window can be moved without re-encoding anything.
    units = [0]
    for ch in text:
        units.append(units[-1] + (2 if ord(ch) > 0xFFFF else 1))

    chunks: list[str] = []
    start = 0  # code-point index into text

    while units[-1] - units[start] > max_len:
        # The longest prefix from `start` that fits: text[start:end] is at most
        # max_len units. `units` is strictly increasing, so the bisect finds the
        # exact boundary even with surrogate pairs in it.
        end = bisect.bisect_right(units, units[start] + max_len) - 1
        if end == start:
            # One code point alone needs more units than max_len allows (an
            # astral character with max_len == 1). Emit it on its own rather than
            # looping forever, exactly as a code-point count always could.
            end = start + 1
        candidate = text[start:end]
        split_idx = -1

        for sep in ("\n\n", "\n", ". ", " "):
            idx = candidate.rfind(sep)
            if idx != -1:
                split_idx = idx + len(sep)
                break

        if split_idx <= 0:
            split_idx = end - start

        chunk = text[start : start + split_idx].rstrip()
        if chunk:
            chunks.append(chunk)
        start += split_idx
        while start < len(text) and text[start].isspace():
            start += 1

    if start < len(text):
        chunks.append(text[start:])

    return chunks


class BaseChannel(ABC):
    """Abstract base class for all communication channels (CLI, WhatsApp, Telegram, Webhook)."""

    def __init__(self) -> None:
        self.runner: Runner | None = None

    def attach(self, runner: Runner) -> "BaseChannel":
        """Attach a multi-agent Runner instance to this channel."""
        self.runner = runner
        return self

    def _ensure_runner_attached(self) -> Runner:
        """Validate that a runner is properly bound."""
        if not self.runner:
            raise ChannelError(
                f"No Runner is attached to channel '{self.__class__.__name__}'. "
                "Call 'channel.attach(runner)' before running or dispatching messages."
            )
        return self.runner

    async def dispatch_async(
        self,
        session_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Forward an incoming message to the attached runner asynchronously."""
        runner = self._ensure_runner_attached()
        return await runner.run_async(session_id, message, metadata=metadata)

    def dispatch(
        self,
        session_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Forward an incoming message to the attached runner synchronously."""
        runner = self._ensure_runner_attached()
        return runner.run(session_id, message, metadata=metadata)

    @abstractmethod
    def run(self, **kwargs: Any) -> None:
        """Start listening for incoming messages or launch the channel event loop."""
        pass
