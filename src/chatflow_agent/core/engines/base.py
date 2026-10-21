"""Base engine interfaces and standardized turn result for chatflow-agent."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from chatflow_agent.types import Message, Role, ToolCall

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent


def system_prompt(instructions: str | None, history: list[Message]) -> str | None:
    """The agent's instructions followed by the Role.SYSTEM messages in history, in order.

    For providers that take one system prompt beside the messages (Anthropic, Gemini). The
    agent's instructions stay first; system messages added to the session later follow them.
    """
    parts = [instructions] if instructions else []
    parts += [msg.content for msg in history if msg.role == Role.SYSTEM and msg.content]
    return "\n\n".join(parts) or None


class EngineTurnResult:
    """Standardized output from a single LLM evaluation turn."""

    def __init__(
        self,
        text: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        raw_response: Any | None = None,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_response = raw_response

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __repr__(self) -> str:
        names = [c.name for c in self.tool_calls]
        return f"<EngineTurnResult text={self.text!r} tool_calls={names}>"


class BaseEngine(ABC):
    """Abstract base interface for all LLM provider engines."""

    @abstractmethod
    async def generate_turn_async(
        self,
        agent: "Agent",
        history: list[Message],
    ) -> EngineTurnResult:
        """Evaluate conversation history with the agent's tools and instructions."""
        raise NotImplementedError
