"""Base engine interfaces and standardized turn result for chatflow-agent."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from chatflow_agent.types import Message, ToolCall

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent


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
