"""Abstract BaseChannel definition for chatflow-agent."""

from abc import ABC, abstractmethod
from typing import Any

from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import ChatFlowError
from chatflow_agent.types import AgentResponse


class ChannelError(ChatFlowError):
    """Raised when an operation on a channel fails or runner is detached."""


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
