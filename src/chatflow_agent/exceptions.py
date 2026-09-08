"""Exception hierarchy for chatflow-agent."""



class ChatFlowError(Exception):
    """Base exception for all chatflow-agent errors."""


class DependencyError(ChatFlowError):
    """Raised when an optional channel or feature requires an uninstalled package."""

    def __init__(self, feature: str, extra_package: str, install_command: str | None = None) -> None:
        cmd = install_command or f'pip install "chatflow-agent[{extra_package}]"'
        message = (
            f"The feature '{feature}' requires optional dependencies from '{extra_package}'. "
            f"Please install them using: {cmd}"
        )
        super().__init__(message)
        self.feature = feature
        self.extra_package = extra_package


class ToolExecutionError(ChatFlowError):
    """Raised when a registered tool fails during execution."""

    def __init__(self, tool_name: str, original_error: Exception) -> None:
        message = f"Error executing tool '{tool_name}': {original_error}"
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class AgentHandoffError(ChatFlowError):
    """Raised when an agent transfer/handoff fails or references an invalid target."""


class ProviderError(ChatFlowError):
    """Raised when communication with the LLM provider fails."""
