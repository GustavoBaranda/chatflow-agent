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


class ToolArgumentValidationError(ToolExecutionError):
    """Raised when tool arguments provided by a model fail schema validation.

    Enforces a sanitized error contract: raw Pydantic ``ValidationError`` instances
    and sensitive inputs are never retained in ``original_error``, string representations,
    or exception tracebacks. ``self.original_error`` always holds a sanitized
    ``ValueError(message)``.
    """

    def __init__(
        self, tool_name: str, message: str, original_error: Exception | None = None
    ) -> None:
        clean_error = ValueError(message)
        super().__init__(tool_name=tool_name, original_error=clean_error)
        self.message = message
        self.__cause__ = None
        self.__context__ = None
        self.__suppress_context__ = True


class ToolSchemaError(ChatFlowError):
    """Raised when a tool's parameter annotation cannot be expressed as a valid schema."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AgentHandoffError(ChatFlowError):
    """Raised when an agent transfer/handoff fails or references an invalid target."""


class ProviderError(ChatFlowError):
    """Raised when communication with the LLM provider fails."""


class ReentrantTurnError(ChatFlowError, RuntimeError):
    """Raised when code running inside a turn starts another turn of the same session.

    Subclasses ``RuntimeError`` for compatibility with code that caught the previous error.
    """


class SessionDeletedError(ChatFlowError):
    """Raised when a turn tries to store a message in a session that was deleted meanwhile."""
