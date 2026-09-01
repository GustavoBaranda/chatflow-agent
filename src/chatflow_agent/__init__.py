"""chatflow-agent: Lightweight, async multi-agent framework with native handoffs."""

from chatflow_agent.core.tools import Tool, tool
from chatflow_agent.exceptions import (
    AgentHandoffError,
    ChatFlowError,
    DependencyError,
    ProviderError,
    ToolExecutionError,
)
from chatflow_agent.types import (
    AgentResponse,
    Handoff,
    Message,
    Role,
    ToolCall,
    ToolResult,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AgentResponse",
    "Handoff",
    "Message",
    "Role",
    "Tool",
    "tool",
    "ToolCall",
    "ToolResult",
    "ChatFlowError",
    "DependencyError",
    "ToolExecutionError",
    "AgentHandoffError",
    "ProviderError",
]
