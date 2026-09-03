"""chatflow-agent: Lightweight, async multi-agent framework with native handoffs."""

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.memory import SessionContext, SessionStore
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
    "Agent",
    "AgentResponse",
    "Handoff",
    "Message",
    "Role",
    "SessionContext",
    "SessionStore",
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
