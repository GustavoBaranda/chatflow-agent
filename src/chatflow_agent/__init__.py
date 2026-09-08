"""chatflow-agent: Lightweight, async multi-agent framework with native handoffs."""

from chatflow_agent.channels.base import BaseChannel, ChannelError
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import GeminiEngine
from chatflow_agent.core.memory import SessionContext, SessionStore
from chatflow_agent.core.runner import Runner
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
    "BaseChannel",
    "ChannelError",
    "CLIChannel",
    "GeminiEngine",
    "Handoff",
    "Message",
    "Role",
    "Runner",
    "SessionContext",
    "SessionStore",
    "Tool",
    "tool",
    "ToolCall",
    "ToolResult",
    "WhatsAppChannel",
    "ChatFlowError",
    "DependencyError",
    "ToolExecutionError",
    "AgentHandoffError",
    "ProviderError",
]
