"""chatflow-agent: Lightweight, async multi-agent framework with native handoffs."""

from chatflow_agent.channels.base import BaseChannel, ChannelError
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.channels.whatsapp import WhatsAppChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import AnthropicEngine, BaseEngine, GeminiEngine, OpenAIEngine
from chatflow_agent.core.memory import (
    BaseSessionStore,
    SessionContext,
    SessionStore,
    SQLiteSessionContext,
    SQLiteSessionStore,
)
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

# Alias for semantic consistency
BaseSessionContext = SessionContext

__version__ = "0.1.1"
__all__ = [
    "__version__",
    "Agent",
    "AgentResponse",
    "BaseChannel",
    "ChannelError",
    "CLIChannel",
    "AnthropicEngine",
    "BaseEngine",
    "GeminiEngine",
    "OpenAIEngine",
    "Handoff",
    "Message",
    "Role",
    "Runner",
    "BaseSessionContext",
    "BaseSessionStore",
    "SessionContext",
    "SessionStore",
    "SQLiteSessionContext",
    "SQLiteSessionStore",
    "TelegramChannel",
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
