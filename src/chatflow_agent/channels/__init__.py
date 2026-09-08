"""Communication channels for chatflow-agent: CLI, WhatsApp, Telegram, and Webhook."""

from chatflow_agent.channels.base import BaseChannel, ChannelError
from chatflow_agent.channels.cli import CLIChannel

__all__ = ["BaseChannel", "CLIChannel", "ChannelError"]
