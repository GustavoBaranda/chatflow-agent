"""Communication channels for chatflow-agent: CLI, WhatsApp, Telegram, and Webhook."""

from chatflow_agent.channels.base import BaseChannel, ChannelError, split_message
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.channels.telegram import TelegramChannel
from chatflow_agent.channels.whatsapp import WhatsAppChannel

__all__ = [
    "BaseChannel",
    "CLIChannel",
    "ChannelError",
    "TelegramChannel",
    "WhatsAppChannel",
    "split_message",
]
