"""Telegram channel adapter using python-telegram-bot."""

import logging
from typing import Any

from chatflow_agent.channels.base import BaseChannel
from chatflow_agent.exceptions import DependencyError
from chatflow_agent.types import AgentResponse

logger = logging.getLogger("chatflow_agent.telegram")


class TelegramChannel(BaseChannel):
    """Telegram bot integration supporting async polling and multi-agent dispatching."""

    def __init__(
        self,
        token: str,
        start_message: str = "Hello! I am your AI multi-agent assistant. How can I help you today?",
        allowed_updates: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.token = token
        self.start_message = start_message
        self.allowed_updates = allowed_updates

        # Verify optional dependencies
        try:
            from telegram import Update
            from telegram.constants import ChatAction
            from telegram.ext import Application, filters
        except ImportError as err:
            raise DependencyError(
                feature="TelegramChannel",
                extra_package="telegram",
            ) from err

        self._update_cls = Update
        self._action_cls = ChatAction
        self._filters = filters

        # Initialize python-telegram-bot application
        self.app = Application.builder().token(self.token).build()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register Telegram command and message handlers."""
        from telegram.ext import CommandHandler, MessageHandler, filters

        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("reset", self._handle_reset))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

    async def _handle_start(self, update: Any, context: Any) -> None:
        """Respond to /start command."""
        if update.message:
            await update.message.reply_text(self.start_message)

    async def _handle_reset(self, update: Any, context: Any) -> None:
        """Clear memory for current chat via /reset command."""
        if not update.effective_chat or not update.message:
            return

        chat_id = str(update.effective_chat.id)
        if self.runner:
            session = self.runner.get_session(chat_id)
            if session:
                session.clear_history()
        await update.message.reply_text("Conversation memory has been reset.")

    async def _handle_message(self, update: Any, context: Any) -> None:
        """Process standard incoming text messages from Telegram."""
        if not update.effective_chat or not update.message or not update.message.text:
            return

        chat_id = str(update.effective_chat.id)
        user_text = update.message.text
        user_name = update.effective_user.username if update.effective_user else None

        # Display typing indicator while processing
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=self._action_cls.TYPING,
            )
        except Exception:
            pass

        try:
            response: AgentResponse = await self.dispatch_async(
                session_id=chat_id,
                message=user_text,
                metadata={
                    "channel": "telegram",
                    "chat_id": chat_id,
                    "username": user_name,
                },
            )
            await update.message.reply_text(response.content)
        except Exception as err:
            logger.error(f"Error processing Telegram message: {err}")
            await update.message.reply_text(
                "An error occurred while processing your request."
            )

    def run(self, **kwargs: Any) -> None:
        """Start polling for Telegram updates."""
        self._ensure_runner_attached()
        logger.info("Starting Telegram bot polling...")
        self.app.run_polling(allowed_updates=self.allowed_updates, **kwargs)
