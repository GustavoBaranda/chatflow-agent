"""Telegram channel adapter using python-telegram-bot."""

import asyncio
import contextlib
import functools
import logging
from typing import Any

from chatflow_agent.channels.base import BaseChannel, split_message
from chatflow_agent.core.memory import _run_blocking
from chatflow_agent.exceptions import DependencyError
from chatflow_agent.types import AgentResponse

logger = logging.getLogger("chatflow_agent.telegram")


class TelegramChannel(BaseChannel):
    """Telegram bot integration supporting async polling and multi-agent dispatching."""

    def __init__(
        self,
        token: str,
        start_message: str = "Hello! I am your AI multi-agent assistant. How can I help you today?",
        fallback_message: str = (
            "Disculpa, ocurrió un inconveniente al procesar tu solicitud. "
            "Por favor intenta nuevamente en unos momentos."
        ),
        allowed_updates: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.token = token
        self.start_message = start_message
        self.fallback_message = fallback_message
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

        # Restrict /reset in group and supergroup chats to creators and administrators
        if getattr(update.effective_chat, "type", None) in ("group", "supergroup"):
            is_admin = False
            if update.effective_user and hasattr(context, "bot") and context.bot:
                try:
                    member = await context.bot.get_chat_member(
                        chat_id=update.effective_chat.id,
                        user_id=update.effective_user.id,
                    )
                    if getattr(member, "status", None) in ("creator", "administrator"):
                        is_admin = True
                except Exception as err:
                    logger.error(
                        f"Failed to verify chat member permissions for /reset: {err}"
                    )
                    is_admin = False

            if not is_admin:
                await update.message.reply_text(
                    "Only group admins can reset the conversation."
                )
                return

        chat_id = str(update.effective_chat.id)
        if self.runner:
            # With a file-backed SQLiteSessionStore these run in worker threads, so a slow
            # or locked database does not stop the bot's event loop.
            runner = self.runner
            store = runner.session_store
            session = await _run_blocking(store, functools.partial(runner.get_session, chat_id))
            if session:
                await _run_blocking(store, session.clear_history)
        await update.message.reply_text("Conversation memory has been reset.")

    async def _handle_message(self, update: Any, context: Any) -> None:
        """Process standard incoming text messages from Telegram with error resilience."""
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
            chunks = split_message(response.content, max_len=4096)
            for chunk in chunks:
                await update.message.reply_text(chunk)
        except Exception as err:
            logger.error(f"Error processing Telegram message: {err}")
            await update.message.reply_text(self.fallback_message)

    def run(self, **kwargs: Any) -> None:
        """Start polling for Telegram updates.

        The attached runner's connection-pooling scope (``async with runner:``) is open for
        the application's lifetime: it is entered in ``post_init`` and closed in
        ``post_shutdown``, after PTB has finished processing pending updates. Callbacks
        already set on ``self.app.post_init``/``post_shutdown`` run inside the scope and are
        restored when polling ends. If a stop signal interrupts ``post_init``, it is cancelled
        at shutdown before the scope closes. If PTB's shutdown fails or is interrupted before
        ``post_shutdown`` completes, the scope is still closed before the event loop is
        (``close_loop`` keeps its meaning).
        """
        runner = self._ensure_runner_attached()
        user_post_init = self.app.post_init
        user_post_shutdown = self.app.post_shutdown
        close_loop = kwargs.pop("close_loop", True)
        scope = contextlib.AsyncExitStack()
        scope_open = False
        post_init_task: asyncio.Task[Any] | None = None
        post_shutdown_task: asyncio.Task[Any] | None = None

        async def close_scope() -> None:
            nonlocal scope_open
            scope_open = False
            await scope.aclose()  # a no-op if the scope was never opened or is already closed

        async def post_init(app: Any) -> None:
            nonlocal post_init_task, scope_open
            post_init_task = asyncio.current_task()
            # Open first, so a user post_init can already use pooled connections.
            await scope.enter_async_context(runner)
            scope_open = True
            if user_post_init is not None:
                await user_post_init(app)

        async def post_shutdown(app: Any) -> None:
            nonlocal post_shutdown_task
            post_shutdown_task = asyncio.current_task()
            try:
                # A stop signal during post_init makes PTB move on without waiting for it.
                # Cancel it here, as asyncio.run would, instead of leaving it to run on during
                # shutdown and be destroyed pending when PTB closes the loop.
                await _cancel_and_wait(post_init_task)
                if user_post_shutdown is not None:
                    await user_post_shutdown(app)
            finally:
                await close_scope()

        def shutdown_unfinished() -> bool:
            pending = [t for t in (post_init_task, post_shutdown_task) if t is not None and not t.done()]
            return scope_open or bool(pending)

        async def finish_shutdown() -> None:
            # PTB raised before post_shutdown, or a second stop signal interrupted it. The
            # user's post_shutdown is not called here: PTB skipped it.
            await _cancel_and_wait(post_init_task)
            await _cancel_and_wait(post_shutdown_task)  # its finally closes the scope
            await close_scope()

        # PTB closes its loop before run_polling returns, which would leave no loop to close
        # the scope on if post_shutdown never completed. So PTB is told not to close it, and
        # it is closed here instead, after that cleanup. The loop is chosen as PTB chooses it;
        # the marker shows whether PTB ran it (it doesn't when it fails before starting),
        # which is exactly when PTB would have closed it.
        loop, loop_created = _polling_event_loop()
        loop_ran: list[bool] = []
        marker = loop.call_soon(loop_ran.append, True)

        def finish_after_polling(polling_failed: bool) -> None:
            try:
                if loop_ran and shutdown_unfinished():
                    try:
                        loop.run_until_complete(finish_shutdown())
                    except Exception as err:
                        if not polling_failed:
                            raise
                        # Keep the error polling failed with; report this one.
                        logger.error(f"Failed to close the runner's connection scope: {err!r}")
            finally:
                if not loop_ran:
                    marker.cancel()
                    if loop_created and not loop.is_running() and not loop.is_closed():
                        # PTB failed before using the loop this call created for it: undo
                        # that, so no loop is left open and set as the thread's current one.
                        asyncio.set_event_loop(None)
                        loop.close()
                elif close_loop:
                    loop.close()

        self.app.post_init = post_init
        self.app.post_shutdown = post_shutdown
        logger.info("Starting Telegram bot polling...")
        try:
            try:
                self.app.run_polling(
                    allowed_updates=self.allowed_updates, close_loop=False, **kwargs
                )
            except BaseException:
                finish_after_polling(polling_failed=True)
                raise
            finish_after_polling(polling_failed=False)
        finally:
            self.app.post_init = user_post_init
            self.app.post_shutdown = user_post_shutdown


def _polling_event_loop() -> tuple[asyncio.AbstractEventLoop, bool]:
    """The event loop ``Application.run_polling`` will use, chosen as PTB 22.8 chooses it, and
    whether it was created here (there was no current loop, so this call owns it)."""
    try:
        return asyncio.get_event_loop(), False
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop, True


async def _cancel_and_wait(task: "asyncio.Task[Any] | None") -> None:
    """Cancel a still-pending lifecycle hook task and wait until it has finished."""
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.wait([task])
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"Telegram lifecycle hook failed after it was cancelled: {task.exception()!r}")
