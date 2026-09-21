
"""WhatsApp channel adapter using FastAPI and Meta Cloud API webhooks."""

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

from chatflow_agent.channels.base import BaseChannel
from chatflow_agent.core.memory import DEFAULT_DEDUP_TTL_HOURS
from chatflow_agent.exceptions import DependencyError
from chatflow_agent.types import AgentResponse

logger = logging.getLogger("chatflow_agent.whatsapp")


def _mask_phone(phone: str) -> str:
    """Mask phone number for GDPR/privacy compliance in logs (e.g. 5491***678)."""
    if len(phone) <= 6:
        return "***"
    return f"{phone[:4]}***{phone[-3:]}"


def split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split long outbound text into sequential chunks within max_len.

    Splits progressively by paragraph (\n\n), line (\n), sentence (. ), or word ( )
    to preserve formatting and readability across messaging channels.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_len:
        candidate = remaining[:max_len]
        split_idx = -1

        for sep in ("\n\n", "\n", ". ", " "):
            idx = candidate.rfind(sep)
            if idx != -1:
                split_idx = idx + len(sep)
                break

        if split_idx <= 0:
            split_idx = max_len

        chunk = remaining[:split_idx].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


class WhatsAppChannel(BaseChannel):
    """WhatsApp integration supporting Meta Cloud API and generic webhooks with resilience."""

    def __init__(
        self,
        verify_token: str,
        access_token: str | None = None,
        phone_number_id: str | None = None,
        app_secret: str | None = None,
        verify_signature: bool = True,
        api_version: str = "v21.0",
        fallback_message: str = (
            "Disculpa, estamos experimentando una demora temporal con nuestro servicio. "
            "Por favor intenta nuevamente en unos momentos."
        ),
        unsupported_media_message: str = (
            "Por el momento solo puedo procesar mensajes de texto."
        ),
        llm_timeout_seconds: float = 30.0,
        max_outbound_retries: int = 3,
        outbound_base_delay: float = 1.0,
        outbound_max_delay: float = 30.0,
        on_24h_window_expired: Any | None = None,
        dedup_ttl_hours: float = DEFAULT_DEDUP_TTL_HOURS,
    ) -> None:
        super().__init__()
        # SEC-01: Fail-closed — require app_secret when signature verification is active.
        # BREAKING CHANGE from v0.1.x: app_secret was previously optional and silently ignored.
        if verify_signature and not app_secret:
            raise ValueError(
                "WhatsAppChannel requires 'app_secret' when verify_signature=True (the default). "
                "Provide your Meta app secret via app_secret=os.environ['WHATSAPP_APP_SECRET'], "
                "or explicitly pass verify_signature=False for local development/testing only."
            )
        if not verify_signature:
            logger.warning(
                "SECURITY WARNING: Webhook signature verification is DISABLED "
                "(verify_signature=False). Do NOT use this in production."
            )
        self.verify_token = verify_token
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.app_secret = app_secret
        self.verify_signature = verify_signature
        self.api_version = api_version
        self.fallback_message = fallback_message
        self.unsupported_media_message = unsupported_media_message
        self.llm_timeout_seconds = llm_timeout_seconds
        self.max_outbound_retries = max_outbound_retries
        self.outbound_base_delay = outbound_base_delay
        self.outbound_max_delay = outbound_max_delay
        self.on_24h_window_expired = on_24h_window_expired
        self.dedup_ttl_hours = dedup_ttl_hours

        # Verify optional dependencies
        try:
            import httpx
            import uvicorn
            from fastapi import FastAPI
        except ImportError as err:
            raise DependencyError(
                feature="WhatsAppChannel",
                extra_package="whatsapp",
            ) from err

        self._httpx = httpx
        self._uvicorn = uvicorn
        self.app = FastAPI(title="ChatFlow WhatsApp Webhook Server")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register webhook endpoints on the FastAPI application."""
        from fastapi import BackgroundTasks, HTTPException, Query, Request, Response, status
        from fastapi.responses import JSONResponse, PlainTextResponse

        @self.app.get("/health")
        async def health_check() -> dict[str, str]:
            return {"status": "healthy", "channel": "whatsapp"}

        @self.app.get("/webhook")
        async def verify_webhook(
            request: Request,
            hub_mode: str | None = Query(None, alias="hub.mode"),
            hub_challenge: str | None = Query(None, alias="hub.challenge"),
            hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
        ) -> Response:
            """Meta webhook verification challenge endpoint with constant-time token comparison."""
            # SEC-02: Use constant-time comparison to prevent timing attacks on verify_token
            token_matches = False
            if hub_verify_token and self.verify_token:
                token_matches = hmac.compare_digest(
                    hub_verify_token.encode(), self.verify_token.encode()
                )
            if hub_mode == "subscribe" and token_matches:
                logger.info("WhatsApp webhook verified successfully.")
                return PlainTextResponse(content=hub_challenge or "")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Verification token mismatch",
            )

        @self.app.post("/webhook")
        async def handle_incoming_message(
            request: Request, background_tasks: BackgroundTasks
        ) -> Response:
            """Validate HMAC, deduplicate by wamid, and dispatch processing to background tasks."""
            raw_body = await request.body()

            # SEC-01: Validate X-Hub-Signature-256 when verify_signature is active
            if self.verify_signature and self.app_secret:
                sig_header = request.headers.get("X-Hub-Signature-256", "")
                if not sig_header.startswith("sha256="):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Missing or malformed X-Hub-Signature-256 header",
                    )
                expected = "sha256=" + hmac.new(
                    self.app_secret.encode(), raw_body, hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(sig_header.encode(), expected.encode()):
                    logger.warning("Rejected webhook: X-Hub-Signature-256 mismatch.")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid X-Hub-Signature-256 signature",
                    )

            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception as err:
                logger.error(f"Failed to decode JSON webhook payload: {err}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON payload",
                ) from err

            extracted_messages = self._extract_messages(payload)
            if not extracted_messages:
                return JSONResponse(content={"status": "ignored_or_status_update"})

            runner = self.runner
            queued_count = 0
            first_sender = extracted_messages[0][1]

            for wamid, sender_phone, user_text in extracted_messages:
                # BUS-01/ARCH-01: Record wamid atomically BEFORE enqueuing background task.
                # Semantics: at-most-once. If process crashes during BG execution the turn
                # is lost and Meta's retry (up to 7 days) will be discarded. OK with 1 worker.
                if runner and hasattr(runner, "session_store") and runner.session_store:
                    is_new = runner.session_store.record_processed_message(wamid)
                    if not is_new:
                        logger.info(f"Skipping duplicate wamid: {wamid}")
                        continue

                queued_count += 1
                background_tasks.add_task(
                    self._process_message_background,
                    wamid=wamid,
                    sender_phone=sender_phone,
                    user_text=user_text,
                )

            return JSONResponse(
                content={
                    "status": "success",
                    "sender": first_sender,
                    "messages_queued": queued_count,
                }
            )



    def _extract_messages(
        self, payload: dict[str, Any]
    ) -> list[tuple[str, str, str | None]]:
        """Extract all (wamid, sender_phone, text_or_None) from a webhook payload.

        Handles Meta Cloud API (entry[].changes[].value.messages[]) and generic formats.
        Returns an empty list for status-update notifications.
        """
        results: list[tuple[str, str, str | None]] = []

        # 1. Standard Meta Cloud API format
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for msg in change.get("value", {}).get("messages", []):
                    wamid = msg.get("id") or f"gen_{hash(str(msg))}"
                    sender = msg.get("from")
                    if not sender:
                        continue
                    if msg.get("type", "text") == "text":
                        body = msg.get("text", {}).get("body")
                        results.append((str(wamid), str(sender), str(body) if body else None))
                    else:
                        results.append((str(wamid), str(sender), None))

        # 2. Generic / Evolution API format
        if not results:
            sender = (
                payload.get("phone") or payload.get("from") or payload.get("sender")
            )
            if sender:
                wamid = (
                    payload.get("id") or payload.get("wamid")
                    or f"gen_{hash(str(payload))}"
                )
                body = payload.get("message") or payload.get("text") or payload.get("body")
                results.append((str(wamid), str(sender), str(body) if body else None))

        return results

    async def _process_message_background(
        self, wamid: str, sender_phone: str, user_text: str | None
    ) -> None:
        """Process one WhatsApp message asynchronously with full anti-500 error shield."""
        try:
            if user_text is None:
                if self.access_token and self.phone_number_id:
                    await self._send_outbound_whatsapp(
                        to_phone=sender_phone, text=self.unsupported_media_message
                    )
                return

            try:
                response: AgentResponse = await asyncio.wait_for(
                    self.dispatch_async(
                        session_id=sender_phone,
                        message=user_text,
                        metadata={"channel": "whatsapp", "phone": sender_phone, "wamid": wamid},
                    ),
                    timeout=self.llm_timeout_seconds,
                )
                reply_text = response.content
            except asyncio.TimeoutError:
                logger.error(
                    f"LLM dispatch timed out after {self.llm_timeout_seconds}s for wamid={wamid} from {sender_phone}"
                )
                reply_text = self.fallback_message
            except Exception as err:
                logger.exception(
                    f"Error dispatching wamid={wamid} from {sender_phone}: {err}"
                )
                reply_text = self.fallback_message

            if self.access_token and self.phone_number_id:
                await self._send_outbound_whatsapp(to_phone=sender_phone, text=reply_text)

        except Exception as unhandled:
            logger.exception(f"Unhandled error in BG task wamid={wamid}: {unhandled}")
            if self.access_token and self.phone_number_id:
                try:
                    await self._send_outbound_whatsapp(
                        to_phone=sender_phone, text=self.fallback_message
                    )
                except Exception as fb_err:
                    logger.exception(
                        f"Failed to deliver fallback to {sender_phone}: {fb_err}"
                    )

    def _extract_message_and_sender(
        self, payload: dict[str, Any]
    ) -> tuple[str, str | None] | None:
        """Extract sender phone number and message from payload.

        Returns:
            (phone, text) for text messages.
            (phone, None) for non-text media messages.
            None for status updates or empty payloads.
        """
        # 1. Standard Meta Cloud API webhook format
        entry = payload.get("entry")
        if isinstance(entry, list) and entry:
            changes = entry[0].get("changes")
            if isinstance(changes, list) and changes:
                value = changes[0].get("value", {})
                messages = value.get("messages")
                if isinstance(messages, list) and messages:
                    msg = messages[0]
                    sender = msg.get("from")
                    if not sender:
                        return None

                    msg_type = msg.get("type", "text")
                    if msg_type == "text":
                        text = msg.get("text", {}).get("body")
                        if text:
                            return str(sender), str(text)
                    # Non-text media message received
                    return str(sender), None

        # 2. Generic / Evolution API webhook format
        sender = (
            payload.get("phone")
            or payload.get("from")
            or payload.get("sender")
        )
        if sender:
            text = (
                payload.get("message")
                or payload.get("text")
                or payload.get("body")
            )
            if text:
                return str(sender), str(text)
            return str(sender), None

        return None

    async def _send_outbound_whatsapp(self, to_phone: str, text: str) -> None:
        """Send message back to user via Meta WhatsApp Cloud API, chunking long replies."""
        chunks = split_message(text, max_len=4096)
        for chunk in chunks:
            await self._send_single_outbound_whatsapp(to_phone=to_phone, text=chunk)

    async def _send_single_outbound_whatsapp(self, to_phone: str, text: str) -> None:
        """Send a single text message chunk via Meta WhatsApp Cloud API with retries and 24h detection."""
        import random

        masked = _mask_phone(to_phone)
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        body = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {"body": text},
        }

        retries = 0
        while True:
            try:
                async with self._httpx.AsyncClient() as client:
                    res = await client.post(url, json=body, headers=headers, timeout=10.0)

                    if res.status_code < 400:
                        return

                    # Handle 429 Too Many Requests: respect Retry-After header + jitter
                    if res.status_code == 429:
                        if retries < self.max_outbound_retries:
                            retry_after_str = res.headers.get("Retry-After")
                            parsed_retry_after: float | None = None
                            if retry_after_str:
                                try:
                                    parsed_retry_after = float(retry_after_str)
                                except (ValueError, TypeError):
                                    parsed_retry_after = None

                            if parsed_retry_after is not None:
                                base_delay = parsed_retry_after
                            else:
                                base_delay = self.outbound_base_delay * (2 ** retries)

                            jitter = random.uniform(0.0, 0.25 * base_delay)
                            delay = min(self.outbound_max_delay, base_delay + jitter)

                            logger.warning(
                                f"Rate limited by Meta (HTTP 429) for {masked}. "
                                f"Retrying in {delay:.2f}s (attempt {retries + 1}/{self.max_outbound_retries})..."
                            )
                            await asyncio.sleep(delay)
                            retries += 1
                            continue
                        logger.error(f"Max retries exceeded for {masked} after HTTP 429.")
                        return

                    # Inspect Meta error payload for 24h window expiration (error code 131047)
                    try:
                        err_data = res.json().get("error", {})
                        err_code = err_data.get("code")
                        if err_code == 131047:
                            logger.warning(
                                f"24h messaging window expired for {masked}. Use template messages."
                            )
                            if self.on_24h_window_expired:
                                try:
                                    if asyncio.iscoroutinefunction(self.on_24h_window_expired):
                                        await self.on_24h_window_expired(to_phone)
                                    else:
                                        self.on_24h_window_expired(to_phone)
                                except Exception as hook_err:
                                    logger.error(
                                        f"Error executing on_24h_window_expired hook for {masked}: {hook_err}"
                                    )
                            return
                    except Exception as parse_err:
                        logger.warning(
                            f"Unable to parse Meta error payload for {masked}: {parse_err} (raw: {res.text[:150]})"
                        )

                    # Non-retriable 4xx/5xx: log error with masked phone and exit without retrying
                    logger.error(
                        f"Failed to send outbound WhatsApp message to {masked}: {res.status_code} - {res.text}"
                    )
                    return

            # ONLY retry on connect errors (connection failed before request could be accepted by Meta)
            except (self._httpx.ConnectError, self._httpx.ConnectTimeout) as net_err:
                if retries < self.max_outbound_retries:
                    base_delay = self.outbound_base_delay * (2 ** retries)
                    jitter = random.uniform(0.0, 0.25 * base_delay)
                    delay = min(self.outbound_max_delay, base_delay + jitter)
                    logger.warning(
                        f"Connect error calling WhatsApp API for {masked}: {net_err}. "
                        f"Retrying in {delay:.2f}s (attempt {retries + 1}/{self.max_outbound_retries})..."
                    )
                    await asyncio.sleep(delay)
                    retries += 1
                    continue
                logger.error(f"Max retries exceeded for {masked} after connect error: {net_err}")
                return

            # Read/Write timeouts must NOT retry because Meta may have already received and delivered the message
            except (self._httpx.ReadTimeout, self._httpx.WriteTimeout) as timeout_err:
                logger.error(
                    f"Read/write timeout sending outbound message to {masked}: {timeout_err}. "
                    "Not retrying to prevent duplicate message dispatch."
                )
                return

            except Exception as err:
                logger.error(f"Error calling WhatsApp Cloud API for {masked}: {err}")
                return

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
        """Launch the FastAPI webhook server using uvicorn."""
        self._ensure_runner_attached()
        self._uvicorn.run(self.app, host=host, port=port, **kwargs)
