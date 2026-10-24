"""WhatsApp channel adapter using FastAPI and Meta Cloud API webhooks."""

import asyncio
import hashlib
import hmac
import json
import logging
import math
import uuid
from typing import Any

from chatflow_agent.channels.base import BaseChannel, split_message
from chatflow_agent.core.memory import DEFAULT_DEDUP_TTL_HOURS
from chatflow_agent.core.runner import (
    _dispatch_limits,
    _DispatchLimits,
    _SessionWaitTimeout,
    _TurnTimeout,
    _within,
)
from chatflow_agent.exceptions import DependencyError
from chatflow_agent.types import AgentResponse

logger = logging.getLogger("chatflow_agent.whatsapp")

DEFAULT_MAX_BODY_SIZE = 1 * 1024 * 1024  # 1 MB maximum allowed webhook payload size

# Extra time the background dispatch backstop allows beyond queue + turn limits. The Runner
# enforces both limits; the backstop only bounds a dispatch_async() that never reaches it.
_DISPATCH_BACKSTOP_GRACE_SECONDS = 1.0


class _DispatchBackstopTimeout(asyncio.TimeoutError):
    """dispatch_async() outlived queue + turn limits: it did not apply the Runner's limits."""


def _mask_phone(phone: str) -> str:
    """Mask phone number for GDPR/privacy compliance in logs (e.g. 5491***678)."""
    if len(phone) <= 6:
        return "***"
    return f"{phone[:4]}***{phone[-3:]}"


__all__ = ["WhatsAppChannel", "split_message"]


class WhatsAppChannel(BaseChannel):
    """WhatsApp integration supporting Meta Cloud API and generic webhooks with resilience.

    Each message gets two separate limits:

    * ``queue_timeout_seconds`` (default 120) bounds the wait behind earlier turns of the
      same session (turns of one session run one at a time). If the session is still
      busy when it expires, the message is answered with ``fallback_message`` and its
      turn never runs: it is not added to the conversation history.
    * ``llm_timeout_seconds`` (default 120) bounds the turn itself (model calls, tools,
      handoffs) and starts once the turn holds the session. When it expires the turn is
      cancelled and rolled back and ``fallback_message`` is sent. It is a turn deadline,
      not a guarantee that every provider retry fits; pass an explicit value to match
      your provider and workload.

    A message may therefore take up to ``queue_timeout_seconds + llm_timeout_seconds``
    before its reply is sent. Webhook HTTP acknowledgement remains immediate and is
    separate from both limits.

    ``queue_timeout_seconds`` must be a positive finite number. ``llm_timeout_seconds``
    accepts the values it accepted before: ``None`` or ``inf`` set no turn limit, and
    zero or negative values make every turn time out. ``nan`` sets no limit (it used to
    time turns out at once or, on Python 3.13, stop the event loop). A warning is logged
    for zero, negative and ``nan`` values.
    """

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
        unsupported_media_message: str = ("Por el momento solo puedo procesar mensajes de texto."),
        llm_timeout_seconds: float = 120.0,
        max_outbound_retries: int = 3,
        outbound_base_delay: float = 1.0,
        outbound_max_delay: float = 30.0,
        on_24h_window_expired: Any | None = None,
        dedup_ttl_hours: float = DEFAULT_DEDUP_TTL_HOURS,
        max_body_size: int = DEFAULT_MAX_BODY_SIZE,
        enable_docs: bool = False,
        queue_timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__()
        if not (math.isfinite(queue_timeout_seconds) and queue_timeout_seconds > 0):
            raise ValueError(
                f"queue_timeout_seconds must be a positive finite number, got {queue_timeout_seconds!r}"
            )
        # llm_timeout_seconds is not validated, to keep accepting what it accepted before.
        if llm_timeout_seconds is not None and (
            math.isnan(llm_timeout_seconds) or llm_timeout_seconds <= 0
        ):
            logger.warning(
                "llm_timeout_seconds=%r: zero or negative values make every turn time out, "
                "and nan is treated as no limit. Use a positive number, or None for no limit.",
                llm_timeout_seconds,
            )
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
        self.queue_timeout_seconds = queue_timeout_seconds
        self.max_outbound_retries = max_outbound_retries
        self.outbound_base_delay = outbound_base_delay
        self.outbound_max_delay = outbound_max_delay
        self.on_24h_window_expired = on_24h_window_expired
        self.dedup_ttl_hours = dedup_ttl_hours
        self.max_body_size = max_body_size
        self.enable_docs = enable_docs

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
        # SEC-05: Disable OpenAPI schema, Swagger UI (/docs) and ReDoc (/redoc) by default in production
        docs_url = "/docs" if enable_docs else None
        redoc_url = "/redoc" if enable_docs else None
        openapi_url = "/openapi.json" if enable_docs else None
        self.app = FastAPI(
            title="ChatFlow WhatsApp Webhook Server",
            docs_url=docs_url,
            redoc_url=redoc_url,
            openapi_url=openapi_url,
        )
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
            """Validate Content-Length, HMAC, deduplicate by wamid, and dispatch processing to background tasks."""
            # SEC-01: Validate Content-Length before buffering body to prevent OOM DoS
            content_length_header = request.headers.get("Content-Length")
            if not content_length_header:
                raise HTTPException(
                    status_code=status.HTTP_411_LENGTH_REQUIRED,
                    detail="Missing Content-Length header",
                )

            try:
                content_length = int(content_length_header)
            except (ValueError, TypeError) as err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Content-Length header",
                ) from err

            status_413 = getattr(
                status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            )

            if content_length < 0 or content_length > self.max_body_size:
                raise HTTPException(
                    status_code=status_413,
                    detail=f"Payload size exceeds maximum limit of {self.max_body_size} bytes",
                )

            raw_body = await request.body()
            if len(raw_body) > self.max_body_size:
                raise HTTPException(
                    status_code=status_413,
                    detail=f"Payload body exceeds maximum limit of {self.max_body_size} bytes",
                )

            # SEC-01: Validate X-Hub-Signature-256 when verify_signature is active
            if self.verify_signature and self.app_secret:
                sig_header = request.headers.get("X-Hub-Signature-256", "")
                if not sig_header.startswith("sha256="):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Missing or malformed X-Hub-Signature-256 header",
                    )
                expected = (
                    "sha256="
                    + hmac.new(self.app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
                )
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

            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON payload must be an object",
                )

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

    def _extract_messages(self, payload: dict[str, Any]) -> list[tuple[str, str, str | None]]:
        """Extract all (wamid, sender_phone, text_or_None) from a webhook payload.

        Handles Meta Cloud API (entry[].changes[].value.messages[]) and generic formats.
        Returns an empty list for status-update notifications.

        ``reaction`` and ``system`` events are ignored (no outbound reply). Other
        non-text types yield ``text_or_None=None`` so the background worker can
        send the unsupported-media notice. Malformed nested shapes are skipped
        with a warning so a signed webhook never raises into a 500.
        """
        results: list[tuple[str, str, str | None]] = []

        def _warn_malformed(kind: str) -> None:
            logger.warning("Ignoring malformed WhatsApp webhook %s (no content logged).", kind)

        # 1. Standard Meta Cloud API format
        entries = payload.get("entry", [])
        if not isinstance(entries, list):
            _warn_malformed("entry")
            entries = []

        for entry in entries:
            if not isinstance(entry, dict):
                _warn_malformed("entry item")
                continue
            changes = entry.get("changes", [])
            if not isinstance(changes, list):
                _warn_malformed("changes")
                continue
            for change in changes:
                if not isinstance(change, dict):
                    _warn_malformed("changes item")
                    continue
                value = change.get("value", {})
                if value is None:
                    _warn_malformed("value")
                    continue
                if not isinstance(value, dict):
                    _warn_malformed("value")
                    continue
                messages = value.get("messages", [])
                if messages is None:
                    continue
                if not isinstance(messages, list):
                    _warn_malformed("messages")
                    continue
                for msg in messages:
                    if not isinstance(msg, dict):
                        _warn_malformed("messages item")
                        continue
                    wamid = msg.get("id") or f"gen_{hash(str(msg))}"
                    sender = msg.get("from")
                    if not sender:
                        continue
                    msg_type = msg.get("type", "text")
                    if msg_type in ("reaction", "system"):
                        # Acknowledge silently: no unsupported-media reply.
                        continue
                    if msg_type == "text":
                        text_node = msg.get("text", {})
                        if isinstance(text_node, str):
                            _warn_malformed("text")
                            continue
                        if not isinstance(text_node, dict):
                            _warn_malformed("text")
                            continue
                        body = text_node.get("body")
                        results.append((str(wamid), str(sender), str(body) if body else None))
                    else:
                        results.append((str(wamid), str(sender), None))

        # 2. Generic / Evolution API format.
        # Meta Cloud API payloads always carry ``entry``. Never fall through to the
        # generic extractor for them: ignored reaction/system events would otherwise
        # be revived by top-level ``from`` / ``text`` fields.
        if not results and "entry" not in payload:
            sender = payload.get("phone") or payload.get("from") or payload.get("sender")
            if sender:
                wamid = payload.get("id") or payload.get("wamid") or f"gen_{uuid.uuid4().hex}"
                body = payload.get("message") or payload.get("text") or payload.get("body")
                results.append((str(wamid), str(sender), str(body) if body else None))

        return results

    async def _process_message_background(
        self, wamid: str, sender_phone: str, user_text: str | None
    ) -> None:
        """Process one WhatsApp message asynchronously with full anti-500 error shield."""
        masked_phone = _mask_phone(sender_phone)
        try:
            if user_text is None:
                if self.access_token and self.phone_number_id:
                    await self._send_outbound_whatsapp(
                        to_phone=sender_phone, text=self.unsupported_media_message
                    )
                return

            # The Runner applies both limits: the wait for the session lock and, once it is
            # held, the turn itself. They reach it through a context variable so that
            # dispatch_async() keeps its signature.
            loop = asyncio.get_running_loop()
            turn_limit = self.llm_timeout_seconds
            # No turn limit (None, inf or nan) means no backstop either.
            backstop: float | None = None
            if turn_limit is not None and not math.isnan(turn_limit) and turn_limit != math.inf:
                backstop = (
                    self.queue_timeout_seconds
                    + max(turn_limit, 0.0)
                    + _DISPATCH_BACKSTOP_GRACE_SECONDS
                )
            limits_token = _dispatch_limits.set(
                _DispatchLimits(
                    wait_deadline=loop.time() + self.queue_timeout_seconds,
                    turn_timeout=turn_limit,
                )
            )
            try:
                response: AgentResponse = await _within(
                    backstop,
                    self.dispatch_async(
                        session_id=sender_phone,
                        message=user_text,
                        metadata={"channel": "whatsapp", "phone": sender_phone, "wamid": wamid},
                    ),
                    _DispatchBackstopTimeout,
                )
                reply_text = response.content
            except _SessionWaitTimeout:
                logger.error(
                    f"Session busy: wamid={wamid} from {masked_phone} waited "
                    f"{self.queue_timeout_seconds}s (queue_timeout_seconds) behind earlier turns "
                    "of the same session; its turn was not run."
                )
                reply_text = self.fallback_message
            except _TurnTimeout:
                logger.error(
                    f"Turn timed out after {turn_limit}s (llm_timeout_seconds) "
                    f"for wamid={wamid} from {masked_phone}; the turn was rolled back."
                )
                reply_text = self.fallback_message
            except _DispatchBackstopTimeout:
                logger.error(
                    f"Dispatch backstop expired after {backstop}s for wamid={wamid} from "
                    f"{masked_phone}: dispatch_async() did not finish within "
                    "queue_timeout_seconds + llm_timeout_seconds + "
                    f"{_DISPATCH_BACKSTOP_GRACE_SECONDS}s, so the Runner's limits were not "
                    "applied (a dispatch_async() override that does not call Runner.run_async()?)."
                )
                reply_text = self.fallback_message
            except asyncio.TimeoutError as err:
                logger.exception(
                    f"TimeoutError raised while dispatching wamid={wamid} from {masked_phone} "
                    f"(not a channel limit): {err}"
                )
                reply_text = self.fallback_message
            except Exception as err:
                logger.exception(f"Error dispatching wamid={wamid} from {masked_phone}: {err}")
                reply_text = self.fallback_message
            finally:
                _dispatch_limits.reset(limits_token)

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
                    logger.exception(f"Failed to deliver fallback to {masked_phone}: {fb_err}")

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
                                base_delay = self.outbound_base_delay * (2**retries)

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
                    base_delay = self.outbound_base_delay * (2**retries)
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
