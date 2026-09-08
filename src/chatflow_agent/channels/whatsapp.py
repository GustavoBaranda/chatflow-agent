"""WhatsApp channel adapter using FastAPI and Meta Cloud API webhooks."""

import logging
from typing import Any, Dict, Optional

from chatflow_agent.channels.base import BaseChannel
from chatflow_agent.exceptions import DependencyError
from chatflow_agent.types import AgentResponse

logger = logging.getLogger("chatflow_agent.whatsapp")


class WhatsAppChannel(BaseChannel):
    """WhatsApp integration supporting Meta Cloud API and generic webhooks."""

    def __init__(
        self,
        verify_token: str,
        access_token: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        api_version: str = "v21.0",
    ) -> None:
        super().__init__()
        self.verify_token = verify_token
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_version = api_version

        # Verify optional dependencies
        try:
            from fastapi import FastAPI, HTTPException, Query, Request, Response, status
            from fastapi.responses import JSONResponse, PlainTextResponse
            import httpx
            import uvicorn
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
        from fastapi import HTTPException, Query, Request, Response, status
        from fastapi.responses import JSONResponse, PlainTextResponse

        @self.app.get("/health")
        async def health_check() -> Dict[str, str]:
            return {"status": "healthy", "channel": "whatsapp"}

        @self.app.get("/webhook")
        async def verify_webhook(
            request: Request,
            hub_mode: Optional[str] = Query(None, alias="hub.mode"),
            hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
            hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
        ) -> Response:
            """Meta webhook verification challenge endpoint."""
            if hub_mode == "subscribe" and hub_verify_token == self.verify_token:
                logger.info("WhatsApp webhook verified successfully.")
                return PlainTextResponse(content=hub_challenge or "")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Verification token mismatch",
            )

        @self.app.post("/webhook")
        async def handle_incoming_message(request: Request) -> Response:
            """Process incoming WhatsApp message events."""
            payload = await request.json()
            message_data = self._extract_message_and_sender(payload)

            if not message_data:
                # Return 200 to acknowledge status notifications (delivered, read, etc.)
                return JSONResponse(content={"status": "ignored_or_status_update"})

            sender_phone, user_text = message_data

            # Dispatch to attached runner
            response: AgentResponse = await self.dispatch_async(
                session_id=sender_phone,
                message=user_text,
                metadata={"channel": "whatsapp", "phone": sender_phone},
            )

            # If Meta credentials are provided, send outbound reply to WhatsApp API
            if self.access_token and self.phone_number_id:
                await self._send_outbound_whatsapp(
                    to_phone=sender_phone, text=response.content
                )

            return JSONResponse(
                content={
                    "status": "success",
                    "sender": sender_phone,
                    "reply": response.content,
                    "active_agent": response.active_agent_name,
                }
            )

    def _extract_message_and_sender(
        self, payload: Dict[str, Any]
    ) -> Optional[tuple[str, str]]:
        """Extract sender phone number and text message from payload.

        Supports both standard Meta Cloud API and direct/Evolution API webhooks.
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
                    # Text message
                    text = msg.get("text", {}).get("body")
                    if sender and text:
                        return str(sender), str(text)

        # 2. Generic / Evolution API webhook format
        sender = (
            payload.get("phone")
            or payload.get("from")
            or payload.get("sender")
        )
        text = (
            payload.get("message")
            or payload.get("text")
            or payload.get("body")
        )
        if sender and text:
            return str(sender), str(text)

        return None

    async def _send_outbound_whatsapp(self, to_phone: str, text: str) -> None:
        """Send message back to user via Meta WhatsApp Cloud API."""
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

        try:
            async with self._httpx.AsyncClient() as client:
                res = await client.post(url, json=body, headers=headers, timeout=10.0)
                if res.status_code >= 400:
                    logger.error(
                        f"Failed to send outbound WhatsApp message: {res.status_code} - {res.text}"
                    )
        except Exception as err:
            logger.error(f"Error calling WhatsApp Cloud API: {err}")

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
        """Launch the FastAPI webhook server using uvicorn."""
        self._ensure_runner_attached()
        self._uvicorn.run(self.app, host=host, port=port, **kwargs)
