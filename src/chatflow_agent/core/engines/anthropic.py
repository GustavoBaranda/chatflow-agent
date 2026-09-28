"""Anthropic Claude engine for chatflow-agent."""

import asyncio
import json
import logging
import os
import random
from typing import TYPE_CHECKING, Any

import httpx

from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.core.engines.openai_compatible import _lowercase_schema_types
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall

logger = logging.getLogger("chatflow_agent.engine.anthropic")

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent


class AnthropicEngine(BaseEngine):
    """Coordinates API requests with Anthropic Claude using Messages API over httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.timeout = timeout
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 10.0
        self._external_client = http_client

    def _format_history(self, history: list[Message]) -> list[dict[str, Any]]:
        """Convert chatflow history into Anthropic Messages format."""
        messages: list[dict[str, Any]] = []

        for msg in history:
            if msg.role == Role.USER and msg.content:
                messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.MODEL:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.args,
                        })
                if content_blocks:
                    messages.append({"role": "assistant", "content": content_blocks})

            elif msg.role == Role.TOOL and msg.tool_results:
                content_blocks = []
                for res in msg.tool_results:
                    content_str = (
                        json.dumps(res.content)
                        if isinstance(res.content, (dict, list))
                        else str(res.content)
                    )
                    content_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": res.tool_call_id,
                        "content": content_str,
                        "is_error": res.is_error,
                    })
                if content_blocks:
                    messages.append({"role": "user", "content": content_blocks})

        return messages

    async def generate_turn_async(
        self,
        agent: "Agent",
        history: list[Message],
    ) -> EngineTurnResult:
        """Call Anthropic Messages API asynchronously."""
        if not self.api_key:
            raise ProviderError(
                "Anthropic API key is not configured. Set ANTHROPIC_API_KEY environment variable."
            )

        messages = self._format_history(history)
        all_tools = agent.get_all_tools()

        payload: dict[str, Any] = {
            "model": agent.model,
            "max_tokens": 4096,
            "messages": messages,
        }

        instructions = agent.get_instructions()
        if instructions:
            payload["system"] = instructions

        if all_tools:
            anthropic_tools = []
            for t in all_tools:
                anthropic_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": _lowercase_schema_types(t.parameters_schema),
                })
            payload["tools"] = anthropic_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

        url = f"{self.base_url}/messages"

        retries = 0
        while True:
            client = self._external_client or httpx.AsyncClient(timeout=self.timeout)
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504, 529):
                    if retries < self.max_retries:
                        retry_after_str = resp.headers.get("Retry-After")
                        parsed_delay = None
                        if retry_after_str:
                            try:
                                parsed_delay = float(retry_after_str)
                            except (ValueError, TypeError):
                                parsed_delay = None

                        base = parsed_delay if parsed_delay is not None else self.base_delay * (2 ** retries)
                        delay = min(self.max_delay, base + random.uniform(0.0, 0.25 * base))
                        logger.warning(
                            f"Anthropic returned HTTP {resp.status_code}. Retrying in {delay:.2f}s (attempt {retries + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(delay)
                        retries += 1
                        continue
                    raise ProviderError(
                        f"Anthropic API error ({resp.status_code}): {resp.text}"
                    )

                if resp.status_code >= 400:
                    raise ProviderError(f"Anthropic API error ({resp.status_code}): {resp.text}")

                data = resp.json()
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as net_err:
                if retries < self.max_retries:
                    base = self.base_delay * (2 ** retries)
                    delay = min(self.max_delay, base + random.uniform(0.0, 0.25 * base))
                    logger.warning(
                        f"Anthropic connect error: {net_err}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    retries += 1
                    continue
                raise ProviderError(f"Anthropic connection error: {net_err}") from net_err
            except httpx.RequestError as err:
                raise ProviderError(f"Anthropic connection error: {err}") from err
            finally:
                if not self._external_client:
                    await client.aclose()

        text_chunks: list[str] = []
        extracted_calls: list[ToolCall] = []

        for block in data.get("content", []):
            b_type = block.get("type")
            if b_type == "text":
                text_chunks.append(block.get("text", ""))
            elif b_type == "tool_use":
                extracted_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        args=block.get("input", {}),
                    )
                )

        return EngineTurnResult(
            text="".join(text_chunks) if text_chunks else None,
            tool_calls=extracted_calls,
            raw_response=data,
        )
