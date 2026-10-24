"""Anthropic Claude engine for chatflow-agent."""

import asyncio
import json
import logging
import random
import re
from typing import TYPE_CHECKING, Any

import httpx

from chatflow_agent.core.engines.base import (
    BaseEngine,
    EngineTurnResult,
    replacement_tool_call_id,
    system_prompt,
)
from chatflow_agent.core.engines.openai_compatible import (
    _body_excerpt,
    _lowercase_schema_types,
    _resolve_api_key,
)
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall

logger = logging.getLogger("chatflow_agent.engine.anthropic")

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

# The format Anthropic's API reference gives for tool_use.id and tool_result.tool_use_id.
_TOOL_USE_ID = re.compile(r"[a-zA-Z0-9_-]+")


def _tool_use_id(call_id: str) -> str:
    """The history's tool-call ID, unless Anthropic would reject it (e.g. one from another provider)."""
    return call_id if _TOOL_USE_ID.fullmatch(call_id) else replacement_tool_call_id(call_id)


class AnthropicEngine(BaseEngine):
    """Coordinates API requests with Anthropic Claude using Messages API over httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = ANTHROPIC_BASE_URL,
        anthropic_version: str = "2023-06-01",
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # ANTHROPIC_API_KEY goes only to the official endpoint; proxies, gateways
        # and local servers need an explicit api_key.
        self.api_key = _resolve_api_key(
            api_key,
            self.base_url,
            official_base_url=ANTHROPIC_BASE_URL,
            env_key="ANTHROPIC_API_KEY",
        )
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
                            "id": _tool_use_id(call.id),
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
                        "tool_use_id": _tool_use_id(res.tool_call_id),
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
        # api_key="" is an explicit choice to send no credential (e.g. an
        # unauthenticated gateway); only a missing key is an error.
        if self.api_key is None:
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

        system = system_prompt(agent.get_instructions(), history)
        if system:
            payload["system"] = system

        if all_tools:
            anthropic_tools = []
            for t in all_tools:
                anthropic_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": _lowercase_schema_types(t.to_openai_parameters()),
                })
            payload["tools"] = anthropic_tools

        headers = {
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        url = f"{self.base_url}/messages"

        retries = 0
        while True:
            client = self._external_client or httpx.AsyncClient(timeout=self.timeout)
            try:
                # Never follow redirects, even on an injected client configured to:
                # httpx keeps x-api-key on cross-origin redirects.
                resp = await client.post(
                    url, json=payload, headers=headers, follow_redirects=False
                )
                if 300 <= resp.status_code < 400:
                    raise ProviderError(
                        f"Anthropic API request was redirected (HTTP {resp.status_code}). "
                        "Redirects are not followed; set base_url to the final endpoint."
                    )
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
                        f"Anthropic API error ({resp.status_code}): {_body_excerpt(resp.text)}"
                    )

                if resp.status_code >= 400:
                    raise ProviderError(
                        f"Anthropic API error ({resp.status_code}): {_body_excerpt(resp.text)}"
                    )

                try:
                    data = resp.json()
                except ValueError as err:
                    raise ProviderError(
                        f"Anthropic returned a malformed response: {err}"
                    ) from err
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

        if not isinstance(data, dict):
            raise ProviderError(
                "Anthropic returned a malformed response: "
                f"expected a JSON object, got {type(data).__name__}"
            )
        content = data.get("content", [])
        if not isinstance(content, list):
            raise ProviderError(
                "Anthropic returned a malformed response: 'content' is not a list"
            )

        text_chunks: list[str] = []
        extracted_calls: list[ToolCall] = []

        for block in content:
            if not isinstance(block, dict):
                raise ProviderError(
                    "Anthropic returned a malformed response: "
                    "a content block is not an object"
                )
            b_type = block.get("type")
            if b_type == "text":
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise ProviderError(
                        "Anthropic returned a malformed response: "
                        "'text' is not a string"
                    )
                text_chunks.append(text)
            elif b_type == "tool_use":
                tool_id = block.get("id", "")
                tool_name = block.get("name", "")
                if not isinstance(tool_id, str) or not isinstance(tool_name, str):
                    raise ProviderError(
                        "Anthropic returned a malformed response: "
                        "'tool_use.id' and 'tool_use.name' must be strings"
                    )
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    raise ProviderError(
                        "Anthropic returned a malformed response: "
                        "'tool_use.input' is not an object"
                    )
                extracted_calls.append(
                    ToolCall(id=tool_id, name=tool_name, args=tool_input)
                )

        return EngineTurnResult(
            text="".join(text_chunks) if text_chunks else None,
            tool_calls=extracted_calls,
            raw_response=data,
        )
