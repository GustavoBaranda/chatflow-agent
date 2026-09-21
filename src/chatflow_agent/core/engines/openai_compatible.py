
"""Universal OpenAI-compatible engine for chatflow-agent.

Supports OpenAI (GPT-4o), xAI Grok, Ollama (Google Gemma, Llama), DeepSeek,
vLLM, LM Studio, Groq, and any endpoint adhering to the OpenAI /chat/completions standard.
"""

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall

logger = logging.getLogger("chatflow_agent.engine.openai")

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent

# Preset configurations for common OpenAI-compatible providers
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "default_model": "grok-2",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "default_model": "grok-2",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "default_model": "gemma2:9b",
    },
    "gemma": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "default_model": "gemma2:9b",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
}


def _lowercase_schema_types(schema: Any) -> Any:
    """Recursively convert uppercase schema types (e.g. 'STRING' -> 'string')."""
    if isinstance(schema, dict):
        new_dict: dict[str, Any] = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                new_dict[k] = v.lower()
            else:
                new_dict[k] = _lowercase_schema_types(v)
        return new_dict
    elif isinstance(schema, list):
        return [_lowercase_schema_types(item) for item in schema]
    return schema


class OpenAIEngine(BaseEngine):
    """Universal OpenAI-compatible API engine powered by async httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        preset = PROVIDER_PRESETS.get(provider.lower(), {})
        self.provider = provider.lower()
        self.base_url = (
            base_url
            or os.environ.get(f"{provider.upper()}_BASE_URL")
            or preset.get("base_url", "https://api.openai.com/v1")
        ).rstrip("/")

        env_key_name = preset.get("env_key", "OPENAI_API_KEY")
        self.api_key = api_key or os.environ.get(env_key_name) or (
            "ollama" if "localhost" in self.base_url or "127.0.0.1" in self.base_url else None
        )
        self.timeout = timeout
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 10.0
        self._external_client = http_client

    def _format_history_to_messages(
        self,
        instructions: str | None,
        history: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert chatflow Message objects into OpenAI messages array."""
        messages: list[dict[str, Any]] = []

        if instructions:
            messages.append({"role": "system", "content": instructions})

        for msg in history:
            if msg.role == Role.USER and msg.content:
                messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.MODEL:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    formatted_calls = []
                    for call in msg.tool_calls:
                        formatted_calls.append({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.args),
                            },
                        })
                    assistant_msg["tool_calls"] = formatted_calls
                messages.append(assistant_msg)

            elif msg.role == Role.TOOL and msg.tool_results:
                for res in msg.tool_results:
                    content_str = (
                        json.dumps(res.content)
                        if isinstance(res.content, (dict, list))
                        else str(res.content)
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": res.tool_call_id,
                        "name": res.name,
                        "content": content_str,
                    })

        return messages

    async def generate_turn_async(
        self,
        agent: "Agent",
        history: list[Message],
    ) -> EngineTurnResult:
        """Execute chat completion against an OpenAI-compatible endpoint."""
        messages = self._format_history_to_messages(agent.get_instructions(), history)
        all_tools = agent.get_all_tools()

        payload: dict[str, Any] = {
            "model": agent.model,
            "messages": messages,
        }

        if all_tools:
            openai_tools = []
            for t in all_tools:
                parameters = _lowercase_schema_types(t.parameters_schema)
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": parameters,
                    },
                })
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        retries = 0
        while True:
            client = self._external_client or httpx.AsyncClient(timeout=self.timeout)
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if retries < self.max_retries:
                        retry_after_str = resp.headers.get("Retry-After")
                        parsed_delay = None
                        if retry_after_str:
                            try:
                                parsed_delay = float(retry_after_str)
                            except (ValueError, TypeError):
                                parsed_delay = None

                        import random
                        base = parsed_delay if parsed_delay is not None else self.base_delay * (2 ** retries)
                        delay = min(self.max_delay, base + random.uniform(0.0, 0.25 * base))
                        logger.warning(
                            f"{self.provider.upper()} returned HTTP {resp.status_code}. Retrying in {delay:.2f}s (attempt {retries + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(delay)
                        retries += 1
                        continue
                    raise ProviderError(
                        f"{self.provider.upper()} API request failed ({resp.status_code}): {resp.text}"
                    )

                if resp.status_code >= 400:
                    raise ProviderError(
                        f"{self.provider.upper()} API request failed ({resp.status_code}): {resp.text}"
                    )
                data = resp.json()
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as net_err:
                if retries < self.max_retries:
                    import random
                    base = self.base_delay * (2 ** retries)
                    delay = min(self.max_delay, base + random.uniform(0.0, 0.25 * base))
                    logger.warning(
                        f"{self.provider.upper()} connect error: {net_err}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    retries += 1
                    continue
                raise ProviderError(f"{self.provider.upper()} connection error: {net_err}") from net_err
            except httpx.RequestError as err:
                raise ProviderError(f"{self.provider.upper()} connection error: {err}") from err
            finally:
                if not self._external_client:
                    await client.aclose()

        choices = data.get("choices", [])
        if not choices:
            return EngineTurnResult(text="", raw_response=data)

        first_choice = choices[0]
        choice_msg = first_choice.get("message", {})
        text_content = choice_msg.get("content")
        raw_tool_calls = choice_msg.get("tool_calls") or []

        extracted_calls: list[ToolCall] = []
        for idx, tc in enumerate(raw_tool_calls):
            call_id = tc.get("id") or f"call_{idx}"
            func_data = tc.get("function", {})
            func_name = func_data.get("name", "")
            raw_args = func_data.get("arguments", "{}")

            parsed_args: dict[str, Any] = {}
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    parsed_args = {"raw_input": raw_args}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args

            extracted_calls.append(ToolCall(id=call_id, name=func_name, args=parsed_args))

        return EngineTurnResult(
            text=text_content,
            tool_calls=extracted_calls,
            raw_response=data,
        )
