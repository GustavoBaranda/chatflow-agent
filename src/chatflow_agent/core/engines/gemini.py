"""Gemini LLM engine implementation using google-genai SDK."""

import os
from typing import TYPE_CHECKING, Any, cast

from google import genai
from google.genai import types

from chatflow_agent.core.engines.base import BaseEngine, EngineTurnResult
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall

if TYPE_CHECKING:
    from chatflow_agent.core.agent import Agent


class GeminiEngine(BaseEngine):
    """Coordinates API requests with Google Gemini using google-genai SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client: genai.Client | None = None
        try:
            self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        except Exception:
            # Allow lazy initialization / mocking during testing
            self.client = None

    def _format_history_to_contents(self, history: list[Message]) -> list[types.Content]:
        """Convert chatflow Message objects into Gemini Content payloads."""
        contents: list[types.Content] = []

        for msg in history:
            parts: list[types.Part] = []

            # 1. User text
            if msg.role == Role.USER and msg.content:
                parts.append(types.Part.from_text(text=msg.content))
                contents.append(types.Content(role="user", parts=parts))

            # 2. Model text or tool calls
            elif msg.role == Role.MODEL:
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        parts.append(
                            types.Part.from_function_call(
                                name=call.name,
                                args=call.args,
                            )
                        )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

            # 3. Tool results returned back to model
            elif msg.role == Role.TOOL and msg.tool_results:
                for res in msg.tool_results:
                    payload = res.content if isinstance(res.content, dict) else {"output": res.content}
                    parts.append(
                        types.Part.from_function_response(
                            name=res.name,
                            response=payload,
                        )
                    )
                if parts:
                    contents.append(types.Content(role="user", parts=parts))

        return contents

    async def generate_turn_async(
        self,
        agent: "Agent",
        history: list[Message],
    ) -> EngineTurnResult:
        """Call Gemini API asynchronously with active agent's tools and system instructions."""
        if not self.client:
            raise ProviderError(
                "Gemini API client is not configured. Set GEMINI_API_KEY environment variable "
                "or pass api_key to Runner."
            )

        contents = self._format_history_to_contents(history)
        all_tools = agent.get_all_tools()

        gemini_tools: list[Any] | None = None
        if all_tools:
            declarations = [t.to_gemini_declaration() for t in all_tools]
            gemini_tools = [types.Tool(function_declarations=cast(Any, declarations))]

        config = types.GenerateContentConfig(
            system_instruction=agent.get_instructions(),
            tools=gemini_tools,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=agent.model,
                contents=contents,
                config=config,
            )
        except Exception as err:
            raise ProviderError(f"Gemini API request failed: {err}") from err

        extracted_text: str | None = None
        extracted_calls: list[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_chunks: list[str] = []
                for idx, part in enumerate(candidate.content.parts):
                    if part.text:
                        text_chunks.append(part.text)
                    if part.function_call:
                        call_id = f"call_{part.function_call.name}_{idx}"
                        func_name = str(part.function_call.name or "")
                        extracted_calls.append(
                            ToolCall(
                                id=call_id,
                                name=func_name,
                                args=dict(part.function_call.args or {}),
                            )
                        )
                if text_chunks:
                    extracted_text = "".join(text_chunks)

        return EngineTurnResult(
            text=extracted_text,
            tool_calls=extracted_calls,
            raw_response=response,
        )
