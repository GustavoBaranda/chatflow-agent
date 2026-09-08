"""Gemini LLM engine interface and payload formatting for chatflow-agent."""

import os
from typing import Any

from google import genai
from google.genai import types

from chatflow_agent.core.agent import Agent
from chatflow_agent.exceptions import ProviderError
from chatflow_agent.types import Message, Role, ToolCall


class EngineTurnResult:
    """Standardized output from a single LLM evaluation turn."""

    def __init__(
        self,
        text: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        raw_response: Any | None = None,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_response = raw_response

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class GeminiEngine:
    """Coordinates API requests with Google Gemini using google-genai SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # Instantiate client if api_key is present or relying on default environment
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
                    # Function response payload must be a mapping/dict
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
        agent: Agent,
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

        # Build tools configuration if agent has any tools
        gemini_tools: list[types.Tool] | None = None
        if all_tools:
            declarations = [t.to_gemini_declaration() for t in all_tools]
            gemini_tools = [types.Tool(function_declarations=declarations)]

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

        # Parse response parts
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
                        extracted_calls.append(
                            ToolCall(
                                id=call_id,
                                name=part.function_call.name,
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
