"""Agent abstraction and peer handoff orchestration for chatflow-agent."""

import hashlib
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from chatflow_agent.core.tools import Tool
from chatflow_agent.types import Handoff

_MAX_TOOL_NAME_LEN = 60
_HANDOFF_PREFIX = "transfer_to_"


def _sanitize_tool_name(name: str) -> str:
    """Convert an agent name into a valid tool identifier.

    Accented Latin characters are transliterated via NFKD normalization.
    Names that yield no ASCII alphanumeric content (non-Latin scripts, emojis)
    fall back to a deterministic hash so the result is never empty.
    The full tool name is capped at 60 characters and matches
    ``^[a-zA-Z_][a-zA-Z0-9_]*$``.
    """
    stripped = name.strip()
    normalized = unicodedata.normalize("NFKD", stripped)
    without_marks = "".join(c for c in normalized if not unicodedata.combining(c))
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", without_marks.lower())
    clean = re.sub(r"_+", "_", clean).strip("_")

    if not clean:
        digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:12]
        clean = f"u{digest}"

    max_clean_len = _MAX_TOOL_NAME_LEN - len(_HANDOFF_PREFIX)
    if len(clean) > max_clean_len:
        clean = clean[:max_clean_len].rstrip("_")
        if not clean:
            digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:12]
            clean = f"u{digest}"[:max_clean_len]

    return f"{_HANDOFF_PREFIX}{clean}"


def _allocate_tool_name(base_name: str, used: set[str]) -> str:
    """Return ``base_name`` or a disambiguated variant absent from ``used``."""
    if base_name not in used:
        return base_name

    counter = 2
    while True:
        suffix = f"_{counter}"
        max_base_len = _MAX_TOOL_NAME_LEN - len(suffix)
        candidate = base_name[:max_base_len].rstrip("_") + suffix
        if candidate not in used:
            return candidate
        counter += 1


class Agent:
    """Represents an autonomous specialized agent with its own tools and peer handoffs."""

    def __init__(
        self,
        name: str,
        instructions: str | Callable[[], str],
        model: str = "gemini-2.5-flash",
        tools: list[Tool | Callable[..., Any]] | None = None,
        handoffs: list["Agent"] | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        engine: Any | None = None,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Agent name must be a non-empty string.")
        self.name = name
        self.instructions = instructions
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.engine = engine
        self.tools: list[Tool] = []
        self.handoffs: list[Agent] = []

        if tools:
            for t in tools:
                self.add_tool(t)

        if handoffs:
            for h in handoffs:
                self.add_handoff(h)

    def get_instructions(self) -> str:
        """Resolve and return the system instructions."""
        if callable(self.instructions):
            return self.instructions()
        return self.instructions

    def add_tool(self, tool_or_callable: Tool | Callable[..., Any]) -> Tool:
        """Register a tool or callable to this agent."""
        if isinstance(tool_or_callable, Tool):
            t = tool_or_callable
        else:
            t = Tool(tool_or_callable)

        # Avoid duplicate tools by name
        self.tools = [existing for existing in self.tools if existing.name != t.name]
        self.tools.append(t)
        return t

    def tool(
        self,
        name_or_func: str | Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Decorator to register a tool directly onto this agent instance.

        Usage:
            @agent.tool
            def query_db(id: int) -> dict: ...

            @agent.tool(name="custom_query", description="Fetch db row")
            def query_db(id: int) -> dict: ...
        """
        if callable(name_or_func):
            t = Tool(name_or_func)
            self.add_tool(t)
            return t

        def decorator(fn: Callable[..., Any]) -> Tool:
            tool_name = name or (name_or_func if isinstance(name_or_func, str) else None)
            t = Tool(fn, name=tool_name, description=description)
            self.add_tool(t)
            return t

        return decorator

    def add_handoff(self, target_agent: "Agent") -> None:
        """Register a peer agent to which this agent can hand off dialogue."""
        if target_agent.name == self.name:
            raise ValueError(f"Agent '{self.name}' cannot register a handoff to itself.")

        # Avoid duplicates
        self.handoffs = [h for h in self.handoffs if h.name != target_agent.name]
        self.handoffs.append(target_agent)

    def _create_handoff_tool(self, target_agent: "Agent", tool_name: str) -> Tool:
        """Generate a synthetic routing tool that transfers control to target_agent."""
        desc = (
            f"Hand off the dialogue to '{target_agent.name}'. "
            f"Target agent specialty: {target_agent.get_instructions()[:120]}..."
        )

        def handoff_executor(reason: str | None = None) -> Handoff:
            """Transfer the conversation to another specialized agent."""
            return Handoff(
                target_agent_name=target_agent.name,
                reason=reason,
            )

        return Tool(func=handoff_executor, name=tool_name, description=desc)

    def get_all_tools(self) -> list[Tool]:
        """Combine user tools with handoff tools, disambiguating name collisions."""
        combined: list[Tool] = list(self.tools)
        tool_names = {t.name for t in combined}
        for peer in self.handoffs:
            base_name = _sanitize_tool_name(peer.name)
            tool_name = _allocate_tool_name(base_name, tool_names)
            combined.append(self._create_handoff_tool(peer, tool_name))
            tool_names.add(tool_name)
        return combined

    def __repr__(self) -> str:
        return (
            f"<Agent name={self.name!r} model={self.model!r} "
            f"tools={len(self.tools)} handoffs={len(self.handoffs)}>"
        )
