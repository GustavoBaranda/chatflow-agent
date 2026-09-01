"""Agent abstraction and peer handoff orchestration for chatflow-agent."""

import re
from typing import Any, Callable, Dict, List, Optional, Union

from chatflow_agent.core.tools import Tool, tool
from chatflow_agent.types import Handoff


def _sanitize_tool_name(name: str) -> str:
    """Convert an agent name into a valid tool identifier (alphanumeric and underscores)."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"transfer_to_{clean}"


class Agent:
    """Represents an autonomous specialized agent with its own tools and peer handoffs."""

    def __init__(
        self,
        name: str,
        instructions: Union[str, Callable[[], str]],
        model: str = "gemini-2.5-flash",
        tools: Optional[List[Union[Tool, Callable[..., Any]]]] = None,
        handoffs: Optional[List["Agent"]] = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools: List[Tool] = []
        self.handoffs: List["Agent"] = []

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

    def add_tool(self, tool_or_callable: Union[Tool, Callable[..., Any]]) -> Tool:
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
        name_or_func: Optional[Union[str, Callable[..., Any]]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
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

    def _create_handoff_tool(self, target_agent: "Agent") -> Tool:
        """Generate a synthetic routing tool that transfers control to target_agent."""
        tool_name = _sanitize_tool_name(target_agent.name)
        desc = (
            f"Hand off the dialogue to '{target_agent.name}'. "
            f"Target agent specialty: {target_agent.get_instructions()[:120]}..."
        )

        def handoff_executor(reason: Optional[str] = None) -> Handoff:
            """Transfer the conversation to another specialized agent."""
            return Handoff(
                target_agent_name=target_agent.name,
                reason=reason,
            )

        return Tool(func=handoff_executor, name=tool_name, description=desc)

    def get_all_tools(self) -> List[Tool]:
        """Return all user tools combined with synthetic handoff tools."""
        combined: List[Tool] = list(self.tools)
        for peer in self.handoffs:
            combined.append(self._create_handoff_tool(peer))
        return combined

    def __repr__(self) -> str:
        return (
            f"<Agent name={self.name!r} model={self.model!r} "
            f"tools={len(self.tools)} handoffs={len(self.handoffs)}>"
        )
