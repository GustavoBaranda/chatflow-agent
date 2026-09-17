"""Multi-agent Swarm-style execution Runner for chatflow-agent."""

import asyncio
from typing import Any

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine, GeminiEngine, resolve_engine
from chatflow_agent.core.memory import SessionContext, SessionStore
from chatflow_agent.exceptions import AgentHandoffError
from chatflow_agent.types import (
    AgentResponse,
    Handoff,
    Message,
    Role,
    ToolResult,
)


class Runner:
    """Orchestrates multi-agent conversation loops, tool calling, and autonomous handoffs."""

    def __init__(
        self,
        starting_agent: Agent,
        session_store: SessionStore | None = None,
        engine: BaseEngine | None = None,
        api_key: str | None = None,
        max_handoffs_per_turn: int = 5,
        max_tool_iterations: int = 10,
    ) -> None:
        self.starting_agent = starting_agent
        self.session_store = session_store or SessionStore()
        self.engine = engine or GeminiEngine(api_key=api_key)
        self.max_handoffs_per_turn = max_handoffs_per_turn
        self.max_tool_iterations = max_tool_iterations

        # Registry of all known agents (starting agent + all transitive handoffs)
        self._agents_registry: dict[str, Agent] = {}
        self._register_agent_tree(self.starting_agent)

    def _register_agent_tree(self, root: Agent) -> None:
        """Traverse and index all agents in the handoff tree."""
        if root.name in self._agents_registry:
            return
        self._agents_registry[root.name] = root
        for peer in root.handoffs:
            self._register_agent_tree(peer)

    def register_agent(self, agent: Agent) -> None:
        """Manually register an agent to the runner index."""
        self._register_agent_tree(agent)

    def get_session(self, session_id: str) -> SessionContext | None:
        """Fetch session context by session_id."""
        return self.session_store.get(session_id)

    async def run_async(
        self,
        session_id: str,
        user_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Process a user message through the active agent and resolve tools/handoffs."""
        session = self.session_store.get_or_create(
            session_id=session_id,
            default_agent=self.starting_agent,
            metadata=metadata,
        )

        async with session.lock:
            # 1. Append inbound user message to history
            session.add_message(Message(role=Role.USER, content=user_message))

            tool_calls_executed: list[str] = []
            last_handoff: Handoff | None = None
            handoff_counter = 0

            # 2. Main agent execution loop
            for _ in range(self.max_tool_iterations):
                current_agent = session.active_agent
                active_engine = resolve_engine(current_agent, default_engine=self.engine)
                turn_result = await active_engine.generate_turn_async(
                    agent=current_agent,
                    history=session.history,
                )

                # If the model requested tool calls
                if turn_result.has_tool_calls:
                    # Record model's intent in history
                    session.add_message(
                        Message(
                            role=Role.MODEL,
                            content=turn_result.text,
                            tool_calls=turn_result.tool_calls,
                        )
                    )

                    tool_results: list[ToolResult] = []
                    available_tools = {t.name: t for t in current_agent.get_all_tools()}

                    for call in turn_result.tool_calls:
                        tool_calls_executed.append(call.name)
                        tool_instance = available_tools.get(call.name)

                        if not tool_instance:
                            tool_results.append(
                                ToolResult(
                                    tool_call_id=call.id,
                                    name=call.name,
                                    content=f"Error: Tool '{call.name}' not found on agent '{current_agent.name}'.",
                                    is_error=True,
                                )
                            )
                            continue

                        # Execute tool safely
                        try:
                            raw_output = await tool_instance.execute_async(**call.args)
                        except Exception as err:
                            tool_results.append(
                                ToolResult(
                                    tool_call_id=call.id,
                                    name=call.name,
                                    content=f"Execution error: {err}",
                                    is_error=True,
                                )
                            )
                            continue

                        # Check if tool execution resulted in a Handoff
                        if isinstance(raw_output, Handoff):
                            target_name = raw_output.target_agent_name
                            target_agent = self._agents_registry.get(target_name)

                            if not target_agent:
                                raise AgentHandoffError(
                                    f"Handoff target agent '{target_name}' is not registered in Runner."
                                )

                            # Switch active agent pointer
                            session.set_active_agent(target_agent)
                            last_handoff = raw_output
                            handoff_counter += 1

                            if handoff_counter > self.max_handoffs_per_turn:
                                tool_results.append(
                                    ToolResult(
                                        tool_call_id=call.id,
                                        name=call.name,
                                        content="Maximum agent transfers exceeded for a single turn.",
                                        is_error=True,
                                    )
                                )
                                break

                            tool_results.append(
                                ToolResult(
                                    tool_call_id=call.id,
                                    name=call.name,
                                    content={
                                        "status": "HANDOFF_SUCCESS",
                                        "transferred_to": target_agent.name,
                                        "reason": raw_output.reason or "Specialist routing",
                                    },
                                )
                            )
                        else:
                            tool_results.append(
                                ToolResult(
                                    tool_call_id=call.id,
                                    name=call.name,
                                    content=raw_output,
                                )
                            )

                    # Append tool results to conversation history
                    session.add_message(Message(role=Role.TOOL, tool_results=tool_results))
                    # Continue loop: active agent evaluates the tool results

                else:
                    # No more tool calls: final text response reached
                    final_text = turn_result.text or ""
                    session.add_message(Message(role=Role.MODEL, content=final_text))

                    return AgentResponse(
                        content=final_text,
                        active_agent_name=session.active_agent.name,
                        tool_calls_executed=tool_calls_executed,
                        handoff=last_handoff,
                    )

            # Fallback if max tool iterations exceeded
            fallback_text = (
                f"Execution iteration limit reached by agent '{session.active_agent.name}'."
            )
            session.add_message(Message(role=Role.MODEL, content=fallback_text))
            return AgentResponse(
                content=fallback_text,
                active_agent_name=session.active_agent.name,
                tool_calls_executed=tool_calls_executed,
                handoff=last_handoff,
            )

    def run(
        self,
        session_id: str,
        user_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Synchronous convenience wrapper for run_async."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(
                self.run_async(session_id, user_message, metadata=metadata)
            )
        except RuntimeError:
            return asyncio.run(
                self.run_async(session_id, user_message, metadata=metadata)
            )

    def __repr__(self) -> str:
        return (
            f"<Runner starting_agent={self.starting_agent.name!r} "
            f"registered_agents={list(self._agents_registry.keys())}>"
        )
