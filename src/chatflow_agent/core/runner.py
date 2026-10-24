"""Multi-agent Swarm-style execution Runner for chatflow-agent."""

import asyncio
import contextvars
import copy
import functools
import logging
import sys
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engines import BaseEngine, GeminiEngine, resolve_engine
from chatflow_agent.core.engines.gemini import _ClientPool, _lent_pool
from chatflow_agent.core.memory import BaseSessionStore, SessionContext, SessionStore
from chatflow_agent.exceptions import AgentHandoffError, ReentrantTurnError
from chatflow_agent.types import (
    AgentResponse,
    Handoff,
    Message,
    Role,
    ToolResult,
)

logger = logging.getLogger("chatflow_agent.runner")

class _TurnToken:
    """Marks a running turn for every piece of code it starts."""

    __slots__ = ("store", "session_id", "active")

    def __init__(self, store: object, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self.active = True


# The turns the current code runs inside. Context variables are copied into tasks created
# during a turn (create_task, gather, wait_for, TaskGroup) and into asyncio.to_thread workers,
# so tools and everything they start see the turn that started them.
_turn_lineage: contextvars.ContextVar[tuple[_TurnToken, ...]] = contextvars.ContextVar(
    "chatflow_turn_lineage", default=()
)


def _refuse_nested_turn(store: object, session_id: str) -> None:
    """Refuse a turn started from inside a still-running turn of the same session.

    Code started by a turn cannot know whether that turn is waiting for it, and if it is,
    waiting for the session lock would never end. The check covers the turn's own task and
    every task or thread it starts, in any asyncio pattern.
    """
    if any(
        token.active and token.store is store and token.session_id == session_id
        for token in _turn_lineage.get()
    ):
        raise ReentrantTurnError(
            f"Runner.run_async() was called for session '{session_id}' while code started by "
            "a turn of that session is already running a turn for it (for example, from inside "
            "a tool or a task it created). Waiting for the session lock could deadlock. To "
            "queue a follow-up turn, use Runner.spawn_turn() and do not await it inside the "
            "current turn."
        )


# Whether run() drives its loop with asyncio.Runner (3.11+), and so knows when the loop's
# teardown starts cancelling every task. Python 3.10 has no public way to know.
_TEARDOWN_OBSERVABLE = sys.version_info >= (3, 11)


class _LoopTeardown:
    """What ``Runner.run()`` knows about the teardown of its ``asyncio.Runner`` (3.11+)."""

    def __init__(self) -> None:
        # The tasks the teardown cancels, waits for and reports: those still running when
        # asyncio.Runner.run raised. None until then.
        self.tasks: set[asyncio.Task[Any]] | None = None

    def reports(self, task: "asyncio.Task[Any]") -> bool:
        """Whether the teardown reports how ``task`` ended."""
        return self.tasks is not None and task in self.tasks


def _report_shutdown_error(task: "asyncio.Task[Any]", teardown: _LoopTeardown | None) -> None:
    """Done callback of a task ``run()`` cancelled: report its error as ``asyncio.run``'s
    teardown reports the tasks it cancels, unless that teardown reports this task itself.

    KeyboardInterrupt and SystemExit are not reported: they propagate out of the loop."""
    if task.cancelled() or (teardown is not None and teardown.reports(task)):
        return
    error = task.exception()
    if isinstance(error, Exception):
        task.get_loop().call_exception_handler(
            {
                "message": "unhandled exception during asyncio.run() shutdown",
                "exception": error,
                "task": task,
            }
        )


async def _cancel_other_tasks(turn_interrupted: bool, teardown: _LoopTeardown | None) -> None:
    """Cancel and wait for every other task on the running loop, as ``asyncio.run`` would next.

    Mirrors ``asyncio.runners._cancel_all_tasks``, including how errors are reported, but runs
    inside the loop and leaves out the current task, so ``Runner.run()`` can do it before
    closing its scope.

    - Unless the loop's teardown has started, every other task is cancelled, including one
      that once absorbed a cancellation, exactly as ``asyncio.run`` would. That holds however
      the turn ended, even if its own task was cancelled or swallowed a cancellation.
    - Once the teardown has started (``teardown.tasks``: a KeyboardInterrupt or SystemExit
      escaped the loop while the turn's task was still running), it has cancelled the tasks
      then running and reports their errors, so they are only waited for here, never
      cancelled again. Tasks created since are neither cancelled nor waited for: the teardown
      leaves them running when it closes the loop, and waiting could block forever.
    - Each task cancelled here is reported once, when it ends, even if this wait is
      interrupted: here if it ended before the teardown started, otherwise by the teardown.
    - Python 3.10 (``teardown`` is None) can't tell whether the teardown has started, so after
      an interrupted turn (``turn_interrupted``: cancelled, KeyboardInterrupt or SystemExit)
      everything is left to it, as are the tasks still running if this wait is interrupted.
    - If this wait is cancelled (Ctrl-C during it), it stops without cancelling the tasks
      again; ``asyncio.run`` deals with those still running.
    """
    # asyncio.run makes one more loop pass after the main task ends, so tasks the turn has
    # just created take their first step before they are cancelled.
    await asyncio.sleep(0)
    current = asyncio.current_task()
    others = asyncio.all_tasks() - {current}

    if teardown is not None and teardown.tasks is not None:
        to_cancel: set[asyncio.Task[Any]] = set()
        others &= teardown.tasks
    elif teardown is None and turn_interrupted:
        return
    else:
        to_cancel = others
    if not others:
        return

    report = functools.partial(_report_shutdown_error, teardown=teardown)
    for task in to_cancel:
        task.cancel()
        task.add_done_callback(report)

    try:
        await asyncio.wait(others)
    finally:
        if teardown is None:
            for task in to_cancel:
                task.remove_done_callback(report)


@asynccontextmanager
async def _hold_session_lock(
    store: object, session_id: str, lock: asyncio.Lock
) -> AsyncIterator[None]:
    """Hold the session lock and mark the turn in the context of the code it runs."""
    async with lock:
        token = _TurnToken(store, session_id)
        reset = _turn_lineage.set(_turn_lineage.get() + (token,))
        try:
            yield
        finally:
            token.active = False
            _turn_lineage.reset(reset)


class Runner:
    """Orchestrates multi-agent conversation loops, tool calling, and autonomous handoffs."""

    def __init__(
        self,
        starting_agent: Agent,
        session_store: BaseSessionStore | SessionStore | None = None,
        engine: BaseEngine | None = None,
        api_key: str | None = None,
        max_handoffs_per_turn: int = 5,
        max_tool_iterations: int = 10,
    ) -> None:
        self.starting_agent = starting_agent
        self.session_store: BaseSessionStore | SessionStore = session_store or SessionStore()
        self.engine = engine or GeminiEngine(api_key=api_key)
        # Remember whether the caller supplied the engine explicitly: an explicit
        # engine must take precedence over the model-prefix heuristics.
        self._engine_provided = engine is not None
        self.max_handoffs_per_turn = max_handoffs_per_turn
        self.max_tool_iterations = max_tool_iterations

        # Turns started with spawn_turn(): asyncio keeps only weak references to tasks, so
        # they are held here until they finish.
        self._spawned_turns: set[asyncio.Task[AgentResponse]] = set()

        # Pooled Gemini HTTP clients of the open ``async with runner:`` scopes, one per event
        # loop, with the scope nesting depth. Only this dict is shared between threads; each
        # pool is used only on its own loop.
        self._client_pools: dict[asyncio.AbstractEventLoop, tuple[_ClientPool, int]] = {}
        self._client_pools_lock = threading.Lock()

        # Registry of all known agents (starting agent + all transitive handoffs)
        self._agents_registry: dict[str, Agent] = {}
        self._registry_lock = threading.Lock()
        self._register_agent_tree(self.starting_agent)

        if hasattr(self.session_store, "set_agent_resolver"):
            self.session_store.set_agent_resolver(lambda name: self._agents_registry.get(name))

    def _register_agent_tree(self, root: Agent) -> None:
        """Traverse and index all agents in the handoff tree."""
        with self._registry_lock:
            self._register_agent_tree_unlocked(root)

    def _register_agent_tree_unlocked(self, root: Agent) -> None:
        """Index ``root`` and its reachable peers. Caller must hold ``_registry_lock``."""
        if root.name in self._agents_registry:
            return
        self._agents_registry[root.name] = root
        for peer in root.handoffs:
            self._register_agent_tree_unlocked(peer)

    def _resolve_handoff_target(self, current_agent: Agent, target_name: str) -> Agent:
        """Resolve a handoff target from the registry or an authorized late-bound peer.

        Peers added via ``Agent.add_handoff()`` after ``Runner`` construction are
        accepted only when declared on ``current_agent.handoffs``. Unauthorized
        names fail closed with ``AgentHandoffError``.
        """
        # Authorization first: the target must be declared on the current
        # agent's own handoffs. Presence in the global registry alone is not
        # sufficient, so one branch cannot jump into another branch's agents.
        authorized = next(
            (peer for peer in current_agent.handoffs if peer.name == target_name),
            None,
        )
        if authorized is None:
            raise AgentHandoffError(
                f"Agent '{current_agent.name}' is not authorized to hand off to '{target_name}'."
            )

        with self._registry_lock:
            self._register_agent_tree_unlocked(authorized)
            return self._agents_registry[target_name]

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
        """Process a user message through the active agent and resolve tools/handoffs.

        Merge incoming metadata into the session before processing this turn.

        Turns of the same session run one at a time. Calling this method for a session from
        code started by a turn of that same session (a tool, or a task or thread it starts)
        while that turn is running raises ``ReentrantTurnError``; use ``spawn_turn()`` to
        queue a follow-up turn instead.
        """
        _refuse_nested_turn(self.session_store, session_id)
        session = self.session_store.get_or_create(
            session_id=session_id,
            default_agent=self.starting_agent,
            metadata=metadata,
        )

        async with _hold_session_lock(self.session_store, session_id, session.lock):
            # Reload the state committed by turns that held the lock before this one.
            # Outside the try: nothing has changed yet if it fails.
            refresh = getattr(session, "refresh", None)
            if callable(refresh):
                refresh()
            checkpoint_history = list(session.history)
            checkpoint_agent = session.active_agent
            checkpoint_metadata = copy.deepcopy(session.metadata)
            try:
                if metadata:
                    session.update_metadata(metadata)

                # 1. Append inbound user message to history
                session.add_message(Message(role=Role.USER, content=user_message))

                tool_calls_executed: list[str] = []
                last_handoff: Handoff | None = None
                handoff_counter = 0

                # 2. Main agent execution loop
                for _ in range(self.max_tool_iterations):
                    current_agent = session.active_agent
                    active_engine = resolve_engine(
                        current_agent,
                        default_engine=self.engine,
                        prefer_default_engine=self._engine_provided,
                    )
                    # Lend this loop's pooled client (if a scope is open) to the engine call.
                    lent = _lent_pool.set(self._client_pool_for_running_loop())
                    try:
                        turn_result = await active_engine.generate_turn_async(
                            agent=current_agent,
                            history=session.history,
                        )
                    finally:
                        _lent_pool.reset(lent)

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

                        for call_index, call in enumerate(turn_result.tool_calls):
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
                                trace_id = str(uuid.uuid4())
                                logger.error(
                                    f"Tool execution failed [trace_id: {trace_id}] - Tool: '{call.name}' on agent '{current_agent.name}': {err}",
                                    exc_info=True,
                                )
                                tool_results.append(
                                    ToolResult(
                                        tool_call_id=call.id,
                                        name=call.name,
                                        content=f"Tool execution failed [trace_id: {trace_id}]. Please try a different approach or inform the user.",
                                        is_error=True,
                                    )
                                )
                                continue

                            # Check if tool execution resulted in a Handoff
                            if isinstance(raw_output, Handoff):
                                target_name = raw_output.target_agent_name
                                target_agent = self._resolve_handoff_target(
                                    current_agent, target_name
                                )

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
                                    # Preserve call/result parity without executing the skipped tools.
                                    tool_results.extend(
                                        ToolResult(
                                            tool_call_id=skipped_call.id,
                                            name=skipped_call.name,
                                            content="Tool skipped because the maximum agent transfers were exceeded for this turn.",
                                            is_error=True,
                                        )
                                        for skipped_call in turn_result.tool_calls[call_index + 1 :]
                                    )
                                    break

                                session.set_active_agent(target_agent, raw_output.context_updates)
                                last_handoff = raw_output
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

            except asyncio.CancelledError:
                session.restore_history(
                    checkpoint_history,
                    active_agent=checkpoint_agent,
                    metadata=checkpoint_metadata,
                )
                raise
            except Exception:
                session.restore_history(
                    checkpoint_history,
                    active_agent=checkpoint_agent,
                    metadata=checkpoint_metadata,
                )
                raise

    def spawn_turn(
        self,
        session_id: str,
        user_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> "asyncio.Task[AgentResponse]":
        """Start a turn as an independent task, outside any turn that is running.

        Meant for tools that need a follow-up turn for their own session: the new turn
        waits until the current turn of that session ends, then runs. It is not a nested
        call: **never await the returned task inside the turn that spawned it** (that turn
        would wait for a turn that waits for it). Await it, or read ``task.exception()``,
        after that turn ends if you need the result or its errors.

        The runner keeps a reference to the task until it finishes. Must be called while an
        event loop is running.
        """
        context = contextvars.copy_context()
        context.run(_turn_lineage.set, ())
        coro = self.run_async(session_id, user_message, metadata=metadata)
        task: asyncio.Task[AgentResponse] = context.run(asyncio.ensure_future, coro)
        self._spawned_turns.add(task)
        task.add_done_callback(self._spawned_turns.discard)
        return task

    def run(
        self,
        session_id: str,
        user_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Synchronous convenience wrapper for run_async.

        Each call runs on its own event loop inside its own ``async with runner:`` scope, so
        the turn's Gemini requests share one pooled client, closed before the call returns.

        Raises RuntimeError if called from a running event loop; use
        ``await runner.run_async(...)`` instead in that case.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if sys.version_info >= (3, 11) and _TEARDOWN_OBSERVABLE:
                # What asyncio.run does, but run() learns when the teardown starts.
                teardown = _LoopTeardown()
                with asyncio.Runner() as loop_runner:
                    try:
                        return loop_runner.run(
                            self._run_in_own_loop(
                                session_id, user_message, metadata=metadata, teardown=teardown
                            )
                        )
                    except BaseException:
                        # Closing loop_runner (the `with` exit) cancels exactly these tasks
                        # next, before the loop runs again.
                        teardown.tasks = asyncio.all_tasks(loop_runner.get_loop())
                        raise
            return asyncio.run(
                self._run_in_own_loop(session_id, user_message, metadata=metadata, teardown=None)
            )
        raise RuntimeError(
            "Runner.run() cannot be called from a running event loop. "
            "Use `await runner.run_async(...)` instead."
        )

    async def _run_in_own_loop(
        self,
        session_id: str,
        user_message: str,
        metadata: dict[str, Any] | None = None,
        *,
        teardown: _LoopTeardown | None,
    ) -> AgentResponse:
        """The coroutine ``run()`` runs on its own loop: one turn inside its own scope."""
        await self.__aenter__()
        try:
            turn_interrupted = False
            try:
                response = await self.run_async(session_id, user_message, metadata=metadata)
            except BaseException as err:
                turn_interrupted = not isinstance(err, Exception)
                raise
            finally:
                # The step asyncio.run performs next, done before the scope closes so that
                # nothing still uses the pooled client when it is closed.
                await _cancel_other_tasks(turn_interrupted, teardown)
        except BaseException:
            try:
                await self.__aexit__(None, None, None)
            except Exception as err:
                # Keep the error the turn failed with; report this one.
                logger.error(f"Failed to close the runner's connection scope: {err!r}")
            raise
        await self.__aexit__(None, None, None)
        return response

    async def __aenter__(self) -> "Runner":
        """Open a connection-pooling scope for the running event loop.

        While the scope is open, Gemini requests the runner makes on this loop share one
        pooled HTTP client, so connections are reused across turns. Engines with their own
        ``http_client`` keep using it. Entering opens no connection. Scopes nest per loop,
        and scopes on other loops or threads are independent. The runner remains usable after
        the scope closes; requests then use a client per attempt again.
        """
        loop = asyncio.get_running_loop()
        with self._client_pools_lock:
            pool, depth = self._client_pools.get(loop) or (None, 0)
            self._client_pools[loop] = (pool or _ClientPool(), depth + 1)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the scope; the outermost exit on a loop closes that loop's pooled client.

        Requests that start afterwards use a client per attempt. Requests already using the
        pooled client finish first; nothing is cancelled. Exceptions from the ``async with``
        body propagate unchanged.
        """
        loop = asyncio.get_running_loop()
        with self._client_pools_lock:
            entry = self._client_pools.get(loop)
            if entry is None:
                raise RuntimeError(
                    "Runner.__aexit__() was called on an event loop without an open "
                    "`async with runner:` scope."
                )
            pool, depth = entry
            if depth > 1:
                self._client_pools[loop] = (pool, depth - 1)
                return
            del self._client_pools[loop]
            pool.stop_lending()
        await pool.aclose()

    def _client_pool_for_running_loop(self) -> _ClientPool | None:
        """The pool of the scope open on the running loop, if any."""
        loop = asyncio.get_running_loop()
        with self._client_pools_lock:
            entry = self._client_pools.get(loop)
        return entry[0] if entry is not None else None

    def __repr__(self) -> str:
        return (
            f"<Runner starting_agent={self.starting_agent.name!r} "
            f"registered_agents={list(self._agents_registry.keys())}>"
        )
