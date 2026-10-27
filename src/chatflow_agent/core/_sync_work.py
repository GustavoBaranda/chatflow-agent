"""Track synchronous tool calls that outlive the asyncio task awaiting them."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

_T = TypeVar("_T")


class _Waiter:
    __slots__ = ("future", "loop")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.future: asyncio.Future[None] = loop.create_future()


class _Entry:
    __slots__ = ("pending", "session_id", "store", "waiters")

    def __init__(self, store: object, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self.pending = 0
        self.waiters: set[_Waiter] = set()


class _Registry:
    """Thread-safe pending-work registry keyed by store identity and session id."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        # Entries hold the store strongly only while real work is pending. That prevents
        # id reuse without retaining inactive stores or sessions indefinitely.
        self._entries: dict[tuple[int, str], _Entry] = {}

    def start(self, store: object, session_id: str) -> _Ticket:
        key = (id(store), session_id)
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(store, session_id)
                self._entries[key] = entry
            elif entry.store is not store:
                # The existing entry keeps its store alive, so this is defensive only.
                raise RuntimeError("Synchronous-work registry identity collision")
            entry.pending += 1
        return _Ticket(self, key, entry)

    def claim(self, ticket: _Ticket) -> bool:
        with self._guard:
            if ticket._state != "pending":
                return False
            ticket._state = "started"
            return True

    def abandon(self, ticket: _Ticket) -> None:
        with self._guard:
            if ticket._state != "pending":
                return
            ticket._state = "finished"
            waiters = self._finish_locked(ticket)
        self._wake(waiters)

    def finish(self, ticket: _Ticket) -> None:
        with self._guard:
            if ticket._state != "started":
                return
            ticket._state = "finished"
            waiters = self._finish_locked(ticket)
        self._wake(waiters)

    def _finish_locked(self, ticket: _Ticket) -> list[_Waiter]:
        entry = ticket._entry
        entry.pending -= 1
        if entry.pending:
            return []
        current = self._entries.get(ticket._key)
        if current is entry:
            del self._entries[ticket._key]
        waiters = list(entry.waiters)
        entry.waiters.clear()
        return waiters

    @staticmethod
    def _wake(waiters: list[_Waiter]) -> None:
        for waiter in waiters:
            try:
                waiter.loop.call_soon_threadsafe(_resolve_waiter, waiter.future)
            except RuntimeError:
                # The waiter's loop was closed. Its task cannot resume and the waiter has
                # already been removed from the registry.
                pass

    def add_waiter(self, store: object, session_id: str, waiter: _Waiter) -> _Entry | None:
        key = (id(store), session_id)
        with self._guard:
            entry = self._entries.get(key)
            if entry is None or entry.store is not store:
                return None
            entry.waiters.add(waiter)
            return entry

    def remove_waiter(self, entry: _Entry, waiter: _Waiter) -> None:
        with self._guard:
            entry.waiters.discard(waiter)

    def is_running_for(self, ticket: _Ticket, store: object, session_id: str) -> bool:
        """Whether ``ticket`` is still executing for exactly this store and session."""
        with self._guard:
            entry = ticket._entry
            return (
                ticket._state == "started"
                and entry.store is store
                and entry.session_id == session_id
            )


class _Ticket:
    """One submitted sync call, including the not-yet-started executor state."""

    __slots__ = ("_entry", "_key", "_registry", "_state")

    def __init__(self, registry: _Registry, key: tuple[int, str], entry: _Entry) -> None:
        self._registry = registry
        self._key = key
        self._entry = entry
        self._state = "pending"

    def run(self, call: Callable[..., _T], kwargs: dict[str, Any]) -> _T:
        if not self._registry.claim(self):
            # Cancellation won the race before the executor began this job.
            return cast(_T, None)
        token = _running_ticket.set(self)
        try:
            return call(**kwargs)
        finally:
            try:
                self._registry.finish(self)
            finally:
                _running_ticket.reset(token)

    def abandon(self) -> None:
        self._registry.abandon(self)


class _Scope:
    __slots__ = ("active", "session_id", "store")

    def __init__(self, store: object, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self.active = True


_registry = _Registry()
_scope: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "chatflow_sync_work_scope", default=None
)
_running_ticket: contextvars.ContextVar[_Ticket | None] = contextvars.ContextVar(
    "chatflow_running_sync_ticket", default=None
)
_tasks: set[asyncio.Task[Any]] = set()
_tasks_guard = threading.Lock()


def _resolve_waiter(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _task_done(task: asyncio.Task[Any], ticket: _Ticket) -> None:
    with _tasks_guard:
        _tasks.discard(task)
    if task.cancelled():
        # If the executor job never claimed the ticket, it will never run its finally.
        ticket.abandon()
        return
    # Retrieve late exceptions even when the turn that awaited the task was cancelled.
    task.exception()


def _is_running_sync_work(store: object, session_id: str) -> bool:
    """Whether this context belongs to a still-running sync tool for the session.

    Context copies retain the ticket object rather than a static session marker. Once the
    callable finishes, its ticket becomes inactive in every copied context, which prevents
    stale false positives in independent work that starts later.
    """
    ticket = _running_ticket.get()
    return ticket is not None and _registry.is_running_for(ticket, store, session_id)


async def _run_sync_tool(call: Callable[..., _T], kwargs: dict[str, Any]) -> _T:
    """Run a sync tool in a thread and track its real lifetime inside a Runner turn."""
    scope = _scope.get()
    if scope is None or not scope.active:
        return await asyncio.to_thread(call, **kwargs)

    ticket = _registry.start(scope.store, scope.session_id)
    try:
        task = asyncio.create_task(asyncio.to_thread(ticket.run, call, kwargs))
    except BaseException:
        ticket.abandon()
        raise
    with _tasks_guard:
        _tasks.add(task)
    task.add_done_callback(lambda finished: _task_done(finished, ticket))
    # Shielding keeps the task alive after the turn is cancelled. The callable itself
    # cannot be cancelled, and its wrapper must reach finally to open the session barrier.
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # This withdraws a queued executor job when possible. If the callable already
        # started, cancelling its asyncio wrapper cannot stop it; the ticket's atomic
        # state keeps the barrier closed until the callable's finally runs.
        task.cancel()
        raise


@contextmanager
def _track_sync_work(store: object, session_id: str) -> Iterator[None]:
    """Associate sync tools started in the body with one session barrier."""
    scope = _Scope(store, session_id)
    token = _scope.set(scope)
    try:
        yield
    finally:
        scope.active = False
        _scope.reset(token)


async def _wait_for_sync_work(store: object, session_id: str) -> None:
    """Wait without blocking the loop until the session has no pending sync calls."""
    loop = asyncio.get_running_loop()
    while True:
        waiter = _Waiter(loop)
        entry = _registry.add_waiter(store, session_id, waiter)
        if entry is None:
            return
        try:
            await waiter.future
        finally:
            # Required for cancellation and timeout; harmless after normal wake-up.
            _registry.remove_waiter(entry, waiter)
