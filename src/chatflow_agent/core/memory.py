"""Session memory management and context tracking for multi-agent conversations."""

import asyncio
import contextvars
import copy
import json
import logging
import sqlite3
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar, cast

from chatflow_agent.core.agent import Agent
from chatflow_agent.exceptions import SessionDeletedError
from chatflow_agent.types import Message, Role

logger = logging.getLogger("chatflow_agent.memory")

# Canonical Single Source of Truth for message deduplication TTL
DEFAULT_DEDUP_TTL_HOURS: int = 192  # 8 days (exceeds Meta's 7-day webhook retry window)
DEFAULT_DEDUP_TTL_SECONDS: float = float(DEFAULT_DEDUP_TTL_HOURS * 3600)

# Worker threads per file-backed SQLiteSessionStore for calls made from async code.
# Writes to one SQLite file are serialized anyway; a small private pool keeps a stalled
# database from filling the default executor that sync tools and asyncio.to_thread use.
_BLOCKING_IO_WORKERS = 4

_T = TypeVar("_T")


class SessionContext:
    """Stores conversation state, dialogue history, and active agent pointer for a session."""

    def __init__(
        self,
        session_id: str,
        initial_agent: Agent,
        max_turns: int = 40,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if max_turns<=0:
            raise ValueError(
                f"max_turns must be positive,got {max_turns}"
            )
        self.session_id = session_id
        self.active_agent = initial_agent
        self.max_turns = max_turns
        self.history: list[Message] = []
        self.metadata: dict[str, Any] = metadata or {}
        self._lock: asyncio.Lock | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """Asynchronous lock to serialize concurrent message processing for this session."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def add_message(self, message: Message) -> None:
        """Append a message to the dialogue history and apply sliding window pruning."""
        self.history.append(message)
        self._prune_history()

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Merge metadata updates without changing the active agent or history."""
        self.metadata.update(updates)

    def set_active_agent(
        self,
        agent: Agent,
        context_updates: dict[str, Any] | None = None,
    ) -> None:
        """Switch the active agent and merge any state forwarded with the handoff."""
        self.active_agent = agent
        if context_updates:
            self.metadata.update(context_updates)

    def clear_history(self) -> None:
        """Reset the conversation dialogue history."""
        self.history.clear()

    def refresh(self) -> None:
        """Reload the committed session state from the backing store.

        Called by ``Runner`` after acquiring ``lock`` and before taking the rollback
        checkpoint. In-memory sessions already hold the latest state, so this is a no-op.
        """

    def rollback_to(self, checkpoint_len: int) -> None:
        """Roll back dialogue history to a previous checkpoint length.

        Note: Rollback reverts conversational dialogue history; it does NOT
        revert external side effects of tools that were already executed.
        """
        self.history = self.history[:checkpoint_len]

    def restore_history(
        self,
        history: list[Message],
        active_agent: Agent | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Restore a complete history snapshot, including entries pruned after it was taken."""
        self.history = list(history)
        if active_agent is not None:
            self.active_agent = active_agent
        if metadata is not None:
            self.metadata = copy.deepcopy(metadata)

    def _prune_history(self) -> None:
        """Keep dialogue history within max_turns by complete turns.

        Preserves system prompt at the beginning. Discards older turns forward so the
        conversation window always begins with a Role.USER message, guaranteeing that
        tool_call and tool_result pairs are never severed.
        """
        system_messages = [m for m in self.history if m.role == Role.SYSTEM]
        non_system = [m for m in self.history if m.role != Role.SYSTEM]

        # Group non-system messages into complete user turns (each turn starts with Role.USER)
        turns: list[list[Message]] = []
        current_turn: list[Message] = []

        for msg in non_system:
            if msg.role == Role.USER:
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                if current_turn:
                    current_turn.append(msg)
                else:
                    # Discard forward any non-USER message until first USER message appears
                    continue

        if current_turn:
            turns.append(current_turn)

        # Retain only the most recent max_turns complete turns
        retained_turns = turns[-self.max_turns :] if len(turns) > self.max_turns else turns

        flattened: list[Message] = []
        for turn in retained_turns:
            flattened.extend(turn)

        self.history = system_messages + flattened

    def __repr__(self) -> str:
        return (
            f"<SessionContext id={self.session_id!r} active_agent={self.active_agent.name!r} "
            f"turns={len(self.history)}>"
        )


class BaseSessionStore(ABC):
    """Abstract interface for session registries."""

    @abstractmethod
    def get_or_create(
        self,
        session_id: str,
        default_agent: Agent,
        metadata: dict[str, Any] | None = None,
    ) -> SessionContext:
        """Retrieve an existing session context or instantiate a new one."""
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> SessionContext | None:
        """Fetch session by ID if it exists."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Remove a session from storage."""
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all active session identifiers."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored sessions."""
        raise NotImplementedError


    def is_message_processed(self, message_id: str) -> bool:
        """Return True if message_id was already recorded."""
        return False

    def record_processed_message(self, message_id: str) -> bool:
        """Atomically record message_id. Returns True if new, False if duplicate."""
        return True


class SessionStore(BaseSessionStore):
    """In-memory session registry with thread-safe at-most-once dedup (TTL + cap).

    Deduplication covers Meta's 7-day retry window using an 8-day default TTL.
    Entries are capped at 10,000 and purged amortized every 5 minutes.
    """

    _DEDUP_TTL_SECONDS: float = DEFAULT_DEDUP_TTL_SECONDS
    _DEDUP_MAX_ENTRIES: int = 10_000
    _DEDUP_PURGE_INTERVAL: float = 300.0          # throttle: purge at most every 5 min

    def __init__(
        self,
        default_max_turns: int = 40,
        dedup_ttl_hours: float = DEFAULT_DEDUP_TTL_HOURS,
    ) -> None:
        import time as _time
        if default_max_turns <= 0:
            raise ValueError(
        f"default_max_turns must be a positive integer, got {default_max_turns}"
    )
        self._sessions: dict[str, SessionContext] = {}
        self.default_max_turns = default_max_turns
        self.dedup_ttl_hours = dedup_ttl_hours
        self._dedup_ttl_seconds = dedup_ttl_hours * 3600.0
        self._dedup_lock = threading.Lock()
        self._processed_ids: dict[str, float] = {}   # {wamid: monotonic timestamp}
        self._last_purge: float = _time.monotonic()

    def is_message_processed(self, message_id: str) -> bool:
        """Return True if message_id is within TTL."""
        import time as _time

        with self._dedup_lock:
            now = _time.monotonic()
            ts = self._processed_ids.get(message_id)
            if ts is None:
                return False
            if now - ts > self._dedup_ttl_seconds:
                del self._processed_ids[message_id]
                return False
            return True

    def record_processed_message(self, message_id: str) -> bool:
        """Atomically record message_id. Returns True if new, False if duplicate.

        Applies amortized TTL purge and caps entries at _DEDUP_MAX_ENTRIES.
        """
        import time as _time

        with self._dedup_lock:
            now = _time.monotonic()

            existing = self._processed_ids.get(message_id)
            if existing is not None:
                if now - existing <= self._dedup_ttl_seconds:
                    return False  # within TTL - duplicate
                del self._processed_ids[message_id]  # stale - allow re-registration

            # Amortized purge
            if now - self._last_purge >= self._DEDUP_PURGE_INTERVAL:
                cutoff = now - self._dedup_ttl_seconds
                self._processed_ids = {
                    k: v for k, v in self._processed_ids.items() if v > cutoff
                }
                self._last_purge = now

            # Cap enforcement
            if len(self._processed_ids) >= self._DEDUP_MAX_ENTRIES:
                oldest = min(self._processed_ids, key=lambda k: self._processed_ids[k])
                del self._processed_ids[oldest]

            self._processed_ids[message_id] = now
            return True

    def get_or_create(
        self,
        session_id: str,
        default_agent: Agent,
        metadata: dict[str, Any] | None = None,
    ) -> SessionContext:
        """Retrieve an existing session context or instantiate a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(
                session_id=session_id,
                initial_agent=default_agent,
                max_turns=self.default_max_turns,
                metadata=metadata,
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> SessionContext | None:
        """Fetch session by ID if it exists."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Remove a session from storage."""
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        """List all active session identifiers."""
        return list(self._sessions.keys())

    def clear(self) -> None:
        """Delete all stored sessions."""
        self._sessions.clear()


def _is_memory_sqlite_path(db_path: str) -> bool:
    """Return True for ephemeral SQLite memory paths (:memory: or shared-cache URI)."""
    if db_path == ":memory:":
        return True
    return db_path.startswith("file:") and "mode=memory" in db_path


@contextmanager
def _get_sqlite_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite connection configured with WAL and busy_timeout."""
    use_uri = db_path.startswith("file:")
    conn = sqlite3.connect(db_path, uri=use_uri, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    if not _is_memory_sqlite_path(db_path):
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class _SessionLockRegistry:
    """Hands out one ``asyncio.Lock`` per (running event loop, session_id).

    Locks are keyed by loop because an ``asyncio.Lock`` becomes bound to the first loop
    that waits on it; ``Runner.run()`` starts a new loop for every call. Both levels are
    weak: a lock lives while a session context uses it, and a loop's entries disappear
    with the loop, so the registry does not grow with the number of sessions.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, weakref.WeakValueDictionary[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()

    def get(self, loop: asyncio.AbstractEventLoop, session_id: str) -> asyncio.Lock:
        with self._guard:
            loop_locks = self._locks.get(loop)
            if loop_locks is None:
                loop_locks = weakref.WeakValueDictionary()
                self._locks[loop] = loop_locks
            lock = loop_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                loop_locks[session_id] = lock
            return lock


class SQLiteSessionContext(SessionContext):
    """Persistent SQLite-backed session context that updates disk on changes."""

    def __init__(
        self,
        session_id: str,
        initial_agent: Agent,
        db_path: str | Path,
        max_turns: int = 40,
        metadata: dict[str, Any] | None = None,
        preloaded_history: list[Message] | None = None,
        store: "SQLiteSessionStore | None" = None,
        *,
        owner: "SQLiteSessionStore | None" = None,
        default_agent: Agent | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError(f"max_turns must be a positive integer, got {max_turns}")
        super().__init__(
            session_id=session_id,
            initial_agent=initial_agent,
            max_turns=max_turns,
            metadata=metadata,
        )
        self.db_path = str(db_path)
        # In-memory contexts bind to their store so writes share one locked connection
        # and fail clearly after store.close() (ephemeral DB is gone with the keeper).
        self._store = store
        # The store that created this context (disk or memory). It supplies the shared
        # session lock and the agent resolver used by refresh().
        self._owner = owner
        self._default_agent = default_agent if default_agent is not None else initial_agent
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        # Turn journal, active between refresh() and restore_history(): what this turn
        # wrote, so a rollback can tell its own changes from other writers'.
        self._journal_active = False
        self._checkpoint_row_ids: set[int] = set()
        self._turn_inserted_ids: list[int] = []
        self._turn_pruned_rows: dict[int, tuple[str, str | None, str]] = {}
        self._turn_written_agent: str | None = None
        self._turn_written_metadata: str | None = None
        # Session row as stored when the turn started, to recognise other writers' changes.
        self._checkpoint_agent_name: str | None = None
        self._checkpoint_metadata_json: str | None = None
        if preloaded_history:
            self.history = list(preloaded_history)

    @property
    def lock(self) -> asyncio.Lock:
        """Lock shared by every context of this session within the running event loop."""
        if self._owner is None:
            return super().lock
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: nothing can await the lock here; keep a private one.
            return super().lock
        if self._lock is None or self._lock_loop is not loop:
            self._lock = self._owner._session_locks.get(loop, self.session_id)
            self._lock_loop = loop
        return self._lock

    def refresh(self) -> None:
        """Reload history, active agent and metadata committed by earlier turns.

        Also starts the turn journal used by ``restore_history()``. If the session was
        deleted while this turn waited for the lock, it starts again from an empty session
        with the default agent; rows orphaned by the deletion are discarded, never reloaded.
        """
        with self._db() as conn:
            row = conn.execute(
                "SELECT active_agent_name, metadata_json FROM chatflow_sessions WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            if row is None:
                logger.warning(
                    "Session %r was deleted while a turn was waiting for it; "
                    "the turn starts a new, empty session.",
                    self.session_id,
                )
                conn.execute(
                    "DELETE FROM chatflow_messages WHERE session_id = ?", (self.session_id,)
                )
                conn.execute(
                    """
                    INSERT INTO chatflow_sessions (session_id, active_agent_name, metadata_json)
                    VALUES (?, ?, '{}')
                    """,
                    (self.session_id, self._default_agent.name),
                )
                agent_name, metadata_json, msg_rows = self._default_agent.name, "{}", []
            else:
                agent_name, metadata_json = row["active_agent_name"], row["metadata_json"]
                msg_rows = conn.execute(
                    """
                    SELECT id, message_json FROM chatflow_messages
                    WHERE session_id = ? ORDER BY id ASC
                    """,
                    (self.session_id,),
                ).fetchall()

        history = [Message.model_validate_json(r["message_json"]) for r in msg_rows]
        metadata = json.loads(metadata_json) if metadata_json else {}
        # Same resolution as SQLiteSessionStore.get_or_create().
        active_agent = self._default_agent
        resolver = self._owner._agent_resolver if self._owner is not None else None
        if resolver is not None:
            resolved = resolver(agent_name)
            if resolved is not None:
                active_agent = resolved

        self.history = history
        self.metadata = metadata
        self.active_agent = active_agent
        self._journal_active = True
        self._checkpoint_agent_name = agent_name
        self._checkpoint_metadata_json = metadata_json
        self._checkpoint_row_ids = {r["id"] for r in msg_rows}
        self._turn_inserted_ids = []
        self._turn_pruned_rows = {}
        self._turn_written_agent = None
        self._turn_written_metadata = None

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a DB connection via the owning store when bound, else a direct path connect."""
        if self._store is not None:
            with self._store._connection() as conn:
                yield conn
        else:
            with _get_sqlite_connection(self.db_path) as conn:
                yield conn

    def add_message(self, message: Message) -> None:
        """Append message to memory, commit to SQLite, and enforce FIFO message pruning.

        Raises ``SessionDeletedError`` if the session was deleted meanwhile; nothing is
        stored then, so a deleted session never receives messages from a running turn.
        """
        previous_history = list(self.history)
        super().add_message(message)
        pruned_rows: dict[int, tuple[str, str | None, str]] = {}
        with self._db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chatflow_messages (session_id, role, content, message_json)
                VALUES (?, ?, ?, ?)
                """,
                (self.session_id, message.role.value, message.content, message.model_dump_json()),
            )
            inserted_id = cursor.lastrowid
            # Prune old turns from SQLite: find the earliest user message ID to retain
            # To keep max_turns complete turns, the oldest retained user message is at offset max_turns - 1
            retain_offset = max(0, self.max_turns - 1)
            cutoff_row = conn.execute(
                """
                SELECT id FROM chatflow_messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY id DESC
                LIMIT 1 OFFSET ?
                """,
                (self.session_id, retain_offset),
            ).fetchone()

            if cutoff_row is not None:
                cutoff_id = cutoff_row["id"]
                if self._journal_active:
                    for pruned in conn.execute(
                        """
                        SELECT id, role, content, message_json FROM chatflow_messages
                        WHERE session_id = ? AND role != 'system' AND id < ?
                        """,
                        (self.session_id, cutoff_id),
                    ).fetchall():
                        pruned_rows[pruned["id"]] = (
                            pruned["role"],
                            pruned["content"],
                            pruned["message_json"],
                        )
                conn.execute(
                    """
                    DELETE FROM chatflow_messages
                    WHERE session_id = ? AND role != 'system' AND id < ?
                    """,
                    (self.session_id, cutoff_id),
                )
            session_row = conn.execute(
                """
                UPDATE chatflow_sessions
                SET active_agent_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (self.active_agent.name, self.session_id),
            )
            if session_row.rowcount == 0:
                # Raised before the commit: the INSERT and pruning above are rolled back.
                self.history = previous_history
                raise SessionDeletedError(
                    f"Session '{self.session_id}' was deleted; the message was not stored."
                )
        # Journal only what was committed.
        if self._journal_active:
            if inserted_id is not None:
                self._turn_inserted_ids.append(inserted_id)
            self._turn_pruned_rows.update(pruned_rows)
            self._turn_written_agent = self.active_agent.name

    def rollback_to(self, checkpoint_len: int) -> None:
        """Roll back dialogue history in memory and database to checkpoint length.

        Note: Rollback reverts conversational dialogue history; it does NOT
        revert external side effects of tools that were already executed.
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id FROM chatflow_messages WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            ).fetchall()
            if len(rows) > checkpoint_len:
                cutoff_id = rows[checkpoint_len]["id"]
                conn.execute(
                    "DELETE FROM chatflow_messages WHERE session_id = ? AND id >= ?",
                    (self.session_id, cutoff_id),
                )
        super().rollback_to(checkpoint_len)

    def restore_history(
        self,
        history: list[Message],
        active_agent: Agent | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Restore a history snapshot in memory and persist it to SQLite.

        During a ``Runner`` turn (after ``refresh()``), checking for other writers and
        rolling back happen in one ``BEGIN IMMEDIATE`` transaction:

        * If only this turn changed the session since its checkpoint (same message rows
          apart from its own, and the stored agent and metadata are still the ones it
          started from or last wrote), the snapshot is written back in full. This also
          restores rows pruned during the turn.
        * Otherwise another writer changed the session meanwhile (another context of the
          same store, another store instance, process or thread, a reset or a delete).
          Only this turn's own changes are reverted and a warning is logged: rows, agent
          and metadata written by others are kept, and a deleted session is not recreated.
          Rows this turn pruned are put back unless the other writer deleted rows itself
          (restoring them could undo a reset, and a concurrent prune would have removed
          them too). The in-memory state of this context is still set to the snapshot.

        Agent and metadata are compared by value: if another writer stores exactly the
        value this turn wrote, the change cannot be told apart from the turn's own.

        Outside a turn, the snapshot is written back in full, as before.
        """
        restored_agent = active_agent if active_agent is not None else self.active_agent
        restored_metadata = (
            copy.deepcopy(metadata) if metadata is not None else dict(self.metadata)
        )
        metadata_json = json.dumps(restored_metadata)
        with self._db() as conn:
            if not self._journal_active:
                self._write_snapshot(conn, history, restored_agent.name, metadata_json)
            else:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT active_agent_name, metadata_json FROM chatflow_sessions WHERE session_id = ?",
                    (self.session_id,),
                ).fetchone()
                current_ids = {
                    r["id"]
                    for r in conn.execute(
                        "SELECT id FROM chatflow_messages WHERE session_id = ?",
                        (self.session_id,),
                    ).fetchall()
                }
                expected_ids = (self._checkpoint_row_ids - self._turn_pruned_rows.keys()) | set(
                    self._turn_inserted_ids
                )
                expected_agent = (
                    self._turn_written_agent
                    if self._turn_written_agent is not None
                    else self._checkpoint_agent_name
                )
                expected_metadata = (
                    self._turn_written_metadata
                    if self._turn_written_metadata is not None
                    else self._checkpoint_metadata_json
                )
                if (
                    row is not None
                    and current_ids == expected_ids
                    and row["active_agent_name"] == expected_agent
                    and row["metadata_json"] == expected_metadata
                ):
                    self._write_snapshot(conn, history, restored_agent.name, metadata_json)
                else:
                    self._revert_own_changes(
                        conn, row, current_ids, restored_agent.name, metadata_json
                    )
        self._journal_active = False
        super().restore_history(history, active_agent=active_agent, metadata=metadata)

    def _write_snapshot(
        self,
        conn: sqlite3.Connection,
        history: list[Message],
        agent_name: str,
        metadata_json: str,
    ) -> None:
        """Replace the persisted session with ``history``, ``agent_name`` and metadata."""
        conn.execute(
            "DELETE FROM chatflow_messages WHERE session_id = ?",
            (self.session_id,),
        )
        conn.executemany(
            """
            INSERT INTO chatflow_messages (session_id, role, content, message_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    self.session_id,
                    message.role.value,
                    message.content,
                    message.model_dump_json(),
                )
                for message in history
            ],
        )
        conn.execute(
            """
            UPDATE chatflow_sessions
            SET active_agent_name = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            (agent_name, metadata_json, self.session_id),
        )

    def _revert_own_changes(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row | None,
        current_ids: set[int],
        agent_name: str,
        metadata_json: str,
    ) -> None:
        """Undo only what this turn wrote, leaving other writers' changes in place."""
        logger.warning(
            "Session %r was modified outside its lock during a failed turn; "
            "reverting only this turn's own changes. Concurrent use of one session from "
            "several store instances, processes or threads is not supported.",
            self.session_id,
        )
        conn.executemany(
            "DELETE FROM chatflow_messages WHERE id = ?",
            [(row_id,) for row_id in self._turn_inserted_ids if row_id in current_ids],
        )
        if row is None:
            # The session was deleted: do not bring back its rows or its session record.
            return
        removed_by_others = (self._checkpoint_row_ids - self._turn_pruned_rows.keys()) - current_ids
        if not removed_by_others:
            # Put back what this turn pruned, with the original ids so ordering is kept.
            # Skipped when another writer deleted rows (e.g. a reset): restoring them would
            # undo that deletion.
            conn.executemany(
                """
                INSERT INTO chatflow_messages (id, session_id, role, content, message_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (row_id, self.session_id, role, content, message_json)
                    for row_id, (role, content, message_json) in sorted(
                        self._turn_pruned_rows.items()
                    )
                    if row_id not in current_ids
                ],
            )
        # Restore the agent and metadata only where the stored value is still ours.
        if (
            self._turn_written_agent is not None
            and row["active_agent_name"] == self._turn_written_agent
        ):
            conn.execute(
                "UPDATE chatflow_sessions SET active_agent_name = ? WHERE session_id = ?",
                (agent_name, self.session_id),
            )
        if (
            self._turn_written_metadata is not None
            and row["metadata_json"] == self._turn_written_metadata
        ):
            conn.execute(
                "UPDATE chatflow_sessions SET metadata_json = ? WHERE session_id = ?",
                (metadata_json, self.session_id),
            )

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Persist merged metadata before updating the in-memory state."""
        if not updates:
            return
        updated_metadata = {**self.metadata, **updates}
        metadata_json = json.dumps(updated_metadata)
        with self._db() as conn:
            conn.execute(
                """
                UPDATE chatflow_sessions
                SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (metadata_json, self.session_id),
            )
        if self._journal_active:
            self._turn_written_metadata = metadata_json
        super().update_metadata(updates)

    def set_active_agent(
        self,
        agent: Agent,
        context_updates: dict[str, Any] | None = None,
    ) -> None:
        """Switch active agent and persist the handoff state to SQLite."""
        updated_metadata = dict(self.metadata)
        if context_updates:
            updated_metadata.update(context_updates)
        metadata_json = json.dumps(updated_metadata)
        with self._db() as conn:
            conn.execute(
                """
                UPDATE chatflow_sessions
                SET active_agent_name = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (agent.name, metadata_json, self.session_id),
            )
        if self._journal_active:
            self._turn_written_agent = agent.name
            self._turn_written_metadata = metadata_json
        super().set_active_agent(agent, context_updates)

    def clear_history(self) -> None:
        """Reset conversation dialogue history in memory and database."""
        super().clear_history()
        with self._db() as conn:
            conn.execute("DELETE FROM chatflow_messages WHERE session_id = ?", (self.session_id,))
            conn.execute(
                """
                UPDATE chatflow_sessions
                SET active_agent_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (self.active_agent.name, self.session_id),
            )



class SQLiteSessionStore(BaseSessionStore):
    """Persistent SQLite session registry.

    Stores conversation state, dialogue turns, active agent pointers, and deduplication
    records in a local or volume-mounted SQLite database with zero external dependencies.

    Note on Concurrency:
        Configured with WAL mode and PRAGMA busy_timeout=10000 ms. Designed for
        single-process deployments (1 Uvicorn worker). For multi-worker or multi-node
        clusters, use an external distributed session store to avoid SQLite file lock contention.

        Within one process, one store instance and one running event loop, ``Runner``
        turns for the same ``session_id`` run one at a time: every context of a session
        shares one lock per event loop, and each turn reloads the committed state after
        acquiring it. A failed or cancelled turn restores the state it started from,
        including messages pruned during the turn, and never deletes messages committed
        by another turn.

        Not provided: serialization across store instances on the same file, across
        processes, or across threads running their own event loops. In those cases a
        failed turn still reverts only its own changes, but turns may interleave and
        read stale history.

        With a database file, the calls ``Runner`` and the WhatsApp and Telegram channels
        make from async code run in this store's own worker threads (4, created on first
        use, shut down by ``close()``), so a slow disk or a database locked by another writer
        does not block the event loop. A call that has started cannot be interrupted: a
        cancelled turn waits for it before rolling back. ``:memory:`` stores and subclasses
        run their calls on the event loop.

        ``delete()`` and ``clear()`` do not wait for running turns. A turn that is running
        when its session is deleted fails with ``SessionDeletedError`` the next time it
        stores a message, and its rollback leaves no rows behind. A turn that was waiting
        for the session starts a new, empty one. Deleting a session erases its data; it
        does not stop new messages from creating a session with the same id again.
    """

    def __init__(
        self,
        db_path: str | Path = "chatflow.db",
        default_max_turns: int = 40,
        dedup_ttl_hours: float = DEFAULT_DEDUP_TTL_HOURS,
    ) -> None:
        if default_max_turns <= 0:
            raise ValueError(
        f"default_max_turns must be a positive integer, got {default_max_turns}")
        self.db_path = str(db_path)
        self.default_max_turns = default_max_turns
        self.dedup_ttl_hours = dedup_ttl_hours
        self._agent_resolver: Callable[[str], Agent | None] | None = None
        self._session_locks = _SessionLockRegistry()
        self._closed = False
        # Worker threads for the blocking SQLite calls made from async code (see
        # _run_blocking); created on first use, shut down by close().
        self._executor: ThreadPoolExecutor | None = None
        self._executor_guard = threading.Lock()
        self._db_lock = threading.RLock()
        self._keeper_conn: sqlite3.Connection | None = None
        # Disk paths use the user path directly. Pure :memory: DBs are connection-private,
        # so each store gets a unique shared-cache URI plus a single keeper connection that
        # owns the ephemeral database for the store lifetime (instance isolation + serialization).
        self._conn_path = self.db_path
        self._in_memory = self.db_path == ":memory:"
        if self._in_memory:
            self._conn_path = f"file:chatflow_mem_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._keeper_conn = sqlite3.connect(
                self._conn_path, uri=True, check_same_thread=False
            )
            self._keeper_conn.row_factory = sqlite3.Row
            self._keeper_conn.execute("PRAGMA busy_timeout = 10000")
        self._init_db()

    def _ensure_open(self) -> None:
        """Raise if this store was closed and can no longer serve operations."""
        if self._closed:
            raise RuntimeError("SQLiteSessionStore is closed and cannot be used")

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a transactional connection; in-memory stores share one locked keeper conn."""
        self._ensure_open()
        if self._in_memory:
            with self._db_lock:
                conn = self._keeper_conn
                if conn is None:
                    raise RuntimeError("SQLiteSessionStore is closed and cannot be used")
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        else:
            with _get_sqlite_connection(self._conn_path) as conn:
                yield conn

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        """The lock every context of ``session_id`` uses on the running event loop.

        Lets ``Runner`` wait for the session before loading it, so that turns whose
        loading runs in worker threads still start waiting in the order they arrive.
        """
        return self._session_locks.get(asyncio.get_running_loop(), session_id)

    def _blocking_executor(self) -> ThreadPoolExecutor | None:
        """Executor for blocking calls made from async code; None once the store is closed."""
        with self._executor_guard:
            if self._closed:
                return None
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=_BLOCKING_IO_WORKERS, thread_name_prefix="chatflow-sqlite"
                )
            return self._executor

    def close(self) -> None:
        """Release keeper connections and mark the store closed. Idempotent.

        Calls already running in worker threads finish on their own; close() does not
        wait for them.
        """
        if self._closed:
            return
        with self._executor_guard:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False)
        if self._keeper_conn is not None:
            with self._db_lock:
                if self._keeper_conn is not None:
                    self._keeper_conn.close()
                    self._keeper_conn = None

    def __enter__(self) -> "SQLiteSessionStore":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def set_agent_resolver(self, resolver: Callable[[str], Agent | None]) -> None:
        """Register a resolver callback to map agent names from SQLite back to Agent instances."""
        self._ensure_open()
        self._agent_resolver = resolver

    def is_message_processed(self, message_id: str) -> bool:
        """Check if message_id was recorded within dedup_ttl_hours."""
        self._ensure_open()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT message_id FROM chatflow_processed_messages
                WHERE message_id = ?
                  AND processed_at >= datetime('now', ?)
                """,
                (message_id, f"-{self.dedup_ttl_hours} hours"),
            ).fetchone()
            return row is not None

    def record_processed_message(self, message_id: str) -> bool:
        """Atomically record message_id if new; return False if duplicate within TTL.

        Purges expired records older than dedup_ttl_hours.

        The check and the write are one statement, so two concurrent callers (threads,
        or store instances on the same file) never both get True for the same id.
        """
        self._ensure_open()
        cutoff = f"-{self.dedup_ttl_hours} hours"
        with self._connection() as conn:
            # Inserts a new id, or renews one whose record expired; leaves a recorded id
            # within the TTL untouched (0 rows changed).
            cursor = conn.execute(
                """
                INSERT INTO chatflow_processed_messages (message_id, processed_at)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(message_id) DO UPDATE SET processed_at = CURRENT_TIMESTAMP
                WHERE processed_at < datetime('now', ?)
                """,
                (message_id, cutoff),
            )
            if cursor.rowcount != 1:
                return False

            conn.execute(
                "DELETE FROM chatflow_processed_messages WHERE processed_at < datetime('now', ?)",
                (cutoff,),
            )
            return True


    def _init_db(self) -> None:
        """Initialize the SQLite tables and indexes."""
        # Ensure directory exists if path contains directories (disk databases only)
        if not _is_memory_sqlite_path(self._conn_path):
            parent = Path(self.db_path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)

        with self._connection() as conn:
            conn.execute("PRAGMA user_version = 1")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chatflow_sessions (
                    session_id TEXT PRIMARY KEY,
                    active_agent_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chatflow_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    message_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chatflow_sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chatflow_processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chatflow_messages_session ON chatflow_messages(session_id, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chatflow_processed_at ON chatflow_processed_messages(processed_at)"
            )

    def get_or_create(
        self,
        session_id: str,
        default_agent: Agent,
        metadata: dict[str, Any] | None = None,
    ) -> SessionContext:
        """Retrieve existing session from SQLite or atomically insert and initialize a new one."""
        self._ensure_open()
        with self._connection() as conn:
            meta_json = json.dumps(metadata or {})
            conn.execute(
                """
                INSERT INTO chatflow_sessions (session_id, active_agent_name, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, default_agent.name, meta_json),
            )
            row = conn.execute(
                "SELECT active_agent_name, metadata_json FROM chatflow_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            agent_name = row["active_agent_name"]
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            active_agent = default_agent
            if self._agent_resolver:
                resolved = self._agent_resolver(agent_name)
                if resolved is not None:
                    active_agent = resolved

            msg_rows = conn.execute(
                "SELECT message_json FROM chatflow_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            history = [Message.model_validate_json(r["message_json"]) for r in msg_rows]

            return SQLiteSessionContext(
                session_id=session_id,
                initial_agent=active_agent,
                db_path=self._conn_path,
                max_turns=self.default_max_turns,
                metadata=meta,
                preloaded_history=history,
                # Bind only in-memory contexts: disk contexts keep independent file access.
                store=self if self._in_memory else None,
                owner=self,
                default_agent=default_agent,
            )

    def get(self, session_id: str) -> SessionContext | None:
        """Fetch session by ID if it exists in SQLite."""
        self._ensure_open()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT active_agent_name FROM chatflow_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            agent_name = row["active_agent_name"]

        active_agent = None
        if self._agent_resolver:
            active_agent = self._agent_resolver(agent_name)
        if active_agent is None:
            active_agent = Agent(name=agent_name, instructions="", model="gemini-2.5-flash")

        return self.get_or_create(session_id, default_agent=active_agent)

    def delete(self, session_id: str) -> bool:
        """Remove a session and its message history from SQLite."""
        self._ensure_open()
        with self._connection() as conn:
            conn.execute("DELETE FROM chatflow_messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM chatflow_sessions WHERE session_id = ?", (session_id,))
            return cursor.rowcount > 0

    def list_sessions(self) -> list[str]:
        """List all session IDs ordered by latest update."""
        self._ensure_open()
        with self._connection() as conn:
            rows = conn.execute("SELECT session_id FROM chatflow_sessions ORDER BY updated_at DESC").fetchall()
            return [r["session_id"] for r in rows]

    def clear(self) -> None:
        """Purge all sessions and history from SQLite."""
        self._ensure_open()
        with self._connection() as conn:
            conn.execute("DELETE FROM chatflow_messages")
            conn.execute("DELETE FROM chatflow_sessions")


def _offloads_blocking_io(store: object) -> bool:
    """Whether calls on ``store`` run in worker threads instead of the event loop.

    Only a file-backed ``SQLiteSessionStore`` itself: ``:memory:`` stores (no disk, no
    other writers), subclasses, ``SessionStore`` and custom stores keep running on the
    event loop exactly as before, so their code never has to be thread-safe.
    """
    return type(store) is SQLiteSessionStore and not store._in_memory


async def _run_blocking(store: object, call: Callable[[], _T], *, must_run: bool = False) -> _T:
    """Run a blocking session-store ``call`` without blocking the event loop.

    For a file-backed ``SQLiteSessionStore`` the call runs in the store's worker threads,
    with a copy of the current context; for any other store it runs here, as before.

    A call that has started cannot be stopped. If the awaiting task is cancelled while
    the call runs, this waits until the call has finished (its effects are then final
    and known to the caller's rollback journal) and only then re-raises the
    cancellation, so a rollback or the release of the session lock never overlaps it.
    A call that has not started yet is withdrawn and never runs, unless ``must_run``
    (rollbacks): such a call is always waited for, and if it fails its error is raised
    instead of the cancellation, so a failed rollback is never reported as done.

    After ``close()``, calls run here, as before; a store-level call then raises the
    usual "closed" error.
    """
    if not _offloads_blocking_io(store):
        return call()
    executor = cast(SQLiteSessionStore, store)._blocking_executor()
    if executor is None:
        return call()
    context = contextvars.copy_context()
    try:
        job = executor.submit(context.run, call)
    except RuntimeError:  # close() shut the executor down meanwhile
        return call()
    done = asyncio.wrap_future(job)
    try:
        return await asyncio.shield(done)
    except asyncio.CancelledError as cancelled:
        if not must_run and job.cancel():
            raise  # withdrawn before it started: it will never run
        while not done.done():
            try:
                await asyncio.shield(done)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        error = None if done.cancelled() else done.exception()
        if must_run and error is not None:
            # A failed rollback must not be hidden behind the cancellation; the error keeps
            # its own traceback, and the cancellation it interrupted is chained as its cause.
            raise error from cancelled
        # Otherwise the call's effects are final and the caller's rollback journal records
        # them; its error, if any, is superseded by the cancellation.
        raise
