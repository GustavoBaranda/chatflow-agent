"""Session memory management and context tracking for multi-agent conversations."""

import asyncio
import copy
import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

from chatflow_agent.core.agent import Agent
from chatflow_agent.types import Message, Role

# Canonical Single Source of Truth for message deduplication TTL
DEFAULT_DEDUP_TTL_HOURS: int = 192  # 8 days (exceeds Meta's 7-day webhook retry window)
DEFAULT_DEDUP_TTL_SECONDS: float = float(DEFAULT_DEDUP_TTL_HOURS * 3600)


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
    """Thread-safe in-memory session registry with at-most-once dedup (TTL + cap).

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
        self._processed_ids: dict[str, float] = {}   # {wamid: monotonic timestamp}
        self._last_purge: float = _time.monotonic()

    def is_message_processed(self, message_id: str) -> bool:
        """Return True if message_id is within TTL."""
        import time as _time
        ts = self._processed_ids.get(message_id)
        if ts is None:
            return False
        if _time.monotonic() - ts > self._dedup_ttl_seconds:
            del self._processed_ids[message_id]
            return False
        return True

    def record_processed_message(self, message_id: str) -> bool:
        """Atomically record message_id. Returns True if new, False if duplicate.

        Applies amortized TTL purge and caps entries at _DEDUP_MAX_ENTRIES.
        """
        import time as _time
        now = _time.monotonic()

        existing = self._processed_ids.get(message_id)
        if existing is not None:
            if now - existing <= self._dedup_ttl_seconds:
                return False  # within TTL - duplicate
            del self._processed_ids[message_id]  # stale - allow re-registration

        # Amortized purge
        if now - self._last_purge >= self._DEDUP_PURGE_INTERVAL:
            cutoff = now - self._dedup_ttl_seconds
            self._processed_ids = {k: v for k, v in self._processed_ids.items() if v > cutoff}
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
        if preloaded_history:
            self.history = list(preloaded_history)

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
        """Append message to memory, commit to SQLite, and enforce FIFO message pruning."""
        super().add_message(message)
        with self._db() as conn:
            conn.execute(
                """
                INSERT INTO chatflow_messages (session_id, role, content, message_json)
                VALUES (?, ?, ?, ?)
                """,
                (self.session_id, message.role.value, message.content, message.model_dump_json()),
            )
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
                conn.execute(
                    """
                    DELETE FROM chatflow_messages
                    WHERE session_id = ? AND role != 'system' AND id < ?
                    """,
                    (self.session_id, cutoff_id),
                )
            conn.execute(
                """
                UPDATE chatflow_sessions
                SET active_agent_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (self.active_agent.name, self.session_id),
            )

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
        """Restore a history snapshot in memory and persist it to SQLite."""
        restored_agent = active_agent if active_agent is not None else self.active_agent
        restored_metadata = (
            copy.deepcopy(metadata) if metadata is not None else dict(self.metadata)
        )
        metadata_json = json.dumps(restored_metadata)
        with self._db() as conn:
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
                (restored_agent.name, metadata_json, self.session_id),
            )
        super().restore_history(history, active_agent=active_agent, metadata=metadata)

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
        self._closed = False
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

    def close(self) -> None:
        """Release keeper connections and mark the store closed. Idempotent."""
        if self._closed:
            return
        self._closed = True
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
        """
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
            if row is not None:
                return False

            conn.execute(
                "DELETE FROM chatflow_processed_messages WHERE processed_at < datetime('now', ?)",
                (f"-{self.dedup_ttl_hours} hours",),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO chatflow_processed_messages (message_id, processed_at)
                VALUES (?, CURRENT_TIMESTAMP)
                """,
                (message_id,),
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
