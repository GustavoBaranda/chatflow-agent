"""Session memory management and context tracking for multi-agent conversations."""

import asyncio
import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chatflow_agent.core.agent import Agent
from chatflow_agent.types import Message, Role


class SessionContext:
    """Stores conversation state, dialogue history, and active agent pointer for a session."""

    def __init__(
        self,
        session_id: str,
        initial_agent: Agent,
        max_turns: int = 40,
        metadata: dict[str, Any] | None = None,
    ) -> None:
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

    def set_active_agent(self, agent: Agent) -> None:
        """Switch the current active agent handling this session."""
        self.active_agent = agent

    def clear_history(self) -> None:
        """Reset the conversation dialogue history."""
        self.history.clear()

    def _prune_history(self) -> None:
        """Keep the dialogue history within the max_turns threshold.

        Retains system messages at the beginning if present, and keeps the most recent turns.
        """
        max_messages = self.max_turns * 2  # 1 turn ~= 1 user message + 1 model response
        if len(self.history) > max_messages:
            # Separate any leading system messages
            system_messages = [m for m in self.history if m.role == Role.SYSTEM]
            non_system_messages = [m for m in self.history if m.role != Role.SYSTEM]

            # Slice the most recent turns
            retained_turns = non_system_messages[-max_messages:]
            self.history = system_messages + retained_turns

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

    _DEDUP_TTL_SECONDS: float = 8 * 24 * 3600   # 8 days > Meta's 7-day retry window
    _DEDUP_MAX_ENTRIES: int = 10_000
    _DEDUP_PURGE_INTERVAL: float = 300.0          # throttle: purge at most every 5 min

    def __init__(self, default_max_turns: int = 40) -> None:
        import time as _time
        self._sessions: dict[str, SessionContext] = {}
        self.default_max_turns = default_max_turns
        self._processed_ids: dict[str, float] = {}   # {wamid: monotonic timestamp}
        self._last_purge: float = _time.monotonic()

    def is_message_processed(self, message_id: str) -> bool:
        """Return True if message_id is within TTL."""
        import time as _time
        ts = self._processed_ids.get(message_id)
        if ts is None:
            return False
        if _time.monotonic() - ts > self._DEDUP_TTL_SECONDS:
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
            if now - existing <= self._DEDUP_TTL_SECONDS:
                return False  # within TTL — duplicate
            del self._processed_ids[message_id]  # stale — allow re-registration

        # Amortized purge
        if now - self._last_purge >= self._DEDUP_PURGE_INTERVAL:
            cutoff = now - self._DEDUP_TTL_SECONDS
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


@contextmanager
def _get_sqlite_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite connection that closes cleanly across operating systems."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
    ) -> None:
        super().__init__(
            session_id=session_id,
            initial_agent=initial_agent,
            max_turns=max_turns,
            metadata=metadata,
        )
        self.db_path = str(db_path)
        if preloaded_history:
            self.history = list(preloaded_history)

    def add_message(self, message: Message) -> None:
        """Append message to memory, commit to SQLite, and enforce FIFO message pruning."""
        super().add_message(message)
        with _get_sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO chatflow_messages (session_id, role, content, message_json)
                VALUES (?, ?, ?, ?)
                """,
                (self.session_id, message.role.value, message.content, message.model_dump_json()),
            )
            # Prune old SQLite rows beyond max_turns * 2
            max_messages = self.max_turns * 2
            conn.execute(
                """
                DELETE FROM chatflow_messages
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM chatflow_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (self.session_id, self.session_id, max_messages),
            )
            conn.execute(
                "UPDATE chatflow_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (self.session_id,),
            )

    def set_active_agent(self, agent: Agent) -> None:
        """Switch current active agent and persist the pointer to SQLite."""
        super().set_active_agent(agent)
        with _get_sqlite_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE chatflow_sessions SET active_agent_name = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (agent.name, self.session_id),
            )

    def clear_history(self) -> None:
        """Reset conversation dialogue history in memory and database."""
        super().clear_history()
        with _get_sqlite_connection(self.db_path) as conn:
            conn.execute("DELETE FROM chatflow_messages WHERE session_id = ?", (self.session_id,))
            conn.execute(
                "UPDATE chatflow_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (self.session_id,),
            )


class SQLiteSessionStore(BaseSessionStore):
    """Persistent SQLite session registry.

    Stores conversation state, dialogue turns, and active agent pointers in a local
    or volume-mounted SQLite database with zero external dependencies.
    """

    def __init__(self, db_path: str | Path = "chatflow.db", default_max_turns: int = 40) -> None:
        self.db_path = str(db_path)
        self.default_max_turns = default_max_turns
        self._agent_resolver: Callable[[str], Agent | None] | None = None
        self._init_db()

    def set_agent_resolver(self, resolver: Callable[[str], Agent | None]) -> None:
        """Register a resolver callback to map agent names from SQLite back to Agent instances."""
        self._agent_resolver = resolver

    def _init_db(self) -> None:
        """Initialize the SQLite tables and indexes."""
        # Ensure directory exists if path contains directories
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

        with _get_sqlite_connection(self.db_path) as conn:
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
                "CREATE INDEX IF NOT EXISTS idx_chatflow_messages_session ON chatflow_messages(session_id, id)"
            )

    def get_or_create(
        self,
        session_id: str,
        default_agent: Agent,
        metadata: dict[str, Any] | None = None,
    ) -> SessionContext:
        """Retrieve existing session from SQLite or insert and initialize a new one."""
        with _get_sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT active_agent_name, metadata_json FROM chatflow_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if row is None:
                meta_json = json.dumps(metadata or {})
                conn.execute(
                    "INSERT INTO chatflow_sessions (session_id, active_agent_name, metadata_json) VALUES (?, ?, ?)",
                    (session_id, default_agent.name, meta_json),
                )
                return SQLiteSessionContext(
                    session_id=session_id,
                    initial_agent=default_agent,
                    db_path=self.db_path,
                    max_turns=self.default_max_turns,
                    metadata=metadata,
                )
            else:
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
                    db_path=self.db_path,
                    max_turns=self.default_max_turns,
                    metadata=meta,
                    preloaded_history=history,
                )

    def get(self, session_id: str) -> SessionContext | None:
        """Fetch session by ID if it exists in SQLite."""
        with _get_sqlite_connection(self.db_path) as conn:
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
        with _get_sqlite_connection(self.db_path) as conn:
            conn.execute("DELETE FROM chatflow_messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM chatflow_sessions WHERE session_id = ?", (session_id,))
            return cursor.rowcount > 0

    def list_sessions(self) -> list[str]:
        """List all session IDs ordered by latest update."""
        with _get_sqlite_connection(self.db_path) as conn:
            rows = conn.execute("SELECT session_id FROM chatflow_sessions ORDER BY updated_at DESC").fetchall()
            return [r["session_id"] for r in rows]

    def clear(self) -> None:
        """Purge all sessions and history from SQLite."""
        with _get_sqlite_connection(self.db_path) as conn:
            conn.execute("DELETE FROM chatflow_messages")
            conn.execute("DELETE FROM chatflow_sessions")
