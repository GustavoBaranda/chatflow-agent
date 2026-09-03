"""Session memory management and context tracking for multi-agent conversations."""

from typing import Any, Dict, List, Optional

from chatflow_agent.core.agent import Agent
from chatflow_agent.types import Message, Role


class SessionContext:
    """Stores conversation state, dialogue history, and active agent pointer for a session."""

    def __init__(
        self,
        session_id: str,
        initial_agent: Agent,
        max_turns: int = 40,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.active_agent = initial_agent
        self.max_turns = max_turns
        self.history: List[Message] = []
        self.metadata: Dict[str, Any] = metadata or {}

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


class SessionStore:
    """Thread-safe in-memory session registry indexable by session_id."""

    def __init__(self, default_max_turns: int = 40) -> None:
        self._sessions: Dict[str, SessionContext] = {}
        self.default_max_turns = default_max_turns

    def get_or_create(
        self,
        session_id: str,
        default_agent: Agent,
        metadata: Optional[Dict[str, Any]] = None,
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

    def get(self, session_id: str) -> Optional[SessionContext]:
        """Fetch session by ID if it exists."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Remove a session from storage."""
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> List[str]:
        """List all active session identifiers."""
        return list(self._sessions.keys())

    def clear(self) -> None:
        """Delete all stored sessions."""
        self._sessions.clear()
