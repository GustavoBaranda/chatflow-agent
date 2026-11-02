"""Shared JSON encoding for tool results and session metadata.

Tool results and session metadata must be JSON-serializable. Values that are not
natively JSON-encodable are coerced the same way across engines and stores:

- Pydantic models → ``model_dump(mode="json")``
- dataclasses → ``asdict``
- ``datetime`` / ``date`` / ``time`` → ISO-8601 strings
- anything else → ``str(value)``

Callers that need a hard failure (e.g. documenting a contract violation to the
user) can catch :class:`JsonCodecError`.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from typing import Any

from pydantic import BaseModel

from chatflow_agent.exceptions import ChatFlowError


class JsonCodecError(ChatFlowError):
    """Raised when a value cannot be prepared for JSON encoding."""

    def __init__(self, what: str) -> None:
        super().__init__(
            f"{what} is not JSON-serializable and could not be coerced. "
            "Use JSON-native types, or datetime/date/time (encoded as ISO-8601)."
        )
        self.what = what


def json_default(value: Any) -> Any:
    """Encode what ``json`` cannot: structurally where possible, else as text."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


def to_jsonable(value: Any) -> Any:
    """Return a JSON-friendly copy of ``value`` using :func:`json_default`."""
    return json.loads(json.dumps(value, default=json_default))


def dumps_json(value: Any, *, what: str = "value") -> str:
    """Serialize ``value`` to JSON text, raising :class:`JsonCodecError` on failure."""
    try:
        return json.dumps(value, default=json_default)
    except (TypeError, ValueError, OverflowError) as err:
        raise JsonCodecError(what) from err


def encode_tool_content(content: Any) -> str:
    """Encode a tool result payload for provider wire formats."""
    if isinstance(content, (dict, list)):
        return dumps_json(content, what="Tool result content")
    return str(content)
