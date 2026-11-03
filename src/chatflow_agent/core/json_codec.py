"""Shared JSON encoding for tool results and session metadata.

Tool results and session metadata must be JSON-serializable. Known conversions:

- Pydantic models → ``model_dump(mode="json")``
- dataclasses → ``asdict``
- ``datetime`` / ``date`` / ``time`` → ISO-8601 strings

Session metadata persistence is strict: anything else raises :class:`JsonCodecError`
so unsupported objects are not silently stored as strings.

Tool wire encoding may additionally coerce remaining leaf values with ``str(...)``
so providers still receive a JSON string for odd tool return shapes.

Callers that need a hard failure can catch :class:`JsonCodecError` from
``chatflow_agent`` / ``chatflow_agent.exceptions``.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from typing import Any

from pydantic import BaseModel

from chatflow_agent.exceptions import JsonCodecError

__all__ = [
    "JsonCodecError",
    "json_default",
    "json_default_coercing",
    "to_jsonable",
    "dumps_json",
    "encode_tool_content",
]


def _structured_default(value: Any) -> Any | None:
    """Return a structured conversion, or ``None`` when the value is unknown."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return None


def json_default(value: Any) -> Any:
    """Encode known structured types; refuse everything else."""
    converted = _structured_default(value)
    if converted is not None:
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_default_coercing(value: Any) -> Any:
    """Like :func:`json_default`, then fall back to ``str(value)`` for tool wire forms."""
    converted = _structured_default(value)
    if converted is not None:
        return converted
    return str(value)


def to_jsonable(value: Any, *, coerce: bool = True) -> Any:
    """Return a JSON-friendly copy of ``value``.

    When ``coerce`` is true (default), unknown leaves become ``str(...)`` for tool
    results. Failures such as circular references raise :class:`JsonCodecError`.
    """
    default = json_default_coercing if coerce else json_default
    try:
        return json.loads(json.dumps(value, default=default))
    except (TypeError, ValueError, OverflowError) as err:
        raise JsonCodecError("value") from err


def dumps_json(value: Any, *, what: str = "value", coerce: bool = False) -> str:
    """Serialize ``value`` to JSON text, raising :class:`JsonCodecError` on failure.

    Metadata callers keep ``coerce=False`` (strict). Tool wire encoding passes
    ``coerce=True``.
    """
    default = json_default_coercing if coerce else json_default
    try:
        return json.dumps(value, default=default)
    except (TypeError, ValueError, OverflowError) as err:
        raise JsonCodecError(what) from err


def encode_tool_content(content: Any) -> str:
    """Encode a tool result payload for provider wire formats."""
    if isinstance(content, (dict, list)):
        return dumps_json(content, what="Tool result content", coerce=True)
    return str(content)
