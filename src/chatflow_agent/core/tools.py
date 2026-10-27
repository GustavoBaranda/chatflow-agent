"""Tool abstraction and introspection engine for chatflow-agent."""

import asyncio
import functools
import inspect
import logging
import types
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
)

from chatflow_agent.core._sync_work import _run_sync_tool
from chatflow_agent.exceptions import ToolExecutionError, ToolSchemaError

logger = logging.getLogger("chatflow_agent.tools")

# Mapping Python types to Gemini / JSON Schema types
TYPE_MAP = {
    str: "STRING",
    int: "INTEGER",
    float: "NUMBER",
    bool: "BOOLEAN",
    list: "ARRAY",
    tuple: "ARRAY",
    set: "ARRAY",
    frozenset: "ARRAY",
    Sequence: "ARRAY",
    dict: "OBJECT",
    Mapping: "OBJECT",
}


def _array_items_schema(container: Any, args: tuple[Any, ...]) -> dict[str, Any] | None:
    """Build the ``items`` schema for an array-like annotation.

    ``container`` is the collection type (``list``, ``tuple``, ``set``, ...) and ``args`` its
    type arguments. Returns ``None`` when the element types cannot be expressed as a single
    ``items`` schema (heterogeneous tuples).
    """
    if container is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            # Variable-length homogeneous tuple: tuple[T, ...]
            return _python_type_to_schema(args[0])
        if args == ((),):
            # tuple[()] is reported as ((),) on Python 3.10 and as () on 3.11+
            args = ()
        element_schemas = [_python_type_to_schema(arg) for arg in args]
        if not element_schemas:
            return {"type": "STRING"}
        if all(schema == element_schemas[0] for schema in element_schemas):
            # Fixed-length homogeneous tuple, e.g. tuple[float, float]
            return element_schemas[0]
        logger.warning(
            "Tuple annotation %r has heterogeneous element types, which cannot be expressed "
            "as a single array 'items' schema; omitting 'items' for this parameter.",
            container[args],
        )
        return None

    if len(args) == 1:
        return _python_type_to_schema(args[0])

    # Unparameterized collection (e.g. bare ``list``): fall back to STRING items, consistent
    # with the STRING fallback used for unannotated parameters.
    return {"type": "STRING"}


def _is_expressible_type(py_type: Any) -> bool:
    """Whether ``py_type`` maps to a concrete schema type rather than the STRING fallback."""
    if py_type is inspect.Parameter.empty or py_type is Any:
        # Explicit fallbacks: unannotated parameters are STRING on purpose.
        return True
    origin = get_origin(py_type)
    while origin is Annotated:
        py_type = get_args(py_type)[0]
        origin = get_origin(py_type)
    if origin in (Union, types.UnionType):
        return all(
            _is_expressible_type(arg)
            for arg in get_args(py_type)
            if arg is not type(None)
        )
    container = origin or py_type
    return container in TYPE_MAP


def _union_to_schema(py_type: Any) -> dict[str, Any]:
    """Build a schema for a ``Union`` / ``X | Y`` annotation.

    * A single non-``None`` member (``Optional[T]`` / ``T | None``) maps to that
      member's schema plus a ``nullable: true`` flag instead of silently losing
      nullability.
    * Unions combining ``int`` and ``float`` (in either order) broaden to
      ``NUMBER``, because IEEE 754 numbers subsume integers in JSON Schema and
      OpenAPI; annotation order must not decide the result.
    * Unions of distinct concrete types keep every member in an ``anyOf`` list.
    * A union member that has no concrete schema mapping cannot be expressed
      validly for any engine, so it fails fast with a descriptive error
      instead of silently emitting a schema the provider would reject.
    """
    members = [arg for arg in get_args(py_type) if arg is not type(None)]
    nullable = type(None) in get_args(py_type)

    if not members:
        return {"type": "STRING"}

    if len(members) == 1:
        if not _is_expressible_type(members[0]):
            raise ToolSchemaError(
                f"annotation {py_type!r}: union member {members[0]!r} has no concrete "
                "schema type mapping, so the union cannot be expressed as a valid "
                "function-calling schema; use supported primitives or collections "
                "(str, int, float, bool, list, tuple, set, frozenset, dict) instead"
            )
        schema = _python_type_to_schema(members[0])
        if nullable:
            schema = dict(schema)
            schema["nullable"] = True
        return schema

    for member in members:
        if not _is_expressible_type(member):
            raise ToolSchemaError(
                f"annotation {py_type!r}: union member {member!r} has no concrete "
                "schema type mapping, so the union cannot be expressed as a valid "
                "function-calling schema; use supported primitives or collections "
                "(str, int, float, bool, list, tuple, set, frozenset, dict) instead"
            )

    if set(members) == {int, float}:
        # A JSON 'number' accepts integers, so int|float and float|int both widen to NUMBER.
        schema = {"type": "NUMBER"}
        if nullable:
            schema["nullable"] = True
        return schema

    schema = {
        "anyOf": [_python_type_to_schema(member) for member in members]
    }
    if nullable:
        schema["nullable"] = True
    return schema


def _python_type_to_schema(py_type: Any) -> dict[str, Any]:
    """Map a Python type annotation to a Gemini schema fragment.

    Array-like types (``list``, ``tuple``, ``set``, ``frozenset``, ``Sequence``) include an
    ``items`` schema derived from their type arguments, recursively for nested collections:

    * ``list[str]`` -> ``{"type": "ARRAY", "items": {"type": "STRING"}}``
    * ``tuple[int, ...]`` and homogeneous tuples such as ``tuple[float, float]`` use the
      shared element type.
    * Unparameterized collections (``list``) fall back to ``{"type": "STRING"}`` items.
    * Heterogeneous tuples (``tuple[int, str]``) cannot be expressed as one ``items`` schema,
      so ``items`` is omitted and a warning is logged.

    ``Annotated`` metadata is ignored and ``Optional``/union annotations use their first
    non-``None`` member, as for non-array types.
    """
    if py_type is inspect.Parameter.empty or py_type is Any:
        return {"type": "STRING"}

    origin = get_origin(py_type)
    if origin is Annotated:
        # Annotated[T, ...] maps the same as its underlying type T
        return _python_type_to_schema(get_args(py_type)[0])

    if origin in (Union, types.UnionType):
        return _union_to_schema(py_type)

    container = origin or py_type
    schema_type = TYPE_MAP.get(container, "STRING")
    schema: dict[str, Any] = {"type": schema_type}
    if schema_type == "ARRAY":
        items = _array_items_schema(container, get_args(py_type))
        if items is not None:
            schema["items"] = items
    return schema


def _python_type_to_gemini_type(py_type: Any) -> str:
    """Map a Python type annotation to Gemini schema type string.

    Multi-type unions have more than one valid type; the first member's is
    reported for this introspection helper.
    """
    schema = _python_type_to_schema(py_type)
    if "anyOf" in schema:
        return str(schema["anyOf"][0]["type"])
    return str(schema["type"])


_JSON_TYPE_CASE = {
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
}


def _to_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical (Gemini-style) schema fragment to JSON Schema.

    Types are lowercased, ``nullable: true`` becomes ``anyOf`` including a
    ``{"type": "null"}`` member (or a ``[type, "null"]`` list for single
    types), ``anyOf`` members are converted recursively, and defaults and the
    required list are preserved untouched.
    """
    result: dict[str, Any] = {}
    source = dict(schema)
    nullable = bool(source.pop("nullable", False))

    for key, value in source.items():
        if key == "type" and isinstance(value, str):
            result["type"] = _JSON_TYPE_CASE.get(value.upper(), value.lower())
        elif key == "type" and isinstance(value, list):
            result["type"] = [
                _JSON_TYPE_CASE.get(str(item).upper(), str(item).lower()) if isinstance(item, str) else item
                for item in value
            ]
        elif key == "items":
            result[key] = _to_json_schema(value)
        elif key == "properties" and isinstance(value, dict):
            result[key] = {name: _to_json_schema(prop) for name, prop in value.items()}
        elif key == "anyOf":
            result[key] = [_to_json_schema(member) for member in value]
        else:
            result[key] = value

    if nullable:
        if "anyOf" in result:
            members = list(result["anyOf"])
            if {"type": "null"} not in members:
                members.append({"type": "null"})
            result["anyOf"] = members
        elif "type" in result:
            declared = result["type"]
            types = declared if isinstance(declared, list) else [declared]
            if "null" not in types:
                result["type"] = [*types, "null"]

    return result


def _resolve_signature(func: Callable[..., Any], tool_name: str) -> inspect.Signature:
    """Return the signature of ``func`` with postponed (string) annotations resolved.

    Modules using ``from __future__ import annotations`` (or explicit string annotations)
    store annotations as strings. ``eval_str=True`` evaluates them against the namespace of
    the underlying callable, which covers functions, bound methods, callable instances and
    ``functools.partial`` objects, while leaving ``Annotated`` metadata intact.

    If an annotation cannot be resolved (for example a name imported only under
    ``TYPE_CHECKING``), the unresolved signature is used instead and a warning is logged.
    Parameters then fall back to the default ``STRING`` schema type.
    """
    try:
        return inspect.signature(func, eval_str=True)
    except (NameError, AttributeError, SyntaxError, TypeError) as err:
        logger.warning(
            "Could not resolve type annotations for tool '%s' (%s: %s); "
            "falling back to unresolved annotations, so parameter types default to STRING.",
            tool_name,
            type(err).__name__,
            err,
        )
        return inspect.signature(func)


class Tool:
    """Encapsulates a callable tool with introspection for LLM function calling."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.func = func
        self.name = name or self._extract_name(func)
        self.description = (
            description or self._extract_description(func) or f"Execute {self.name}"
        )
        self.signature = _resolve_signature(func, self.name)
        self.is_async = self._detect_async(func)
        self.parameters_schema = self._build_parameters_schema()

    @staticmethod
    def _detect_async(func: Callable[..., Any]) -> bool:
        """Detect whether a callable returns a coroutine when invoked.

        Handles plain async functions, ``functools.partial`` wrappers (including
        nested ones), and callable instances that implement ``async __call__``.
        Plain ``partial`` wrappers are unwrapped so that an async callable instance
        wrapped in ``partial`` is still classified as async. A ``partial``
        subclass that overrides ``__call__`` is classified by its own
        ``__call__`` instead of the wrapped ``.func``.
        """
        while isinstance(func, functools.partial):
            # A partial subclass that overrides __call__ defines its own
            # calling behaviour; that takes precedence over the wrapped .func.
            if type(func).__call__ is not functools.partial.__call__:
                return inspect.iscoroutinefunction(type(func).__call__)
            func = func.func
        if inspect.iscoroutinefunction(func):
            return True
        if callable(func) and inspect.iscoroutinefunction(
            getattr(func, "__call__", None)  # noqa: B004
        ):
            return True
        return False

    @staticmethod
    def _extract_name(func: Callable[..., Any]) -> str:
        """Extract a readable name from a function, partial, or callable instance."""
        name = getattr(func, "__name__", None)
        if isinstance(name, str):
            return name
        underlying = getattr(func, "func", None)
        underlying_name = getattr(underlying, "__name__", None)
        if isinstance(underlying_name, str):
            return underlying_name
        return type(func).__name__

    @staticmethod
    def _extract_description(func: Callable[..., Any]) -> str:
        """Extract docstring from a function, partial, or callable instance."""
        doc = inspect.getdoc(func)
        if isinstance(func, functools.partial) and hasattr(func, "func"):
            underlying_doc = inspect.getdoc(func.func)
            if underlying_doc:
                return underlying_doc.strip()
        return (doc or "").strip()

    def _build_parameters_schema(self) -> dict[str, Any]:
        """Introspect function signature and generate schema for parameters."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in self.signature.parameters.items():
            if param_name in ("self", "cls") or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            try:
                param_def = _python_type_to_schema(param.annotation)
            except ToolSchemaError as err:
                raise ToolSchemaError(f"tool '{self.name}': parameter '{param_name}': {err}") from err

            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                param_def["default"] = param.default

            properties[param_name] = param_def

        return {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
        }

    def to_gemini_declaration(self) -> dict[str, Any]:
        """Return the function declaration format expected by Google Gemini API.

        The canonical schema already uses OpenAPI 3.0 conventions (uppercase
        type names, ``nullable: true``, ``anyOf`` for multi-type unions).
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    def to_openai_parameters(self) -> dict[str, Any]:
        """Return JSON-Schema parameters for OpenAI-compatible and Anthropic engines.

        JSON Schema (used by OpenAI and Anthropic tool ``input_schema``) has no
        ``nullable`` keyword and no uppercase types: types are lowercased and
        nullability becomes an ``anyOf`` member of ``{"type": "null"}`` (or a
        ``[type, "null"]`` list for single types). The canonical schema is not
        mutated.
        """
        return _to_json_schema(self.parameters_schema)

    async def execute_async(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously with error handling.

        This is the recommended execution method in all asynchronous contexts
        (e.g., within Runners, FastAPI endpoints, Telegram handlers, or active asyncio tasks).
        If the underlying tool is a synchronous function, it is dispatched to a worker thread
        via ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        try:
            if self.is_async:
                return await self.func(**kwargs)
            # Run sync function in an executor thread to avoid blocking event loop. Inside
            # a Runner turn its real lifetime is tracked even if this await is cancelled.
            return await _run_sync_tool(self.func, kwargs)
        except Exception as err:
            raise ToolExecutionError(tool_name=self.name, original_error=err) from err

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously.

        Note:
            If this tool is asynchronous (defined with ``async def``), calling
            ``execute()`` from within an already-running event loop (e.g. inside
            FastAPI, Telegram handlers, or async background tasks) cannot be safely
            delegated synchronously because Python event loops cannot be re-entered.
            In asynchronous environments, always use ``await tool.execute_async(**kwargs)``
            instead.
        """
        try:
            if self.is_async:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    raise RuntimeError(
                        f"Cannot execute async tool '{self.name}' synchronously via execute() "
                        "from within an already-running event loop. "
                        f"Use 'await tool.execute_async(**kwargs)' instead."
                    )
                return asyncio.run(self.func(**kwargs))
            return self.func(**kwargs)
        except Exception as err:
            raise ToolExecutionError(tool_name=self.name, original_error=err) from err

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r} async={self.is_async}>"


def tool(
    name_or_func: str | Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator to register a function as an introspected Tool.

    Usage:
        @tool
        def my_func(a: int) -> str: ...

        @tool(name="custom_name", description="Custom description")
        def my_func(a: int) -> str: ...
    """
    if callable(name_or_func):
        # Bare decorator: @tool
        return Tool(func=name_or_func)

    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or (name_or_func if isinstance(name_or_func, str) else None)
        return Tool(func=fn, name=tool_name, description=description)

    return decorator
