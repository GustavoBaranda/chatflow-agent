"""Tool abstraction and introspection engine for chatflow-agent."""

import asyncio
import functools
import inspect
import types
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
)

from chatflow_agent.exceptions import ToolExecutionError

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


def _python_type_to_gemini_type(py_type: Any) -> str:
    """Map a Python type annotation to Gemini schema type string."""
    if py_type is inspect.Parameter.empty or py_type is Any:
        return "STRING"

    origin = get_origin(py_type)
    if origin is Annotated:
        # Annotated[T, ...] maps the same as its underlying type T
        return _python_type_to_gemini_type(get_args(py_type)[0])

    if origin in (Union, types.UnionType):
        # Handle Optional[T] which is Union[T, NoneType]
        args = [arg for arg in get_args(py_type) if arg is not type(None)]
        if args:
            return _python_type_to_gemini_type(args[0])
        return "STRING"

    return TYPE_MAP.get(origin or py_type, "STRING")


def _is_async_callable(func: Any) -> bool:
    """Detect whether calling ``func`` returns a coroutine.

    Handles regular coroutine functions, ``functools.partial`` objects (including
    recursively nested ones) and callable instances that define an ``async
    __call__``. A ``functools.partial`` is unwrapped to the callable it wraps
    before the underlying callable is inspected.
    """
    if isinstance(func, functools.partial):
        underlying = func.func
        # ``partial`` accepts any callable, including other partials.
        if underlying is not func:
            return _is_async_callable(underlying)
        return False

    if inspect.iscoroutinefunction(func):
        return True

    if callable(func):
        # ``partial.__call__`` is a built-in slot wrapper, not the wrapped callable,
        # so only inspect ``__call__`` for genuine callable instances.
        call = getattr(func, "__call__", None)  # noqa: B004
        return call is not None and inspect.iscoroutinefunction(call)

    return False


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
        self.signature = inspect.signature(func)
        self.is_async = _is_async_callable(func)
        self.parameters_schema = self._build_parameters_schema()

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

            gemini_type = _python_type_to_gemini_type(param.annotation)
            param_def: dict[str, Any] = {
                "type": gemini_type,
            }

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
        """Return the function declaration format expected by Google Gemini API."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

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
            # Run sync function in an executor thread to avoid blocking event loop
            return await asyncio.to_thread(self.func, **kwargs)
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
