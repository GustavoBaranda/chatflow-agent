"""Tool abstraction and introspection engine for chatflow-agent."""

import asyncio
import functools
import inspect
import types
from collections.abc import Callable, Mapping, Sequence
from typing import (
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

    # Unwrap typing.Annotated[T, ...] to the underlying type T
    if origin is not None:
        # Check if this is Annotated (works for typing.Annotated and typing_extensions)
        origin_name = getattr(origin, "__name__", "") or getattr(origin, "__qualname__", "")
        if "Annotated" in origin_name:
            args = get_args(py_type)
            if args:
                return _python_type_to_gemini_type(args[0])

    if origin in (Union, types.UnionType):
        # Handle Optional[T] which is Union[T, NoneType]
        args = [arg for arg in get_args(py_type) if arg is not type(None)]
        if args:
            return _python_type_to_gemini_type(args[0])
        return "STRING"

    return TYPE_MAP.get(origin or py_type, "STRING")


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
        self.is_async = inspect.iscoroutinefunction(func) or (
            callable(func) and inspect.iscoroutinefunction(getattr(func, "__call__", None))  # noqa: B004
        )
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

            prop: dict[str, Any] = {
                "type": _python_type_to_gemini_type(param.annotation),
            }
            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)

            # Prefer annotation over default for type if available
            if param.annotation is not inspect.Parameter.empty:
                prop["type"] = _python_type_to_gemini_type(param.annotation)

            properties[param_name] = prop

        return {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
        }

    def to_gemini_declaration(self) -> dict[str, Any]:
        """Return Gemini-compatible function declaration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously."""
        try:
            return self.func(**kwargs)
        except Exception as e:
            raise ToolExecutionError(f"Tool {self.name} failed: {e}") from e

    async def execute_async(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously."""
        try:
            if self.is_async:
                return await self.func(**kwargs)
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.func(**kwargs))
        except Exception as e:
            raise ToolExecutionError(f"Tool {self.name} failed: {e}") from e


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any] | Tool:
    """Decorator to turn a function into a Tool."""
    def decorator(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description)

    if func is None:
        return decorator
    return decorator(func)
