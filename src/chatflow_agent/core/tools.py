"""Tool abstraction and introspection engine for chatflow-agent."""

import asyncio
import inspect
from collections.abc import Callable
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
    dict: "OBJECT",
}


def _python_type_to_gemini_type(py_type: Any) -> str:
    """Map a Python type annotation to Gemini schema type string."""
    if py_type is inspect.Parameter.empty or py_type is Any:
        return "STRING"

    origin = get_origin(py_type)
    if origin is Union:
        # Handle Optional[T] which is Union[T, NoneType]
        args = [arg for arg in get_args(py_type) if arg is not type(None)]
        if args:
            return _python_type_to_gemini_type(args[0])
        return "STRING"

    if origin in (list, list):
        return "ARRAY"
    if origin in (dict, dict):
        return "OBJECT"

    return TYPE_MAP.get(py_type, "STRING")


class Tool:
    """Encapsulates a callable tool with introspection for LLM function calling."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").strip() or f"Execute {self.name}"
        self.signature = inspect.signature(func)
        self.is_async = inspect.iscoroutinefunction(func)
        self.parameters_schema = self._build_parameters_schema()

    def _build_parameters_schema(self) -> dict[str, Any]:
        """Introspect function signature and generate schema for parameters."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in self.signature.parameters.items():
            if param_name in ("self", "cls"):
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
        """Execute the tool asynchronously with error handling."""
        try:
            if self.is_async:
                return await self.func(**kwargs)
            # Run sync function in an executor thread to avoid blocking event loop
            return await asyncio.to_thread(self.func, **kwargs)
        except Exception as err:
            raise ToolExecutionError(tool_name=self.name, original_error=err) from err

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously."""
        try:
            if self.is_async:
                # If running within an already active event loop, run via task
                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(self.func(**kwargs))
                except RuntimeError:
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
