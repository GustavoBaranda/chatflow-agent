"""Tests for chatflow-agent tool introspection and execution engine."""

import functools
import inspect
import typing
from collections.abc import Mapping, Sequence
from typing import Annotated, Optional

import pytest

from chatflow_agent.core.tools import Tool, _python_type_to_gemini_type, tool
from chatflow_agent.exceptions import ToolExecutionError


def sample_function(query: str, limit: int = 10, verbose: bool = False) -> dict:
    """Search for relevant documents in database.

    Args:
        query: Search term.
        limit: Max results to return.
        verbose: Include debug metrics.
    """
    return {"query": query, "count": limit, "verbose": verbose}


async def async_sample_function(order_id: str, tags: list[str]) -> str:
    """Async process an order with specified tags."""
    return f"Order {order_id} tagged with {len(tags)} tags"


def error_function() -> None:
    """Function that fails intentionally."""
    raise ValueError("Something went wrong inside the tool")


def test_tool_introspection_parameters() -> None:
    t = Tool(sample_function)
    assert t.name == "sample_function"
    assert "Search for relevant documents" in t.description
    assert t.is_async is False

    decl = t.to_gemini_declaration()
    assert decl["name"] == "sample_function"
    params = decl["parameters"]
    assert params["type"] == "OBJECT"
    assert "query" in params["required"]
    assert "limit" not in params["required"]
    assert "verbose" not in params["required"]
    assert params["properties"]["query"]["type"] == "STRING"
    assert params["properties"]["limit"]["type"] == "INTEGER"
    assert params["properties"]["limit"]["default"] == 10
    assert params["properties"]["verbose"]["type"] == "BOOLEAN"
    assert params["properties"]["verbose"]["default"] is False


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Optional[int], "INTEGER"),  # noqa: UP045 - preserve legacy Optional coverage
        (int | None, "INTEGER"),
        (float | None, "NUMBER"),
        (bool | None, "BOOLEAN"),
        (list[str] | None, "ARRAY"),
    ],
)
def test_pep604_union_type_introspection(annotation: object, expected: str) -> None:
    assert _python_type_to_gemini_type(annotation) == expected


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (list, "ARRAY"),
        (list[str], "ARRAY"),
        (dict, "OBJECT"),
        (dict[str, int], "OBJECT"),
        (tuple, "ARRAY"),
        (tuple[int, str], "ARRAY"),
        (tuple[int, ...], "ARRAY"),
        (set, "ARRAY"),
        (set[str], "ARRAY"),
        (frozenset, "ARRAY"),
        (frozenset[str], "ARRAY"),
        (Sequence, "ARRAY"),
        (Sequence[str], "ARRAY"),
        (Mapping, "OBJECT"),
        (Mapping[str, int], "OBJECT"),
        (typing.Tuple, "ARRAY"),  # noqa: UP006 - preserve legacy alias coverage
        (typing.Tuple[int, ...], "ARRAY"),  # noqa: UP006 - preserve legacy alias coverage
        (typing.Set, "ARRAY"),  # noqa: UP006 - preserve legacy alias coverage
        (typing.Set[str], "ARRAY"),  # noqa: UP006 - preserve legacy alias coverage
        (typing.FrozenSet, "ARRAY"),  # noqa: UP006 - preserve legacy alias coverage
        (typing.FrozenSet[str], "ARRAY"),  # noqa: UP006 - preserve legacy alias coverage
        (typing.Sequence, "ARRAY"),
        (typing.Sequence[str], "ARRAY"),
        (typing.Mapping, "OBJECT"),
        (typing.Mapping[str, int], "OBJECT"),
        (tuple[int, ...] | None, "ARRAY"),
        (set[str] | None, "ARRAY"),
        (frozenset[str] | None, "ARRAY"),
        (Sequence[str] | None, "ARRAY"),
        (Mapping[str, int] | None, "OBJECT"),
        (Optional[Sequence[str]], "ARRAY"),  # noqa: UP045 - preserve legacy Optional coverage
        (Optional[Mapping[str, int]], "OBJECT"),  # noqa: UP045 - preserve legacy Optional coverage
        (Optional[frozenset[str]], "ARRAY"),  # noqa: UP045 - preserve legacy Optional coverage
        (str, "STRING"),
        (bytes, "STRING"),
        (typing.Any, "STRING"),
    ],
)
def test_container_type_introspection(annotation: object, expected: str) -> None:
    assert _python_type_to_gemini_type(annotation) == expected


def test_tool_container_parameters_schema() -> None:
    def search(
        coordinates: tuple[float, float],
        tags: set[str],
        excluded_tags: frozenset[str],
        queries: Sequence[str],
        filters: Mapping[str, int] | None = None,
    ) -> None:
        pass

    schema = Tool(search).to_gemini_declaration()["parameters"]
    assert schema["properties"] == {
        "coordinates": {"type": "ARRAY"},
        "tags": {"type": "ARRAY"},
        "excluded_tags": {"type": "ARRAY"},
        "queries": {"type": "ARRAY"},
        "filters": {"type": "OBJECT", "default": None},
    }
    assert schema["required"] == ["coordinates", "tags", "excluded_tags", "queries"]


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Annotated[int, "positive"], "INTEGER"),
        (Annotated[bool, "flag"], "BOOLEAN"),
        (Annotated[str, "name"], "STRING"),
        (Annotated[float, "ratio"], "NUMBER"),
        (Annotated[list[str], "tags"], "ARRAY"),
        (Annotated[dict[str, int], "mapping"], "OBJECT"),
        (Optional[Annotated[int, "positive"]], "INTEGER"),  # noqa: UP045 - preserve legacy Optional coverage
        (Annotated[int, "positive"] | None, "INTEGER"),
        (Annotated[int | None, "positive"], "INTEGER"),
        (Annotated[Annotated[int, "inner"], "outer"], "INTEGER"),
    ],
)
def test_annotated_type_introspection(annotation: object, expected: str) -> None:
    assert _python_type_to_gemini_type(annotation) == expected


def test_tool_annotated_parameters_schema() -> None:
    def annotated(
        count: Annotated[int, "positive"],
        enabled: Annotated[bool, "flag"],
        tags: Annotated[list[str], "tags"],
        limit: Annotated[int, "limit"] | None = None,
        verbose: Annotated[bool, "flag"] = False,
    ) -> None:
        pass

    schema = Tool(annotated).parameters_schema
    assert schema["properties"] == {
        "count": {"type": "INTEGER"},
        "enabled": {"type": "BOOLEAN"},
        "tags": {"type": "ARRAY"},
        "limit": {"type": "INTEGER", "default": None},
        "verbose": {"type": "BOOLEAN", "default": False},
    }
    assert schema["required"] == ["count", "enabled", "tags"]


def test_tool_async_introspection() -> None:
    t = Tool(async_sample_function)
    assert t.name == "async_sample_function"
    assert t.is_async is True
    decl = t.to_gemini_declaration()
    assert decl["parameters"]["properties"]["tags"]["type"] == "ARRAY"
    assert "order_id" in decl["parameters"]["required"]
    assert "tags" in decl["parameters"]["required"]


def test_tool_execution_sync() -> None:
    t = Tool(sample_function)
    result = t.execute(query="invoice #42", limit=5)
    assert result == {"query": "invoice #42", "count": 5, "verbose": False}


@pytest.mark.asyncio
async def test_tool_execution_async() -> None:
    t = Tool(async_sample_function)
    result = await t.execute_async(order_id="ORD-101", tags=["urgent", "vip"])
    assert result == "Order ORD-101 tagged with 2 tags"


def test_tool_execution_error_handling() -> None:
    t = Tool(error_function)
    with pytest.raises(ToolExecutionError) as exc_info:
        t.execute()
    assert exc_info.value.tool_name == "error_function"
    assert "Something went wrong" in str(exc_info.value)


def test_tool_decorator_bare() -> None:
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    assert isinstance(add_numbers, Tool)
    assert add_numbers.name == "add_numbers"
    assert add_numbers.execute(a=3, b=7) == 10


def test_tool_decorator_with_custom_name_and_desc() -> None:
    @tool(name="calc_sum", description="Custom math addition")
    def add_numbers(a: int, b: int) -> int:
        return a + b

    assert isinstance(add_numbers, Tool)
    assert add_numbers.name == "calc_sum"
    assert add_numbers.description == "Custom math addition"
    assert add_numbers.execute(a=10, b=20) == 30


@pytest.mark.asyncio
async def test_async_tool_sync_execute_in_running_loop_raises_clear_error() -> None:
    """Calling execute() on an async tool from an active event loop must raise a clear ToolExecutionError (SEC-09)."""
    t = Tool(async_sample_function)
    with pytest.raises(ToolExecutionError) as exc_info:
        t.execute(order_id="ORD-999", tags=["test"])

    assert exc_info.value.tool_name == "async_sample_function"
    assert "Cannot execute async tool 'async_sample_function' synchronously via execute()" in str(
        exc_info.value
    )
    assert "Use 'await tool.execute_async(**kwargs)' instead." in str(exc_info.value)


def test_async_tool_sync_execute_outside_loop_succeeds() -> None:
    """Calling execute() on an async tool when no event loop is running succeeds via asyncio.run()."""
    t = Tool(async_sample_function)
    result = t.execute(order_id="ORD-100", tags=["sync_context"])
    assert result == "Order ORD-100 tagged with 1 tags"


@pytest.mark.asyncio
async def test_sync_tool_execute_async_succeeds() -> None:
    """Calling execute_async() on a sync tool dispatches to thread and returns successfully."""
    t = Tool(sample_function)
    result = await t.execute_async(query="async_thread_test", limit=3)
    assert result == {"query": "async_thread_test", "count": 3, "verbose": False}


def test_variadic_parameters_are_not_exposed() -> None:
    def search(query: str, *args: object, limit: int = 10, **kwargs: object) -> str:
        return query

    schema = Tool(search).parameters_schema
    assert set(schema["properties"]) == {"query", "limit"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"]["default"] == 10


def test_variadic_only_tool_has_no_declared_arguments() -> None:
    def flexible(*args: object, **kwargs: object) -> str:
        return "ok"

    schema = Tool(flexible).parameters_schema
    assert schema["properties"] == {}
    assert schema["required"] == []


def test_partial_function_tool() -> None:
    """Tool works with functools.partial instances that lack __name__."""
    import functools

    def greet(greeting: str, name: str) -> str:
        """Send a greeting to a user."""
        return f"{greeting}, {name}!"

    say_hello = functools.partial(greet, "Hello")
    t = Tool(say_hello)
    assert t.name == "greet"
    assert "Send a greeting" in t.description
    assert t.is_async is False
    assert "name" in t.parameters_schema["required"]
    assert "greeting" not in t.parameters_schema["properties"]
    assert t.execute(name="Alice") == "Hello, Alice!"


def test_callable_instance_tool() -> None:
    """Tool works with class instances implementing __call__ without __name__."""

    class Greeter:
        """A class that greets people."""

        def __call__(self, name: str) -> str:
            return f"Welcome, {name}!"

    greeter = Greeter()
    t = Tool(greeter)
    assert t.name == "Greeter"
    assert "A class that greets people" in t.description
    assert t.is_async is False
    assert t.execute(name="Bob") == "Welcome, Bob!"


@pytest.mark.asyncio
async def test_async_callable_instance_tool() -> None:
    """Tool detects async on callable class instances."""

    class AsyncMultiplier:
        """Multiplies asynchronously."""

        async def __call__(self, x: int) -> int:
            return x * 3

    m = AsyncMultiplier()
    t = Tool(m)
    assert t.name == "AsyncMultiplier"
    assert t.is_async is True
    res = await t.execute_async(x=5)
    assert res == 15


def test_partial_async_function_tool() -> None:
    """A partial wrapping a plain async function is still detected as async."""
    import functools

    async def fetch(item_id: str, limit: int = 5) -> str:
        """Fetch an item asynchronously."""
        return f"{item_id}:{limit}"

    bound = functools.partial(fetch, "item-1")
    t = Tool(bound)
    assert t.is_async is True
    assert "limit" in t.parameters_schema["properties"]
    assert "item_id" not in t.parameters_schema["properties"]


def test_partial_sync_function_tool() -> None:
    """A partial wrapping a sync function stays sync."""
    import functools

    def fetch(item_id: str, limit: int = 5) -> str:
        """Fetch an item synchronously."""
        return f"{item_id}:{limit}"

    bound = functools.partial(fetch, "item-1")
    t = Tool(bound)
    assert t.is_async is False
    assert t.execute(limit=2) == "item-1:2"


@pytest.mark.asyncio
async def test_partial_wrapping_async_callable_instance_is_awaited() -> None:
    """Regression for #65: partial(async callable instance) must be awaited, not returned raw."""

    class AsyncDoubler:
        """Doubles values asynchronously."""

        async def __call__(self, value: int, offset: int = 0) -> int:
            return value * 2 + offset

    wrapped = functools.partial(AsyncDoubler(), offset=1)
    t = Tool(wrapped)
    assert t.is_async is True
    # The bound keyword argument keeps its default in the schema.
    assert set(t.parameters_schema["properties"]) == {"value", "offset"}
    assert t.parameters_schema["properties"]["offset"]["default"] == 1

    result = await t.execute_async(value=3)
    assert not inspect.iscoroutine(result)
    assert result == 7


@pytest.mark.asyncio
async def test_partial_positional_binding_of_async_callable_instance() -> None:
    """Positionally bound arguments of a partial-wrapped callable instance stay out of the schema."""

    class AsyncFormula:
        """Computes a formula asynchronously."""

        async def __call__(self, base: int, factor: int, offset: int = 0) -> int:
            return base * factor + offset

    wrapped = functools.partial(AsyncFormula(), 4, 3)
    t = Tool(wrapped)
    assert t.is_async is True
    assert set(t.parameters_schema["properties"]) == {"offset"}
    assert t.parameters_schema["required"] == []

    result = await t.execute_async(offset=2)
    assert result == 14


@pytest.mark.asyncio
async def test_partial_wrapping_sync_callable_instance_stays_sync() -> None:
    """A partial wrapping a sync callable instance must remain synchronous."""

    class SyncDoubler:
        """Doubles values synchronously."""

        def __call__(self, value: int, offset: int = 0) -> int:
            return value * 2 + offset

    wrapped = functools.partial(SyncDoubler(), offset=1)
    t = Tool(wrapped)
    assert t.is_async is False
    assert set(t.parameters_schema["properties"]) == {"value", "offset"}

    result = await t.execute_async(value=3)
    assert result == 7


@pytest.mark.asyncio
async def test_nested_partial_wrapping_async_callable_instance() -> None:
    """Nested partials around an async callable instance are unwrapped recursively."""

    class AsyncScaler:
        """Scales values asynchronously."""

        async def __call__(self, value: int) -> int:
            return value * 5

    wrapped = functools.partial(functools.partial(AsyncScaler()))
    t = Tool(wrapped)
    assert t.is_async is True
    result = await t.execute_async(value=4)
    assert not inspect.iscoroutine(result)
    assert result == 20

