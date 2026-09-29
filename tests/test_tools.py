"""Tests for chatflow-agent tool introspection and execution engine."""


import pytest

from chatflow_agent.core.tools import Tool, tool
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
    assert "Cannot execute async tool 'async_sample_function' synchronously via execute()" in str(exc_info.value)
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

