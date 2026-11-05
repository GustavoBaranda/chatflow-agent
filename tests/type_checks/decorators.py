"""Static type checks for the ``tool()`` and ``Agent.tool()`` decorators (#176).

Checked in CI with ``python -m mypy --strict tests/type_checks/decorators.py``; pytest does
not collect this module. Every supported decorator form must be inferred as ``Tool``. If a
form regresses to ``Any``, the ``type: ignore[attr-defined]`` comments below become unused
and ``--strict`` (``warn_unused_ignores``) reports them, so the check fails.
"""

from collections.abc import Callable

from chatflow_agent import Agent, Tool, tool


@tool
def bare(a: int) -> str:
    return str(a)


@tool()
def empty_call(a: int) -> str:
    return str(a)


@tool("positional_name")
def positional_name(a: int) -> str:
    return str(a)


@tool(name="keyword_name", description="Keyword arguments.", validate_args=False)
def keyword_name(a: int) -> str:
    return str(a)


@tool
async def async_bare(a: int) -> str:
    return str(a)


@tool(name="async_keyword_name")
async def async_keyword_name(a: int) -> str:
    return str(a)


def plain(a: int) -> str:
    return str(a)


by_keyword = tool(name_or_func=plain)
from_lambda = tool(lambda a: a)

agent = Agent(name="Typing", instructions="Static checks only.")


@agent.tool
def agent_bare(a: int) -> str:
    return str(a)


@agent.tool()
def agent_empty_call(a: int) -> str:
    return str(a)


@agent.tool("agent_positional_name")
def agent_positional_name(a: int) -> str:
    return str(a)


@agent.tool(name="agent_keyword_name", validate_args=False)
async def agent_async_keyword_name(a: int) -> str:
    return str(a)


# Every form is a Tool.
tools: list[Tool] = [
    bare,
    empty_call,
    positional_name,
    keyword_name,
    async_bare,
    async_keyword_name,
    by_keyword,
    from_lambda,
    agent_bare,
    agent_empty_call,
    agent_positional_name,
    agent_async_keyword_name,
]
typed_tool: Tool = bare
names: list[str] = [t.name for t in tools]
tool_name: str = agent_async_keyword_name.name

# Existing ways of using and invoking tools keep type-checking.
as_callable: Callable[..., object] = bare
registered = Agent(name="Registered", instructions="x", tools=[bare, agent_bare])
direct_result = bare(a=1)
sync_result = bare.execute(a=1)


async def run_async_tool() -> None:
    await async_bare.execute_async(a=1)


# Negative checks: these accesses are errors only while the decorators return Tool.
_ = bare.not_an_attribute  # type: ignore[attr-defined]
_ = agent_bare.not_an_attribute  # type: ignore[attr-defined]
_ = empty_call.not_an_attribute  # type: ignore[attr-defined]
_ = agent_empty_call.not_an_attribute  # type: ignore[attr-defined]
