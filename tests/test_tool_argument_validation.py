"""Comprehensive tests for LLM runtime tool argument validation."""

import inspect
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict, get_args, get_origin

import pytest
from pydantic import BaseModel, Field

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner
from chatflow_agent.core.tools import Tool, _sanitize_annotation_for_validation, tool
from chatflow_agent.exceptions import (
    ToolArgumentValidationError,
    ToolExecutionError,
)
from chatflow_agent.types import Message, Role, ToolCall, ToolResult


def test_exception_hierarchy() -> None:
    """ToolArgumentValidationError inherits from ToolExecutionError."""
    assert issubclass(ToolArgumentValidationError, ToolExecutionError)
    err = ToolArgumentValidationError(
        tool_name="test_tool",
        message="parameter 'a': Input should be a valid integer",
    )
    assert err.tool_name == "test_tool"
    assert err.message == "parameter 'a': Input should be a valid integer"
    assert "test_tool" in str(err)
    assert isinstance(err.original_error, Exception)


def test_boolean_argument_coercion_and_rejection() -> None:
    """Boolean arguments are safely coerced from strings/ints or rejected when invalid."""

    @tool
    def confirm_action(proceed: bool = False) -> str:
        return "executed" if proceed else "cancelled"

    # Coercion of truthy representations
    assert confirm_action.execute(proceed="true") == "executed"
    assert confirm_action.execute(proceed="True") == "executed"
    assert confirm_action.execute(proceed=1) == "executed"
    assert confirm_action.execute(proceed="1") == "executed"
    assert confirm_action.execute(proceed=True) == "executed"

    # Coercion of falsy representations
    assert confirm_action.execute(proceed="false") == "cancelled"
    assert confirm_action.execute(proceed="False") == "cancelled"
    assert confirm_action.execute(proceed=0) == "cancelled"
    assert confirm_action.execute(proceed="0") == "cancelled"
    assert confirm_action.execute(proceed=False) == "cancelled"

    # Rejection of invalid booleans
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        confirm_action.execute(proceed="not_a_bool")
    assert "confirm_action" in exc_info.value.tool_name
    assert "parameter 'proceed'" in exc_info.value.message

    with pytest.raises(ToolArgumentValidationError):
        confirm_action.execute(proceed=[])


def test_numeric_argument_coercion_and_rejection() -> None:
    """Numeric arguments are coerced or rejected."""

    @tool
    def calculate_total(quantity: int, unit_price: float) -> float:
        return quantity * unit_price

    # Coerced strings
    res = calculate_total.execute(quantity="5", unit_price="12.50")
    assert res == 62.5
    assert isinstance(res, float)

    # Scientific notation for float
    res2 = calculate_total.execute(quantity=2, unit_price="1e2")
    assert res2 == 200.0

    # Non-integer string for int parameter fails
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        calculate_total.execute(quantity="12.5", unit_price="10.0")
    assert "parameter 'quantity'" in exc_info.value.message

    # Non-numeric string fails
    with pytest.raises(ToolArgumentValidationError) as exc_info2:
        calculate_total.execute(quantity="five", unit_price="10.0")
    assert "parameter 'quantity'" in exc_info2.value.message


def test_collection_arguments_validation() -> None:
    """Lists, sets, tuples, and dicts are validated."""

    @tool
    def process_data(
        tags: list[str],
        scores: set[int],
        coords: tuple[float, float],
        meta: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "tags": tags,
            "scores": scores,
            "coords": coords,
            "meta": meta,
        }

    valid_res = process_data.execute(
        tags=["python", "agent"],
        scores=[10, "20", 30],
        coords=["1.5", "2.5"],
        meta={"level": "1"},
    )
    assert valid_res["tags"] == ["python", "agent"]
    assert valid_res["scores"] == {10, 20, 30}
    assert valid_res["coords"] == (1.5, 2.5)
    assert valid_res["meta"] == {"level": 1}

    # Reject string passed for list
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        process_data.execute(
            tags="not_a_list",
            scores=[1],
            coords=[1.0, 2.0],
            meta={},
        )
    assert "parameter 'tags'" in exc_info.value.message


def test_required_and_optional_parameters() -> None:
    """Missing required arguments raise validation error; defaults are respected."""

    @tool
    def greet(name: str, greeting: str = "Hello", count: int = 1) -> str:
        return f"{greeting}, {name}! " * count

    # Valid with defaults
    assert greet.execute(name="World").strip() == "Hello, World!"

    # Valid overriding defaults
    assert greet.execute(name="World", greeting="Hi", count="2").strip() == "Hi, World! Hi, World!"

    # Missing required 'name'
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        greet.execute(greeting="Hi")
    assert "parameter 'name'" in exc_info.value.message
    assert "required" in exc_info.value.message.lower()


def test_nullability_and_optional_types() -> None:
    """Optional types (Union with None) allow None; non-nullable types reject None."""

    @tool
    def search(query: str, limit: int | None = None, required_nullable: int | None = None) -> dict[str, Any]:
        return {"query": query, "limit": limit, "required_nullable": required_nullable}

    # Explicit None for nullable fields
    res = search.execute(query="test", limit=None, required_nullable=None)
    assert res == {"query": "test", "limit": None, "required_nullable": None}

    # None passed to non-nullable query raises validation error
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        search.execute(query=None, limit=5, required_nullable=1)
    assert "parameter 'query'" in exc_info.value.message


def test_extra_arguments_forbidden_when_no_var_kw() -> None:
    """When a tool function does not accept **kwargs, extra arguments are rejected."""

    @tool
    def strict_tool(a: int, b: str) -> str:
        return f"{a}:{b}"

    # Valid
    assert strict_tool.execute(a=1, b="two") == "1:two"

    # Extra argument rejected with safe label without echoing untrusted key
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        strict_tool.execute(a=1, b="two", unexpected_param="extra")
    assert "unexpected parameter is not permitted" in exc_info.value.message
    assert "unexpected_param" not in exc_info.value.message


def test_extra_arguments_allowed_when_has_var_kw() -> None:
    """When a tool function accepts **kwargs, extra arguments pass through."""

    @tool
    def flexible_tool(a: int, **kwargs: Any) -> dict[str, Any]:
        return {"a": a, "extra": kwargs}

    res = flexible_tool.execute(a="10", custom="value", other=123)
    assert res == {"a": 10, "extra": {"custom": "value", "other": 123}}


def test_no_arguments_tool() -> None:
    """Tool with no parameters validates correctly."""

    @tool
    def no_args() -> str:
        return "success"

    assert no_args.execute() == "success"

    with pytest.raises(ToolArgumentValidationError):
        no_args.execute(unexpected="val")


def test_sync_tool_worker_thread_shielded_on_validation_failure() -> None:
    """Sync tools do NOT execute or spawn thread pool work if argument validation fails."""
    executed = []

    @tool
    def risky_sync_action(count: int) -> str:
        executed.append(count)
        return "done"

    with pytest.raises(ToolArgumentValidationError):
        risky_sync_action.execute(count="invalid_int")

    assert len(executed) == 0, "Sync tool must not execute when validation fails"


@pytest.mark.asyncio
async def test_async_tool_coroutine_shielded_on_validation_failure() -> None:
    """Async tools do NOT execute when argument validation fails."""
    executed = []

    @tool
    async def risky_async_action(count: int) -> str:
        executed.append(count)
        return "done"

    with pytest.raises(ToolArgumentValidationError):
        await risky_async_action.execute_async(count="invalid_int")

    assert len(executed) == 0, "Async tool must not execute when validation fails"


def test_unannotated_parameters_and_unknown_types_fallback() -> None:
    """Unannotated parameters and custom types falling back to STRING accept inputs without error."""

    class MyCustomService:
        pass

    @tool
    def fallback_tool(x, svc: MyCustomService, flag: bool) -> tuple[Any, Any, bool]:
        return x, svc, flag

    # Accepts any input for x and svc, while strictly validating flag
    res = fallback_tool.execute(x="arbitrary_value", svc="custom_service_str", flag="true")
    assert res == ("arbitrary_value", "custom_service_str", True)

    with pytest.raises(ToolArgumentValidationError):
        fallback_tool.execute(x="val", svc="val", flag="invalid_flag")


def test_validate_args_disabled_flag() -> None:
    """Passing validate_args=False disables validation and allows uncoerced arguments through."""

    @tool(validate_args=False)
    def raw_tool(proceed: bool = False) -> Any:
        return proceed

    # With validation disabled, 'false' is passed as raw string (truthy in Python)
    res = raw_tool.execute(proceed="false")
    assert res == "false"

    # With validation disabled, raw kwargs pass directly to the function
    t2 = Tool(lambda a, **kw: (a, kw), name="t2", validate_args=False)
    assert t2.execute(a=1, unexpected="extra") == (1, {"unexpected": "extra"})


def test_agent_tool_decorator_supports_validate_args() -> None:
    """Agent.tool decorator supports validate_args parameter."""
    agent = Agent(name="TestAgent", instructions="Test agent instructions")

    @agent.tool(validate_args=False)
    def test_func(proceed: bool) -> Any:
        return proceed

    t = agent.tools[0]
    assert t.validate_args is False
    assert t.execute(proceed="false") == "false"


@pytest.mark.asyncio
async def test_runner_handles_validation_error_with_actionable_tool_result() -> None:
    """Runner catches ToolArgumentValidationError and returns actionable ToolResult to the LLM."""
    agent = Agent(name="CalculatorAgent", instructions="You calculate stuff.")

    @agent.tool
    def multiply(a: int, b: int) -> int:
        return a * b

    class MockEngine(GeminiEngine):
        def __init__(self) -> None:
            super().__init__()
            self.turn_count = 0

        async def generate_turn_async(self, agent: Agent, history: list[Message]) -> EngineTurnResult:
            self.turn_count += 1
            if self.turn_count == 1:
                # LLM calls multiply with an invalid string for 'b'
                return EngineTurnResult(
                    tool_calls=[
                        ToolCall(id="call_1", name="multiply", args={"a": 10, "b": "not_an_int"})
                    ]
                )
            # Second turn: inspect history and finish
            return EngineTurnResult(text="I corrected the call.")

    engine = MockEngine()
    runner = Runner(starting_agent=agent, engine=engine)

    response = await runner.run_async(session_id="val_test_session", user_message="Multiply 10 by x")
    assert response.content == "I corrected the call."

    session = runner.get_session("val_test_session")
    assert session is not None

    tool_messages = [m for m in session.history if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_results is not None

    tool_result: ToolResult = tool_messages[0].tool_results[0]
    assert tool_result.is_error is True
    content_str = str(tool_result.content)

    # 1. Actionable feedback given to LLM
    assert "Invalid arguments for tool 'multiply'" in content_str
    assert "parameter 'b'" in content_str
    assert "Please provide valid arguments matching the required schema." in content_str

    # 2. Must NOT be the masked internal error message
    assert "[trace_id:" not in content_str


def test_error_message_does_not_leak_sensitive_values() -> None:
    """ValidationError messages do not include the raw input values, avoiding leakage of secrets."""

    @tool
    def authenticate(pin: int) -> str:
        return "authenticated"

    secret_pin = "super_confidential_token_9999"
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        authenticate.execute(pin=secret_pin)

    msg = exc_info.value.message
    assert "parameter 'pin'" in msg
    assert secret_pin not in msg, f"Secret value '{secret_pin}' must NOT be in error message: '{msg}'"


def test_h1_secret_leakage_comprehensive() -> None:
    """Ensure sensitive input values never leak into str, repr, args, cause, context or tracebacks."""

    @tool
    def process_credentials(token: int) -> str:
        return "ok"

    sensitive_value = "VERY_SECRET_KEY_NEVER_LEAK_ME_777"
    caught_sync = None
    try:
        process_credentials.execute(token=sensitive_value)
    except ToolArgumentValidationError as err:
        caught_sync = err

    assert caught_sync is not None
    assert sensitive_value not in str(caught_sync)
    assert sensitive_value not in repr(caught_sync)
    assert sensitive_value not in str(caught_sync.args)
    assert sensitive_value not in str(caught_sync.original_error)
    assert caught_sync.__cause__ is None
    assert caught_sync.__context__ is None
    tb_text = "".join(traceback.format_exception(caught_sync))
    assert sensitive_value not in tb_text

    # Also verify direct call
    caught_call = None
    try:
        process_credentials(token=sensitive_value)
    except ToolArgumentValidationError as err:
        caught_call = err

    assert caught_call is not None
    assert sensitive_value not in str(caught_call)
    assert caught_call.__cause__ is None
    assert caught_call.__context__ is None


def test_h2_underscore_and_model_config_parameters() -> None:
    """Parameters with leading underscores and model_config validate and preserve exact names."""

    @tool
    def db_operation(
        _id: int,
        model_config: str = "default_config",
        _internal_flag: bool = False,
        schema: str = "public",
        fields: list[str] = None,
    ) -> dict[str, Any]:
        return {
            "_id": _id,
            "model_config": model_config,
            "_internal_flag": _internal_flag,
            "schema": schema,
            "fields": fields,
        }

    # 1. Valid invocation with required and defaults
    res = db_operation.execute(_id="101", _internal_flag="true", schema="analytics")
    assert res == {
        "_id": 101,
        "model_config": "default_config",
        "_internal_flag": True,
        "schema": "analytics",
        "fields": None,
    }

    # 2. Overriding model_config
    res2 = db_operation.execute(_id=202, model_config="custom_config")
    assert res2["_id"] == 202
    assert res2["model_config"] == "custom_config"

    # 3. Missing required parameter with underscore
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        db_operation.execute(schema="public")
    assert "parameter '_id'" in exc_info.value.message

    # 4. Invalid type for parameter with underscore
    with pytest.raises(ToolArgumentValidationError) as exc_info2:
        db_operation.execute(_id="not_an_int")
    assert "parameter '_id'" in exc_info2.value.message


def test_m1_mutable_defaults_preservation() -> None:
    """Omitting an argument with a mutable default preserves native Python default semantics."""

    @tool
    def append_collector(item: str, container: list[str] = []) -> list[str]:  # noqa: B006 - test intentional mutable default preservation
        container.append(item)
        return container

    # Call 1: uses function's default list
    r1 = append_collector.execute(item="first")
    assert r1 == ["first"]

    # Call 2: accumulates into the same default list as defined by Python semantics
    r2 = append_collector.execute(item="second")
    assert r2 == ["first", "second"]

    # Call 3: explicitly provided container overrides without mutating the default
    r3 = append_collector.execute(item="custom_item", container=["custom"])
    assert r3 == ["custom", "custom_item"]

    # Call 4: resuming with default accumulates onto the original default
    r4 = append_collector.execute(item="third")
    assert r4 == ["first", "second", "third"]


def test_m2_string_coercion_policy() -> None:
    """Numbers are coerced to str; booleans and collections are rejected for str."""

    @tool
    def lookup_by_code(code: str) -> str:
        return code

    # Int and float coerced to str
    assert lookup_by_code.execute(code=12345) == "12345"
    assert lookup_by_code.execute(code=-42) == "-42"
    assert lookup_by_code.execute(code=12.5) == "12.5"
    assert lookup_by_code.execute(code=0) == "0"

    # Booleans are rejected
    with pytest.raises(ToolArgumentValidationError) as exc_info_bool:
        lookup_by_code.execute(code=True)
    assert "parameter 'code'" in exc_info_bool.value.message

    with pytest.raises(ToolArgumentValidationError):
        lookup_by_code.execute(code=False)

    # Collections are rejected
    with pytest.raises(ToolArgumentValidationError):
        lookup_by_code.execute(code=["123"])

    with pytest.raises(ToolArgumentValidationError):
        lookup_by_code.execute(code={"key": "val"})


def _union_tool(annotation: Any) -> Tool:
    def echo(x: Any) -> Any:
        return x

    echo.__annotations__["x"] = annotation
    return Tool(echo, name="echo")


@pytest.mark.parametrize("annotation", [str | int, int | str])
def test_union_member_order_does_not_change_delivery(annotation: Any) -> None:
    """A value that already matches a union member exactly is delivered unchanged (#168)."""
    deliver = _union_tool(annotation).execute

    result = deliver(x=5)
    assert result == 5
    assert type(result) is int
    assert deliver(x="5") == "5"
    assert deliver(x=5.0) == "5.0"

    with pytest.raises(ToolArgumentValidationError):
        deliver(x=True)


@pytest.mark.parametrize("annotation", [str | int | None, int | str | None])
def test_optional_union_member_order_does_not_change_delivery(annotation: Any) -> None:
    """Optional unions keep the same member-order independence, including None."""
    deliver = _union_tool(annotation).execute

    assert deliver(x=5) == 5
    assert deliver(x="5") == "5"
    assert deliver(x=None) is None

    with pytest.raises(ToolArgumentValidationError):
        deliver(x=True)


@pytest.mark.parametrize("annotation", [list[str | int], list[int | str]])
def test_union_member_order_inside_a_list(annotation: Any) -> None:
    """The union clean-up recurses into containers, so list elements stay consistent."""
    delivered = _union_tool(annotation).execute(x=[5, "5", 5.0])

    assert delivered == [5, "5", "5.0"]
    assert [type(item) for item in delivered] == [int, str, str]


def test_reject_bool_for_numeric() -> None:
    """Booleans are rejected when int or float is expected."""

    @tool
    def set_threshold(count: int, ratio: float) -> tuple[int, float]:
        return count, ratio

    # Normal values work
    assert set_threshold.execute(count="10", ratio="0.75") == (10, 0.75)

    # Reject bool for int
    with pytest.raises(ToolArgumentValidationError) as exc_int:
        set_threshold.execute(count=True, ratio=0.5)
    assert "parameter 'count'" in exc_int.value.message
    assert "not a boolean" in exc_int.value.message

    # Reject bool for float
    with pytest.raises(ToolArgumentValidationError) as exc_float:
        set_threshold.execute(count=5, ratio=False)
    assert "parameter 'ratio'" in exc_float.value.message
    assert "not a boolean" in exc_float.value.message


def test_reject_nan_and_infinity() -> None:
    """NaN and Infinity values are rejected for numeric parameters."""

    @tool
    def process_measurement(val: float) -> float:
        return val

    for invalid_val in [float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf"]:
        with pytest.raises(ToolArgumentValidationError) as exc_info:
            process_measurement.execute(val=invalid_val)
        assert "parameter 'val'" in exc_info.value.message
        assert "finite" in exc_info.value.message.lower()


def test_unsupported_annotations_fallback_behavior() -> None:
    """N1 & N2: Unsupported annotations (TypedDict, dataclass, BaseModel, Enum, Literal) retain STRING fallback."""

    class OrderStatus(str, Enum):
        PENDING = "pending"
        SHIPPED = "shipped"

    class UserDict(TypedDict):
        username: str
        level: int

    @dataclass
    class Point2D:
        x: float
        y: float

    class ToolConfig(BaseModel):
        retries: int
        active: bool

    @tool
    def process_order(
        status: OrderStatus,
        priority: Literal["low", "high"],
        user: UserDict,
        point: Point2D,
        config: ToolConfig,
    ) -> tuple[str, str, str]:
        return str(status), priority, str(user)

    # In schema, unknown non-union annotations retain the backward-compatible STRING fallback
    schema_props = process_order.parameters_schema["properties"]
    assert schema_props["status"]["type"] == "STRING"
    assert schema_props["priority"]["type"] == "STRING"
    assert schema_props["user"]["type"] == "STRING"
    assert schema_props["point"]["type"] == "STRING"
    assert schema_props["config"]["type"] == "STRING"

    # In validation, STRING is not advertised while requiring an object/enum (fallback to Any)
    res = process_order.execute(
        status="shipped",
        priority="high",
        user={"username": "Alice"},
        point={"x": 1.0, "y": 2.0},
        config={"retries": 3, "active": True},
    )
    assert res == ("shipped", "high", "{'username': 'Alice'}")


def test_supported_dict_and_mapping_validation() -> None:
    """Supported dictionary and mapping annotations generate OBJECT schema and validate dicts."""

    @tool
    def update_settings(settings: dict[str, Any]) -> int:
        return len(settings)

    assert update_settings.parameters_schema["properties"]["settings"]["type"] == "OBJECT"
    assert update_settings.execute(settings={"theme": "dark", "zoom": 100}) == 2

    # Non-dictionary value rejected
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        update_settings.execute(settings="not_a_dict")
    assert "parameter 'settings'" in exc_info.value.message


def test_annotated_metadata_preservation() -> None:
    """Annotated types with Pydantic Field constraints are enforced."""

    @tool
    def transfer(amount: Annotated[int, Field(gt=0)]) -> int:
        return amount

    assert transfer.execute(amount=100) == 100

    with pytest.raises(ToolArgumentValidationError) as exc_info:
        transfer.execute(amount=0)
    assert "parameter 'amount'" in exc_info.value.message


def test_tool_call_consistency() -> None:
    """Direct Tool.__call__ validates arguments consistently with execute and execute_async."""

    @tool
    def confirm(proceed: bool = False, count: int = 1) -> str:
        return f"{proceed}:{count}"

    # Direct call with coerced arguments
    assert confirm(proceed="true", count="5") == "True:5"

    # Direct call with positional arguments
    assert confirm("false", "3") == "False:3"

    # Direct call with invalid arguments raises ToolArgumentValidationError
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        confirm(proceed="not_a_bool")
    assert "confirm" in exc_info.value.tool_name

    # Async tool direct call
    @tool
    async def async_confirm(proceed: bool) -> bool:
        return proceed

    import asyncio
    assert asyncio.run(async_confirm(proceed="true")) is True


class UserProfile(TypedDict):
    name: str
    age: int


def test_n1_typeddict_registration_compatibility() -> None:
    """N1: Tools annotated with typing.TypedDict register without crashing."""
    @tool
    def get_profile(profile: UserProfile) -> str:
        return profile["name"]

    assert isinstance(get_profile, Tool)
    # Under existing behavior, unsupported non-union annotations retain STRING schema
    assert get_profile.parameters_schema["properties"]["profile"]["type"] == "STRING"
    # Fallback contract: accepts provided value without crashing or requiring Pydantic object
    assert get_profile.execute(profile={"name": "Alice", "age": 30}) == "Alice"


def test_n2_schema_validator_consistency() -> None:
    """N2: Schema and validator are consistent; STRING is not advertised while requiring object/enum."""
    class CustomData:
        pass

    @tool
    def handle_custom(data: CustomData) -> str:
        return "handled"

    # Retains STRING fallback in schema
    assert handle_custom.parameters_schema["properties"]["data"]["type"] == "STRING"
    # Validator does not reject under STRING fallback
    assert handle_custom.execute(data="some_string") == "handled"

    # Union with unrepresentable type still raises ToolSchemaError at registration
    from chatflow_agent.exceptions import ToolSchemaError

    def invalid_union_tool(data: CustomData | None) -> None:
        pass

    with pytest.raises(ToolSchemaError):
        Tool(invalid_union_tool)


def test_n3_tool_call_all_parameter_kinds() -> None:
    """N3: Tool.__call__ preserves Python calling semantics for all parameter kinds."""
    @tool
    def complex_fn(
        pos_only: int,
        /,
        pos_or_kw: int,
        *args: int,
        kw_only: int = 10,
        **kwargs: str,
    ) -> tuple[Any, ...]:
        return (pos_only, pos_or_kw, args, kw_only, kwargs)

    # Calling directly using Tool.__call__ with positional-only, var-positional, keyword-only, var-keyword
    res = complex_fn(1, "2", "3", "4", kw_only="20", extra="hello")
    assert res == (1, 2, ("3", "4"), 20, {"extra": "hello"})

    # Validation failure in positional argument raises ToolArgumentValidationError
    with pytest.raises(ToolArgumentValidationError):
        complex_fn(True, 2)  # bool rejected for int pos_only


def test_n4_sensitive_keys_not_leaked_in_messages() -> None:
    """N4: Model-supplied untrusted keys and nested paths do not leak in error messages."""
    @tool
    def lookup_item(item_id: str) -> str:
        return item_id

    secret_key = "sk_live_SUPER_SECRET_TOKEN_99999"
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        lookup_item.validate_arguments({secret_key: "value"})

    # The secret key itself must not be echoed in the message or string representation
    assert secret_key not in exc_info.value.message
    assert secret_key not in str(exc_info.value)
    assert secret_key not in repr(exc_info.value)


def test_n5_bounded_validation_errors() -> None:
    """N5: Validation errors are bounded in count and total message length."""
    @tool
    def simple_tool(item_id: str) -> str:
        return item_id

    # 50 unexpected keys
    many_bad_keys = {f"extra_untrusted_key_{i}": f"val_{i}" for i in range(50)}
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        simple_tool.validate_arguments(many_bad_keys)

    msg = exc_info.value.message
    assert len(msg) <= 300
    # Should not repeat "unexpected" 50 times
    assert msg.count("unexpected") <= 3


def test_n6_implicit_optional_compatibility() -> None:
    """N6: Implicit Optional annotations (e.g. x: int = None) accept None and omitted values."""
    @tool
    def fetch_data(query: str, limit: int = None) -> tuple[str, Any]:
        return query, limit

    # 1. Omitted value
    assert fetch_data.execute(query="test") == ("test", None)
    assert fetch_data(query="test") == ("test", None)

    # 2. Explicit None
    assert fetch_data.execute(query="test", limit=None) == ("test", None)
    assert fetch_data(query="test", limit=None) == ("test", None)

    # 3. Valid integer
    assert fetch_data.execute(query="test", limit=10) == ("test", 10)
    assert fetch_data(query="test", limit="10") == ("test", 10)

    # 4. Invalid value raises ToolArgumentValidationError
    with pytest.raises(ToolArgumentValidationError):
        fetch_data.execute(query="test", limit="not_an_int")

    # 5. Schema reflects nullable
    schema = fetch_data.parameters_schema["properties"]["limit"]
    assert schema.get("nullable") is True


def test_l1_secret_leakage_in_tool_call() -> None:
    """L1: Tool.__call__ does not leak unexpected keyword names, raw bind() messages, or raw input in traceback/cause."""
    import traceback

    @tool
    def process_data(value: int) -> int:
        return value

    secret_key = "sk_live_SECRET_API_KEY_CALL_LEAK"

    with pytest.raises(ToolArgumentValidationError) as exc_info:
        process_data(1, **{secret_key: "forbidden_payload"})

    exc = exc_info.value
    # 1. Message and string representations
    assert secret_key not in exc.message
    assert secret_key not in str(exc)
    assert secret_key not in repr(exc)
    assert secret_key not in str(exc.args)

    # 2. Traceback, cause, context
    formatted_tb = "".join(traceback.format_exception(exc))
    assert secret_key not in formatted_tb
    assert exc.__cause__ is None
    assert exc.__context__ is None


def test_l2_positional_only_and_kwargs_collision() -> None:
    """L2: Signatures with positional-only args and **kw preserve separate namespaces."""
    def native_f(x: int, /, **kw: Any) -> tuple[int, dict[str, Any]]:
        return x, kw

    tool_f = Tool(native_f)

    # Native Python call
    native_res = native_f(1, x="2")
    assert native_res == (1, {"x": "2"})

    # Tool direct call with string coercion
    tool_res = tool_f("1", x="2")
    assert tool_res == (1, {"x": "2"})

    # Tool direct call with multiple keyword args in kw
    tool_res_multi = tool_f(5, x="custom", other="value")
    assert tool_res_multi == (5, {"x": "custom", "other": "value"})


def test_l3_unbound_method_with_first_param_self() -> None:
    """L3: Unbound methods with 'self' as first parameter bind and validate correctly."""
    class CounterService:
        def increment(self, step: int) -> str:
            return f"incremented by {step}"

    svc = CounterService()
    unbound_tool = Tool(CounterService.increment)

    # Valid call with string coercion for step
    assert unbound_tool(svc, "7") == "incremented by 7"
    assert unbound_tool(svc, step="10") == "incremented by 10"

    # Invalid call fails validation cleanly for 'step'
    with pytest.raises(ToolArgumentValidationError) as exc_info:
        unbound_tool(svc, "not_an_int")
    assert "parameter 'step'" in exc_info.value.message


def test_l5_public_api_consistency() -> None:
    """L5: Sanitized error contract, Agent.add_tool override isolation, and mutating tool.validate_args."""
    from chatflow_agent import Agent

    # Part A: ToolArgumentValidationError enforces sanitized error contract
    orig_exc = RuntimeError("underlying failure")
    val_err = ToolArgumentValidationError("tool_name", "custom message", original_error=orig_exc)
    assert isinstance(val_err.original_error, ValueError)
    assert "custom message" in str(val_err.original_error)

    # Part B: Agent.add_tool with Tool instance respects validate_args override via isolated copy
    def sample_func(num: int) -> int:
        return num

    t_strict = Tool(sample_func, validate_args=True)
    agent = Agent(name="test_agent", instructions="test")
    registered_tool = agent.add_tool(t_strict, validate_args=False)
    # Caller's original tool is preserved
    assert t_strict.validate_args is True
    # Agent's registered tool has the override
    assert registered_tool.validate_args is False
    assert registered_tool.execute(num="invalid_for_int") == "invalid_for_int"

    # Part C: Mutating tool.validate_args after construction rebuilds validator atomically
    t_dynamic = Tool(sample_func, validate_args=False)
    assert t_dynamic._validator is None
    assert t_dynamic.validate_args is False

    # Turn validation ON dynamically
    t_dynamic.validate_args = True
    assert t_dynamic._validator is not None
    assert t_dynamic.validate_args is True

    with pytest.raises(ToolArgumentValidationError):
        t_dynamic.execute(num="bad_type")

    # Turn validation OFF dynamically
    t_dynamic.validate_args = False
    assert t_dynamic._validator is None
    assert t_dynamic.execute(num="bad_type") == "bad_type"


def test_r1_shared_tool_configuration_isolation() -> None:
    """R1: Agent.add_tool(existing_tool, validate_args=...) isolates shared tool instances."""
    from chatflow_agent import Agent

    def calculate(value: int) -> int:
        return value * 2

    shared_tool = Tool(calculate, validate_args=True)
    agent_a = Agent(name="agent_a", instructions="agent A")
    agent_a.add_tool(shared_tool)

    agent_b = Agent(name="agent_b", instructions="agent B")
    # Registering on agent_b with validate_args=False must NOT mutate shared_tool
    b_tool = agent_b.add_tool(shared_tool, validate_args=False)

    # 1. shared_tool and agent_a retain validate_args=True
    assert shared_tool.validate_args is True
    assert agent_a.tools[0].validate_args is True
    assert agent_a.tools[0] is shared_tool

    # 2. agent_b has an independently configured copy with validate_args=False
    assert b_tool.validate_args is False
    assert agent_b.tools[0].validate_args is False
    assert b_tool is not shared_tool

    # 3. Validation is strictly active on agent_a's tool
    with pytest.raises(ToolArgumentValidationError):
        agent_a.tools[0].execute(value="invalid_int")

    # 4. Validation is bypassed on agent_b's tool
    assert agent_b.tools[0].execute(value="raw_unvalidated") == "raw_unvalidated" * 2


def test_r2_atomic_validator_configuration_concurrency() -> None:
    """R2: Mutating validate_args publishes configuration atomically without intermediate inconsistent state."""
    import concurrent.futures

    def multiply(factor: int) -> int:
        return factor * 10

    test_tool = Tool(multiply, validate_args=True)
    errors: list[str] = []

    def writer() -> None:
        for i in range(100):
            test_tool.validate_args = (i % 2 == 0)

    def reader() -> None:
        for _ in range(200):
            # Check internal state consistency: enabled must never be True while validator is None
            config = test_tool._validation_config
            if config.enabled and config.validator is None:
                errors.append("Observed enabled=True with validator=None")
            if not config.enabled and config.validator is not None:
                errors.append("Observed enabled=False with validator!=None")

            # Call execute with a valid integer; should always succeed regardless of mode
            res = test_tool(5)
            if res != 50:
                errors.append(f"Unexpected result: {res}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(writer),
            executor.submit(writer),
            executor.submit(reader),
            executor.submit(reader),
            executor.submit(reader),
            executor.submit(reader),
        ]
        concurrent.futures.wait(futures)

    assert errors == []


def test_r3_sanitized_error_contract_no_pydantic_leakage() -> None:
    """R3: ToolArgumentValidationError enforces sanitized error contract and never retains raw Pydantic ValidationError."""
    import traceback

    from pydantic import TypeAdapter, ValidationError

    secret = "sk_live_SECRET_KEY_123"
    adapter = TypeAdapter(int)
    raw_pydantic_err: ValidationError | None = None
    try:
        adapter.validate_python(secret)
    except ValidationError as exc:
        raw_pydantic_err = exc

    assert raw_pydantic_err is not None
    assert secret in str(raw_pydantic_err)

    # Instantiate ToolArgumentValidationError passing the raw Pydantic ValidationError
    err = ToolArgumentValidationError(
        tool_name="sensitive_tool",
        message="tool 'sensitive_tool': parameter 'id': invalid value",
        original_error=raw_pydantic_err,
    )

    # 1. Enforce sanitized contract on original_error: raw ValidationError is NEVER retained
    assert not isinstance(err.original_error, ValidationError)
    assert isinstance(err.original_error, ValueError)

    # 2. Secret does not leak in str/repr of original_error
    assert secret not in str(err.original_error)
    assert secret not in repr(err.original_error)

    # 3. Secret does not leak in str, repr, args, or traceback of ToolArgumentValidationError
    assert secret not in str(err)
    assert secret not in repr(err)
    assert secret not in str(err.args)
    formatted_tb = "".join(traceback.format_exception(err))
    assert secret not in formatted_tb


def test_r4_positional_only_keyword_error_classification() -> None:
    """R4: Positional-only arguments passed as keywords are classified accurately."""
    # Case 1: Required positional-only parameter
    def f_req(x: int, /) -> int:
        return x

    t_req = Tool(f_req)
    with pytest.raises(ToolArgumentValidationError) as exc_req:
        t_req(x=1)
    assert "positional-only argument passed as keyword" in exc_req.value.message

    # Case 2: Positional-only parameter with default
    def f_def(x: int = 10, /) -> int:
        return x

    t_def = Tool(f_def)
    with pytest.raises(ToolArgumentValidationError) as exc_def:
        t_def(x=20)
    assert "positional-only argument passed as keyword" in exc_def.value.message


def test_annotated_with_multiple_metadata_items_keeps_all_constraints() -> None:
    """Annotated metadata is rebuilt without 3.11-only syntax and every constraint applies."""
    lower, upper = Field(gt=0), Field(lt=10)

    @tool
    def pick_level(level: Annotated[int, lower, upper]) -> int:
        return level

    assert pick_level(level="5") == 5
    for out_of_range in (0, 10):
        with pytest.raises(ToolArgumentValidationError):
            pick_level(level=out_of_range)

    sanitized = _sanitize_annotation_for_validation(Annotated[int, lower, upper])
    assert get_origin(sanitized) is Annotated
    # The sanitized int carries its own coercion validator first; the caller's
    # constraints follow it in their original order and identity.
    metadata = get_args(sanitized)[1:]
    assert metadata[-2] is lower and metadata[-1] is upper


def _posonly(x: int, /, y: int = 0) -> int:
    return x + y


def _posonly_with_var_keyword(x: int, /, **extra: Any) -> Any:
    return x, extra


def _kwonly(*, n: int) -> int:
    return n


def _two(a: int, b: int) -> int:
    return a + b


def _named_missing(missing_id: int) -> int:
    return missing_id


@pytest.mark.parametrize(
    ("func", "args", "kwargs", "reason"),
    [
        (_posonly, (), {"x": 1}, "positional-only argument passed as keyword"),
        (_posonly, (), {"x": 1, "y": 2}, "positional-only argument passed as keyword"),
        (_posonly, (), {"y": 2}, "missing required argument"),
        (_posonly_with_var_keyword, (), {"x": 1}, "missing required argument"),
        (_two, (1, 2, 3), {}, "too many positional arguments"),
        (_two, (1,), {"a": 2}, "multiple values provided for argument"),
        (_two, (1,), {}, "missing required argument"),
        (_kwonly, (), {}, "missing required argument"),
        (_kwonly, (), {"n": 1, "unknown": 2}, "unexpected parameter is not permitted"),
    ],
    ids=[
        "posonly-as-keyword",
        "posonly-as-keyword-with-other-args",
        "posonly-missing",
        "posonly-name-absorbed-by-var-keyword",
        "too-many-positional",
        "multiple-values",
        "missing-positional",
        "missing-keyword-only",
        "unexpected-keyword",
    ],
)
def test_bind_failures_are_classified_structurally(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], reason: str
) -> None:
    """Classification depends on parameter kinds, identically on every supported Python."""
    with pytest.raises(ToolArgumentValidationError) as exc:
        Tool(func)(*args, **kwargs)
    assert exc.value.message == f"tool '{func.__name__}': {reason}"


@pytest.mark.parametrize(
    "untrusted_key",
    ["positional-only", "missing", "multiple values", "too many", "unexpected keyword"],
)
def test_untrusted_keyword_names_cannot_steer_classification(untrusted_key: str) -> None:
    """An unexpected keyword is reported as unexpected whatever its name, and is not echoed."""
    with pytest.raises(ToolArgumentValidationError) as exc:
        Tool(_kwonly)(n=1, **{untrusted_key: "secret-value"})
    assert exc.value.message == "tool '_kwonly': unexpected parameter is not permitted"
    rendered = "".join([str(exc.value), repr(exc.value), str(exc.value.args)])
    assert untrusted_key not in rendered
    assert "secret-value" not in rendered


def test_duplicate_value_for_parameter_named_missing_is_not_reported_as_missing() -> None:
    """A parameter name containing 'missing' cannot turn a duplicate value into 'missing'."""
    with pytest.raises(ToolArgumentValidationError) as exc:
        Tool(_named_missing)(1, missing_id=2)
    assert exc.value.message == "tool '_named_missing': multiple values provided for argument"


def test_bind_failure_classification_ignores_type_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Misleading TypeError wording (any Python version) does not affect the reason."""

    def misleading_bind(self: inspect.Signature, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("positional-only missing multiple values unexpected keyword")

    two_tool = Tool(_two)
    monkeypatch.setattr(inspect.Signature, "bind", misleading_bind)
    with pytest.raises(ToolArgumentValidationError) as exc:
        two_tool(1, 2, 3)
    assert exc.value.message == "tool '_two': too many positional arguments"


class _Receiver:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def record(self, amount: int) -> int:
        """Record an amount."""
        self.seen.append(amount)
        return amount


class _AuditedTool(Tool):
    """Tool subclass with a required constructor argument and overridden execution."""

    def __init__(self, func: Callable[..., Any], *, audit_label: str, validate_args: bool = True) -> None:
        super().__init__(func, validate_args=validate_args)
        self.audit_label = audit_label

    async def execute_async(self, **kwargs: Any) -> Any:
        return (self.audit_label, await super().execute_async(**kwargs))

    def execute(self, **kwargs: Any) -> Any:
        return (self.audit_label, super().execute(**kwargs))


@pytest.mark.asyncio
async def test_subclassed_tool_keeps_type_and_behavior_across_agents() -> None:
    """add_tool() clones a subclass with a different setting without mutating the original."""
    receiver = _Receiver()
    shared = _AuditedTool(receiver.record, audit_label="audited")
    strict_agent = Agent(name="Strict", instructions="strict")
    lenient_agent = Agent(name="Lenient", instructions="lenient")

    strict_tool = strict_agent.add_tool(shared)
    lenient_tool = lenient_agent.add_tool(shared, validate_args=False)

    assert strict_tool is shared and shared.validate_args is True
    assert lenient_tool is not shared and type(lenient_tool) is _AuditedTool
    assert lenient_tool.validate_args is False
    assert lenient_tool.audit_label == "audited"
    assert lenient_tool.func is shared.func
    assert lenient_tool.func.__self__ is receiver
    assert (lenient_tool.name, lenient_tool.description) == (shared.name, shared.description)
    assert lenient_tool.signature == shared.signature
    assert lenient_tool.parameters_schema == shared.parameters_schema

    assert await strict_tool.execute_async(amount="2") == ("audited", 2)
    assert await lenient_tool.execute_async(amount="2") == ("audited", "2")
    assert strict_tool.execute(amount="3") == ("audited", 3)
    with pytest.raises(ToolArgumentValidationError):
        await strict_tool.execute_async(amount=True)
    assert await lenient_tool.execute_async(amount=True) == ("audited", True)
    assert receiver.seen == [2, "2", 3, True]

    # The shared instance keeps its own configuration after the clone is toggled.
    lenient_tool.validate_args = True
    assert shared.validate_args is True
    lenient_tool.validate_args = False
    assert shared.validate_args is True and strict_tool.validate_args is True


def test_clone_copies_schema_containers_but_keeps_leaf_identity() -> None:
    """Clones never share mutable schema containers; defaults keep their identity."""
    default_tags: list[str] = []

    def tagged(label: str, tags: list[str] = default_tags) -> list[str]:
        """Return tags."""
        return tags

    original = Tool(tagged)
    duplicate = original.clone(validate_args=False)

    assert duplicate.parameters_schema == original.parameters_schema
    assert duplicate.parameters_schema is not original.parameters_schema
    assert duplicate.parameters_schema["properties"]["tags"]["default"] is default_tags
    duplicate.parameters_schema["properties"]["label"]["description"] = "changed"
    assert original.parameters_schema["properties"]["label"].get("description") != "changed"
    assert duplicate.execute(label="x") is default_tags
    assert original.execute(label="x") is default_tags


def _two_defaulted_posonly(a: int = 0, b: int = 0, /) -> tuple[int, int]:
    return a, b


def _posonly_default_with_var_keyword(x: int = 0, /, **extra: Any) -> tuple[int, dict[str, Any]]:
    return x, extra


def test_defaulted_posonly_passed_as_keyword_is_rejected_on_every_version() -> None:
    """Rejected even where Signature.bind() accepts it (Python 3.13), before any call."""
    with pytest.raises(ToolArgumentValidationError) as exc:
        Tool(_two_defaulted_posonly)(b=1)
    assert exc.value.message == (
        "tool '_two_defaulted_posonly': positional-only argument passed as keyword"
    )
    assert exc.value.__context__ is None and exc.value.__cause__ is None


def test_posonly_name_is_forwarded_into_var_keyword() -> None:
    """Accepted even where Signature.bind() rejects it (Python 3.10/3.11), as Python does."""
    forwarding = Tool(_posonly_default_with_var_keyword)
    assert forwarding(x=5) == (0, {"x": 5})
    assert forwarding(1, x=5) == (1, {"x": 5})
    assert forwarding(x=5, y=6) == (0, {"x": 5, "y": 6})
    assert list(forwarding(y=6, x=5)[1]) == ["y", "x"]
    # The positional value is validated and coerced; the forwarded keyword is untouched.
    assert forwarding("3", x="5") == (3, {"x": "5"})
    assert _posonly_default_with_var_keyword(x=5) == forwarding(x=5)


def test_posonly_rejection_does_not_depend_on_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permissive Signature.bind() (as on Python 3.13) cannot make an invalid call pass."""
    posonly_tool = Tool(_two_defaulted_posonly)
    monkeypatch.setattr(inspect.Signature, "bind", inspect.Signature.bind_partial)
    with pytest.raises(ToolArgumentValidationError) as exc:
        posonly_tool(b=1)
    assert exc.value.message.endswith("positional-only argument passed as keyword")


def test_varargs_and_kwargs_keep_python_semantics() -> None:
    """Declared parameters are validated; *args and **kwargs entries pass through unchanged."""

    def mixed(a: int, *rest: Any, k: int = 0, **extra: Any) -> tuple[Any, ...]:
        return a, rest, k, extra

    assert Tool(mixed)("1", 2, "3", k="4", z="9") == (1, (2, "3"), 4, {"z": "9"})


class _CopyReturnsSelf(Tool):
    def __copy__(self) -> "_CopyReturnsSelf":
        return self


class _CopySharesDict(Tool):
    def __copy__(self) -> "_CopySharesDict":
        duplicate = object.__new__(type(self))
        duplicate.__dict__ = self.__dict__
        return duplicate


@pytest.mark.parametrize("unsafe_cls", [_CopyReturnsSelf, _CopySharesDict])
def test_unsafe_subclass_copy_is_rejected_without_mutation(unsafe_cls: type[Tool]) -> None:
    """clone() refuses copies that alias the original; nothing is mutated or registered."""
    receiver = _Receiver()
    shared = unsafe_cls(receiver.record)
    config_before = shared._validation_config
    schema_before = shared.parameters_schema
    strict_agent = Agent(name="Strict", instructions="strict")
    lenient_agent = Agent(name="Lenient", instructions="lenient")
    strict_agent.add_tool(shared)

    with pytest.raises(TypeError, match=r"must return a distinct Tool instance"):
        lenient_agent.add_tool(shared, validate_args=False)
    with pytest.raises(TypeError, match=r"must return a distinct Tool instance"):
        shared.clone(validate_args=False)

    assert shared.validate_args is True
    assert shared._validation_config is config_before
    assert shared.parameters_schema is schema_before
    assert lenient_agent.tools == []
    assert strict_agent.tools == [shared]
    with pytest.raises(ToolArgumentValidationError):
        strict_agent.tools[0].execute(amount=True)
