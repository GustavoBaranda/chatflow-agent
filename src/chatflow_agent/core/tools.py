"""Tool abstraction and introspection engine for chatflow-agent."""

import asyncio
import copy
import functools
import inspect
import logging
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)

from chatflow_agent.core._sync_work import _run_sync_tool
from chatflow_agent.exceptions import (
    ToolArgumentValidationError,
    ToolExecutionError,
    ToolSchemaError,
)

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


def _coerce_str(v: Any) -> Any:
    """Coerce int and float numbers to str, rejecting booleans, collections, and objects."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return v


def _reject_bool_for_int(v: Any) -> Any:
    """Reject booleans when an integer is expected."""
    if isinstance(v, bool):
        raise ValueError("Input should be a valid integer, not a boolean")
    return v


def _reject_bool_for_float(v: Any) -> Any:
    """Reject booleans when a number/float is expected."""
    if isinstance(v, bool):
        raise ValueError("Input should be a valid number, not a boolean")
    return v


# ``Annotated`` typed as ``Any`` so annotations can be rebuilt from runtime metadata.
_ANNOTATED_FORM: Any = Annotated

CoercedStr = Annotated[str, BeforeValidator(_coerce_str)]
CoercedInt = Annotated[int, BeforeValidator(_reject_bool_for_int)]
CoercedFloat = Annotated[float, BeforeValidator(_reject_bool_for_float)]


def _is_str_union_member(arg: Any) -> bool:
    """Whether a union member is ``str`` (possibly ``Annotated[str, ...]``)."""
    if isinstance(arg, str):  # forward reference, sanitized to Any elsewhere
        return False
    origin = get_origin(arg)
    while origin is Annotated:
        arg = get_args(arg)[0]
        if isinstance(arg, str):
            return False
        origin = get_origin(arg)
    return (origin or arg) is str


def _sanitize_annotation_for_validation(ann: Any) -> Any:
    """Sanitize annotations for runtime validation.

    Coerces numeric strings and rejects booleans for numbers. Only the schema-mapped
    types in ``TYPE_MAP`` (and unions, ``Annotated`` wrappers, and containers of them)
    are validated. Every other annotation, including ``Enum``, ``Literal``,
    ``TypedDict``, dataclasses, ``BaseModel`` subclasses, string forward references,
    and unresolvable types, falls back to ``Any`` and is accepted without validation.
    """
    if ann is inspect.Parameter.empty or ann is Any:
        return Any
    if isinstance(ann, str):
        return Any
    origin = get_origin(ann)
    if origin is Annotated:
        underlying = get_args(ann)[0]
        metadata = get_args(ann)[1:]
        clean = _sanitize_annotation_for_validation(underlying)
        if clean is Any:
            return Any
        # Build the parameter tuple separately: subscripting with a starred expression
        # is Python 3.11+ syntax, and the project targets Python 3.10.
        annotated_params = (clean, *metadata)
        return _ANNOTATED_FORM[annotated_params]
    if origin in (Union, types.UnionType):
        clean_union_args: list[Any] = []
        for arg in get_args(ann):
            if arg is type(None):
                clean_union_args.append(type(None))
            else:
                clean_arg = _sanitize_annotation_for_validation(arg)
                clean_union_args.append(clean_arg)
        if Any in clean_union_args:
            return Any
        # Pydantic's smart union takes the first member that validates in a pass,
        # and CoercedStr turns a number into text during that pass. A str member
        # listed first therefore captures values that already match another member
        # exactly (str | int sent 5 as "5"). Trying the str member last keeps the
        # exact match; its coercion still applies when no other member accepts.
        clean_union_args.sort(key=_is_str_union_member)
        return Union[tuple(clean_union_args)]  # noqa: UP007

    container = origin or ann
    if container is str:
        return CoercedStr
    if container is int:
        return CoercedInt
    if container is float:
        return CoercedFloat
    if container is bool:
        return bool

    if container not in TYPE_MAP:
        return Any

    args = get_args(ann)
    if args:
        if origin is tuple:
            if args == () or args == ((),):
                return tuple[()]
            if len(args) == 2 and args[1] is Ellipsis:
                clean_item = _sanitize_annotation_for_validation(args[0])
                if origin is not None:
                    return origin[(clean_item, Ellipsis)]
                return tuple[Any, ...]
        clean_args = tuple(_sanitize_annotation_for_validation(arg) for arg in args)
        if origin is not None:
            try:
                return origin[clean_args]
            except Exception:
                return origin
        return ann

    return ann


MAX_VALIDATION_ERRORS = 3
MAX_ERROR_MESSAGE_LENGTH = 300


def _format_validation_error(
    err: ValidationError,
    tool_name: str,
    known_params: Sequence[str] | None = None,
) -> str:
    """Format Pydantic ValidationError into a safe, bounded message.

    * Only echoes safe parameter names declared in the tool signature; model-supplied keys
      and nested paths are never echoed to prevent leaking secrets.
    * Limits the number of reported errors (max 3) and total message length (max 300 chars)
      to prevent excessive logs or LLM context usage.
    * Removes internal Pydantic implementation details (e.g. URLs, error codes).
    """
    known = set(known_params or ())
    messages: list[str] = []
    has_unexpected = False

    raw_errors = err.errors()
    for e in raw_errors[:MAX_VALIDATION_ERRORS]:
        err_type = e.get("type", "")
        loc = e.get("loc", ())
        root_key = str(loc[0]) if loc else ""

        if err_type == "extra_forbidden" or root_key not in known:
            # Untrusted model-supplied key: do not echo the untrusted name
            if not has_unexpected:
                messages.append("unexpected parameter is not permitted")
                has_unexpected = True
            continue

        # For known parameters, only use the safe root parameter name declared in Python
        safe_param = root_key
        msg = e.get("msg", "invalid value")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        elif msg == "Field required":
            msg = "value is required"
        if " [type=" in msg:
            msg = msg.split(" [type=")[0]

        messages.append(f"parameter '{safe_param}': {msg}")

    if len(raw_errors) > MAX_VALIDATION_ERRORS:
        messages.append(f"... ({len(raw_errors) - MAX_VALIDATION_ERRORS} additional errors omitted)")

    details = "; ".join(messages)
    full_message = f"Invalid arguments for tool '{tool_name}': {details}"
    if len(full_message) > MAX_ERROR_MESSAGE_LENGTH:
        full_message = full_message[:MAX_ERROR_MESSAGE_LENGTH - 3] + "..."
    return full_message


def _structural_call_error(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> str | None:
    """Return a fixed, sanitized reason why a call is invalid, or ``None`` if it is valid.

    Acceptance and rejection follow Python's calling semantics, derived only from parameter
    kinds and from which parameters the supplied arguments fill. ``Signature.bind()`` is not
    used for this decision because it disagrees across supported Python versions for some
    positional-only cases. No ``TypeError`` text is inspected, so neither the Python
    version's wording nor caller-supplied argument names can change the result, and no
    argument name or value is echoed.
    """
    params = list(signature.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    if not has_var_positional and len(args) > len(positional):
        return "too many positional arguments"

    # Without **kwargs, a keyword matching a positional-only name can never bind.
    if not has_var_keyword and any(
        p.kind == inspect.Parameter.POSITIONAL_ONLY and p.name in kwargs for p in positional
    ):
        return "positional-only argument passed as keyword"

    if any(
        p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name in kwargs
        for p in positional[: len(args)]
    ):
        return "multiple values provided for argument"

    keyword_names = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if not has_var_keyword and any(key not in keyword_names for key in kwargs):
        return "unexpected parameter is not permitted"

    for index, p in enumerate(positional):
        supplied = index < len(args) or (
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name in kwargs
        )
        if not supplied and p.default is inspect.Parameter.empty:
            return "missing required argument"
    if any(
        p.kind == inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
        and p.name not in kwargs
        for p in params
    ):
        return "missing required argument"

    return None


def _bind_call(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> inspect.BoundArguments:
    """Bind a call that ``_structural_call_error`` accepted.

    With ``**kwargs``, a keyword named like a positional-only parameter belongs in the
    var-keyword mapping. Python 3.10 and 3.11 ``Signature.bind()`` reject that call, so such
    keywords are bound separately and merged into the mapping in their original order.
    """
    params = signature.parameters.values()
    var_keyword = next((p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD), None)
    if var_keyword is None:
        return signature.bind(*args, **kwargs)

    positional_only = {p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_ONLY}
    absorbed = {key: value for key, value in kwargs.items() if key in positional_only}
    if not absorbed:
        return signature.bind(*args, **kwargs)

    regular = {key: value for key, value in kwargs.items() if key not in positional_only}
    bound = signature.bind(*args, **regular)
    bound_extra = bound.arguments.get(var_keyword.name, {})
    bound.arguments[var_keyword.name] = {
        key: absorbed[key] if key in absorbed else bound_extra[key]
        for key in kwargs
        if key in absorbed or key in bound_extra
    }
    return bound


def _copy_schema_containers(schema: Any) -> Any:
    """Copy the dict and list structure of a schema, keeping leaf values by identity.

    ``default`` values are user objects taken from the signature (possibly mutable lists
    or dicts), so they are kept as-is rather than treated as schema containers.
    """
    if isinstance(schema, dict):
        return {
            key: value if key == "default" else _copy_schema_containers(value)
            for key, value in schema.items()
        }
    if isinstance(schema, list):
        return [_copy_schema_containers(item) for item in schema]
    return schema


@dataclass(frozen=True)
class _ValidationConfig:
    """Immutable validation state ensuring atomic state publishing."""

    enabled: bool
    validator: tuple[TypeAdapter[Any], dict[str, str], bool] | None


class Tool:
    """Encapsulates a callable tool with introspection for LLM function calling."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        *,
        validate_args: bool = True,
    ) -> None:
        self.func = func
        self.name = name or self._extract_name(func)
        self.description = (
            description or self._extract_description(func) or f"Execute {self.name}"
        )
        self.signature = _resolve_signature(func, self.name)
        self.is_async = self._detect_async(func)
        self.parameters_schema = self._build_parameters_schema()
        initial_enabled = bool(validate_args)
        initial_validator = self._build_validator() if initial_enabled else None
        self._validation_config = _ValidationConfig(
            enabled=initial_enabled,
            validator=initial_validator,
        )

    @property
    def validate_args(self) -> bool:
        """Whether runtime argument validation is enabled for this tool."""
        return self._validation_config.enabled

    @validate_args.setter
    def validate_args(self, value: bool) -> None:
        enabled = bool(value)
        # Construct the new validator before publishing the enabled configuration
        # to ensure concurrent readers never observe an inconsistent state.
        new_validator = self._build_validator() if enabled else None
        self._validation_config = _ValidationConfig(
            enabled=enabled,
            validator=new_validator,
        )

    @property
    def _validator(self) -> tuple[TypeAdapter[Any], dict[str, str], bool] | None:
        """Internal validator tuple, exposed for compatibility."""
        return self._validation_config.validator

    def clone(self, *, validate_args: bool | None = None) -> "Tool":
        """Create an independently configured shallow copy of this tool.

        The copy is made with ``copy.copy()``, so it keeps the concrete ``Tool`` subclass,
        overridden methods, callable identity (including bound-method receivers), metadata,
        signature, and defaults. Instance attributes are copied shallowly: custom mutable
        attributes (for example a subclass's list or dict) are the same objects in both tools.
        Schema containers are copied so the tools never share mutable schema state; schema
        leaf values such as defaults keep their identity. Validation configuration is
        immutable and is never shared mutably.

        Raises ``TypeError``, before modifying either object, if a subclass ``__copy__``
        returns the original instance, a non-``Tool`` object, or an object sharing the
        original's instance dictionary. The original instance is never mutated.
        """
        target_enabled = self.validate_args if validate_args is None else bool(validate_args)
        duplicate = copy.copy(self)
        if (
            not isinstance(duplicate, Tool)
            or duplicate is self
            or duplicate.__dict__ is self.__dict__
        ):
            raise TypeError(
                f"{type(self).__name__}.__copy__() must return a distinct Tool instance "
                "that does not share the original's attributes; Tool.clone() cannot "
                "isolate its validation configuration."
            )
        duplicate.parameters_schema = _copy_schema_containers(self.parameters_schema)
        current = self._validation_config
        if current.enabled == target_enabled:
            # Immutable configuration: safe to share between the original and the copy.
            duplicate._validation_config = current
        else:
            duplicate._validation_config = _ValidationConfig(
                enabled=target_enabled,
                validator=duplicate._build_validator() if target_enabled else None,
            )
        return duplicate

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
                if param.default is None:
                    param_def["nullable"] = True

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

    def _build_validator(
        self,
    ) -> tuple[TypeAdapter[Any], dict[str, str], bool] | None:
        """Build a Pydantic TypeAdapter with aliased fields to avoid collisions with BaseModel.

        Returns (type_adapter, param_map, has_var_keyword) or None.
        param_map maps internal_field_name -> original_param_name.
        """
        fields: dict[str, Any] = {}
        param_map: dict[str, str] = {}
        has_var_keyword = False

        for i, (param_name, param) in enumerate(self.signature.parameters.items()):
            if param_name in ("self", "cls"):
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue

            internal_name = f"arg_{i}_{param_name.lstrip('_') or 'p'}"
            param_map[internal_name] = param_name

            ann = _sanitize_annotation_for_validation(param.annotation)
            if param.default is None and ann is not Any:
                ann = Union[ann, type(None)]  # noqa: UP007

            field_def = (
                Field(..., alias=param_name)
                if param.default is inspect.Parameter.empty
                else Field(param.default, alias=param_name)
            )
            fields[internal_name] = (ann, field_def)

        config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="allow" if has_var_keyword else "forbid",
            protected_namespaces=(),
            allow_inf_nan=False,
        )
        model = create_model(f"ToolParams_{self.name}", __config__=config, **fields)
        return TypeAdapter(model), param_map, has_var_keyword

    def validate_arguments(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and coerce tool arguments against parameter annotations.

        Returns a dictionary containing validated arguments that were provided in kwargs.
        Raises ToolArgumentValidationError if validation fails.
        """
        config = self._validation_config
        if not config.enabled or config.validator is None:
            return dict(kwargs)

        adapter, param_map, has_var_keyword = config.validator
        validation_error: ToolArgumentValidationError | None = None
        known_params = list(self.signature.parameters.keys())

        try:
            validated_instance = adapter.validate_python(kwargs)
        except ValidationError as err:
            formatted_msg = _format_validation_error(err, self.name, known_params)
            validation_error = ToolArgumentValidationError(
                tool_name=self.name,
                message=formatted_msg,
            )

        # Raise strictly outside the except block to ensure __cause__ and __context__ remain None
        if validation_error is not None:
            raise validation_error

        # Extract only arguments provided in kwargs, preserving native default semantics
        result: dict[str, Any] = {}
        for internal_name, original_name in param_map.items():
            if original_name in kwargs:
                result[original_name] = getattr(validated_instance, internal_name)

        if has_var_keyword:
            extra = getattr(validated_instance, "__pydantic_extra__", None)
            if extra:
                result.update(extra)

        return result

    async def execute_async(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously with error handling.

        This is the recommended execution method in all asynchronous contexts
        (e.g., within Runners, FastAPI endpoints, Telegram handlers, or active asyncio tasks).
        If the underlying tool is a synchronous function, it is dispatched to a worker thread
        via ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        try:
            validated_kwargs = self.validate_arguments(kwargs)
        except ToolArgumentValidationError:
            raise
        except Exception as err:
            raise ToolExecutionError(tool_name=self.name, original_error=err) from err

        try:
            if self.is_async:
                return await self.func(**validated_kwargs)
            # Run sync function in an executor thread to avoid blocking event loop. Inside
            # a Runner turn its real lifetime is tracked even if this await is cancelled.
            return await _run_sync_tool(self.func, validated_kwargs)
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
            validated_kwargs = self.validate_arguments(kwargs)
        except ToolArgumentValidationError:
            raise
        except Exception as err:
            raise ToolExecutionError(tool_name=self.name, original_error=err) from err

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
                return asyncio.run(self.func(**validated_kwargs))
            return self.func(**validated_kwargs)
        except Exception as err:
            raise ToolExecutionError(tool_name=self.name, original_error=err) from err

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the tool callable directly.

        When validate_args is True, binds arguments against the signature,
        validates them against annotations, and invokes the function preserving
        all parameter kinds (positional-only, var-positional, keyword-only, var-keyword).
        """
        config = self._validation_config
        if not config.enabled or config.validator is None:
            return self.func(*args, **kwargs)

        # Decide validity structurally, never from Signature.bind() alone: bind() disagrees
        # across supported Python versions for some positional-only cases.
        reason = _structural_call_error(self.signature, args, kwargs)
        bound: inspect.BoundArguments | None = None
        if reason is None:
            try:
                bound = _bind_call(self.signature, args, kwargs)
            except TypeError:
                reason = "invalid arguments"

        if bound is None:
            # Raised outside any except block so no TypeError is chained as context.
            raise ToolArgumentValidationError(
                tool_name=self.name,
                message=f"tool '{self.name}': {reason or 'invalid arguments'}",
            )

        # Filter declared parameters for validation, excluding:
        # 1. Receiver instance/class parameters ("self", "cls") for unbound methods
        # 2. Var-positional (*args)
        # 3. Var-keyword (**kwargs)
        to_validate: dict[str, Any] = {}
        for name, param in self.signature.parameters.items():
            if name in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in bound.arguments:
                to_validate[name] = bound.arguments[name]

        validated_kwargs = self.validate_arguments(to_validate)

        # Update bound arguments with coerced/validated values for declared parameters only.
        # This preserves positional-only arguments and leaves separate **kwargs entries untouched.
        for name in list(bound.arguments.keys()):
            if name in validated_kwargs:
                bound.arguments[name] = validated_kwargs[name]

        return self.func(*bound.args, **bound.kwargs)

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r} async={self.is_async}>"


def tool(
    name_or_func: str | Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    validate_args: bool = True,
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
        return Tool(func=name_or_func, validate_args=validate_args)

    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or (name_or_func if isinstance(name_or_func, str) else None)
        return Tool(
            func=fn,
            name=tool_name,
            description=description,
            validate_args=validate_args,
        )

    return decorator
