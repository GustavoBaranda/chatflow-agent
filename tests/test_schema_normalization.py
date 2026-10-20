
from chatflow_agent.core.engines.openai_compatible import _lowercase_schema_types
from chatflow_agent.core.tools import Tool


def test_normalize_schema_type_preserves_string_default():
    schema = {
        "type": "OBJECT",
        "properties": {
            "shipping": {
                "type": "STRING",
                "default": "EXPRESS",
            }
        },
    }

    result = _lowercase_schema_types(schema)

    assert result["type"] == "object"
    assert result["properties"]["shipping"]["type"] == "string"
    assert result["properties"]["shipping"]["default"] == "EXPRESS"


def test_normalize_schema_preserves_nested_default():
    schema = {
        "type": "OBJECT",
        "default": {
            "type": "Premium",
            "settings": {
                "type": "FAST",
            },
        },
    }

    result = _lowercase_schema_types(schema)

    assert result["type"] == "object"
    assert result["default"] == schema["default"]


def test_normalize_schema_preserves_list_default():
    schema = {
        "type": "ARRAY",
        "items": {"type": "STRING"},
        "default": [
            {"type": "EXPRESS"},
            {"type": "Standard"},
        ],
    }

    result = _lowercase_schema_types(schema)

    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    assert result["default"] == schema["default"]


def test_normalize_schema_property_named_default():
    schema = {
        "type": "OBJECT",
        "properties": {
            "default": {
                "type": "INTEGER",
                "default": 1,
            }
        },
    }

    result = _lowercase_schema_types(schema)

    assert result["type"] == "object"
    assert result["properties"]["default"] == {
        "type": "integer",
        "default": 1,
    }


def test_normalize_schema_preserves_nested_array_items():
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
        },
    }

    result = _lowercase_schema_types(schema)

    assert result == {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
        },
    }


def test_normalize_tool_array_schema_preserves_items_and_defaults():
    def process(
        matrix: list[list[int]],
        tags: set[str] | None = None,
    ) -> None:
        pass

    result = _lowercase_schema_types(Tool(process).parameters_schema)

    assert result["properties"] == {
        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "tags": {"type": "array", "items": {"type": "string"}, "default": None},
    }
    assert result["required"] == ["matrix"]
