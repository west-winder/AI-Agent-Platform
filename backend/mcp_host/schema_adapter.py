"""A deliberately finite JSON Schema dialect backed by lossless Pydantic fields."""

from copy import deepcopy
import json
import math
from typing import Any

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 20_000
MAX_JSON_BYTES = 1_048_576
META = {"title", "description", "default", "examples"}
SCALARS = {"string", "boolean", "integer", "number", "null"}
DIALECT = "https://json-schema.org/draft/2020-12/schema"


class UnsupportedSchema(ValueError):
    """Registration failure with no remote schema included in the message."""


def check_json(value: Any) -> None:
    """Bounded standard JSON tree; no Python coercion or non-finite numbers."""
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > MAX_JSON_DEPTH or count > MAX_JSON_NODES:
            raise ValueError("JSON resource limit exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            if count + len(pending) + len(item) > MAX_JSON_NODES:
                raise ValueError("JSON resource limit exceeded")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if count + len(pending) + len(item) > MAX_JSON_NODES:
                raise ValueError("JSON resource limit exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Non-JSON value")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("Non-finite JSON number")
    try:
        size = len(json.dumps(value, ensure_ascii=True, allow_nan=False).encode("ascii"))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid JSON tree") from exc
    if size > MAX_JSON_BYTES:
        raise ValueError("JSON resource limit exceeded")


def _audit(schema: dict) -> None:
    check_json(schema)
    if type(schema) is not dict or schema.get("type") != "object":
        raise UnsupportedSchema("Expected an object input schema")
    if set(schema) - (META | {"$schema", "type", "properties", "required", "additionalProperties"}):
        raise UnsupportedSchema("Unsupported root keyword")
    if "$schema" in schema and schema["$schema"] != DIALECT:
        raise UnsupportedSchema("Unsupported schema dialect")
    if type(schema.get("additionalProperties", True)) is not bool:
        raise UnsupportedSchema("Unsupported additionalProperties")
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if type(props) is not dict or type(required) is not list:
        raise UnsupportedSchema("Invalid object schema")
    if len(props) > 256:
        raise UnsupportedSchema("Schema property limit exceeded")
    if any(type(x) is not str or x not in props for x in required) or len(set(required)) != len(required):
        raise UnsupportedSchema("Invalid required properties")
    for prop in props.values():
        if type(prop) is not dict or set(prop) - (META | {"type", "enum", "const", "minLength", "maxLength"}):
            raise UnsupportedSchema("Unsupported property keyword")
        if not ({"type", "enum", "const"} & set(prop)):
            raise UnsupportedSchema("Unconstrained property")
        if "type" in prop:
            kinds = prop["type"] if type(prop["type"]) is list else [prop["type"]]
            if not kinds or any(type(k) is not str or k not in SCALARS for k in kinds) or len(set(kinds)) != len(kinds):
                raise UnsupportedSchema("Unsupported property type")
        values = prop.get("enum", [])
        if "enum" in prop and (type(values) is not list or not values or len(values) > 256):
            raise UnsupportedSchema("Invalid enum")
        for i, value in enumerate(values):
            if type(value) in (dict, list) or any(
                Draft202012Validator({"const": old}).is_valid(value) for old in values[:i]
            ):
                raise UnsupportedSchema("Unsupported enum")
        if "const" in prop and type(prop["const"]) in (dict, list):
            raise UnsupportedSchema("Unsupported const")
        for key in ("minLength", "maxLength"):
            if key in prop and (prop.get("type") != "string" or type(prop[key]) is not int or prop[key] < 0):
                raise UnsupportedSchema("Unsupported string constraint")
        if prop.get("minLength", 0) > prop.get("maxLength", math.inf):
            raise UnsupportedSchema("Inconsistent string lengths")
    Draft202012Validator.check_schema(schema)


def build_args_model(input_schema: dict) -> type[BaseModel]:
    try:
        _audit(input_schema)
    except (ValueError, TypeError, SchemaError) as exc:
        raise UnsupportedSchema("Unsupported MCP input schema") from None
    source = deepcopy(input_schema)
    validator = Draft202012Validator(source)

    class Arguments(BaseModel):
        model_config = ConfigDict(
            extra="forbid" if source.get("additionalProperties") is False else "allow",
            validate_by_alias=True, validate_by_name=False, validate_default=False,
        )

        @model_validator(mode="wrap")
        @classmethod
        def validate_original(cls, value, handler):
            check_json(value)
            if type(value) is not dict or next(validator.iter_errors(value), None) is not None:
                raise ValueError("MCP arguments do not match input schema")
            model = handler(value)
            # Pydantic includes extra keys in fields_set. An extra named field_0
            # must not mark the declared field aliased by optional_text as present.
            # Extras remain in model_extra and are still serialized independently.
            model.model_fields_set.difference_update(
                name for name, field in cls.model_fields.items() if field.alias not in value
            )
            return model

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema, handler):
            return deepcopy(source)

    required = source.get("required", [])
    fields = {
        f"field_{i}": (Any, Field(default=... if name in required else None, alias=name))
        for i, name in enumerate(source.get("properties", {}))
    }
    return create_model("MCPArguments", __base__=Arguments, **fields)
