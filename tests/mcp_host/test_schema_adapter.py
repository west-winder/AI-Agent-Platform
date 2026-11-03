from copy import deepcopy
import math

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from backend.mcp_host.schema_adapter import build_args_model, check_json, UnsupportedSchema


def schema(prop, **root):
    return {"type": "object", "properties": {"value": prop}, **root}


@pytest.mark.parametrize("prop,values", [
    ({"type": "string", "minLength": 1, "maxLength": 2}, ["", "中", "😀中", "abc", 1, None]),
    ({"type": "integer"}, [1, 1.0, 1.1, True, "1", None]),
    ({"type": "number"}, [1, 1.5, True, "1"]),
    ({"type": "boolean"}, [True, False, 0, 1, "true"]),
    ({"type": "null"}, [None, False, ""]),
    ({"type": ["string", "null"]}, [None, "", 1]),
    ({"enum": [1]}, [1, 1.0, True, "1", None]),
    ({"const": False}, [False, 0, None]),
    ({"type": "string", "enum": ["a", "b"], "const": "a"}, ["a", "b", "c", 1]),
])
@pytest.mark.parametrize("required", [False, True])
def test_semantic_oracle(prop, values, required):
    source = schema(prop, required=["value"] if required else [])
    model = build_args_model(source)
    oracle = Draft202012Validator(source)
    for data in [{}, *({"value": x} for x in values)]:
        if oracle.is_valid(data):
            dumped = model.model_validate(data).model_dump(mode="python", by_alias=True, exclude_unset=True)
            assert dumped == data
            if "value" in data:
                assert type(dumped["value"]) is type(data["value"])
        else:
            with pytest.raises(ValidationError):
                model.model_validate(data)


@pytest.mark.parametrize("extra", [None, True, False])
def test_extra_and_defaults(extra):
    source = schema({"type": "string", "default": "do-not-insert"})
    if extra is not None:
        source["additionalProperties"] = extra
    model = build_args_model(source)
    assert model.model_validate({}).model_dump(by_alias=True, exclude_unset=True) == {}
    data = {"other": {"nested": [None, True, 1.0]}}
    if extra is False:
        with pytest.raises(ValidationError):
            model.model_validate(data)
    else:
        assert model.model_validate(data).model_dump(by_alias=True, exclude_unset=True) == data


def test_alias_hook_snapshot_and_required_default():
    source = {"type": "object", "properties": {
        "model_dump": {"type": "string"}, "_hidden": {"type": "null"},
        "field_0": {"type": "string"}, "a.b": {"type": "boolean", "default": True},
    }, "required": ["a.b"]}
    model = build_args_model(source)
    original = deepcopy(source)
    data = {"model_dump": "ok", "_hidden": None, "field_0": "kept", "a.b": False, "field_1": 42}
    assert model.model_validate(data).model_dump(by_alias=True, exclude_unset=True) == data
    exported = model.model_json_schema()
    assert exported == original
    exported["properties"].clear()
    source.clear()
    assert model.model_json_schema() == original
    with pytest.raises(ValidationError):
        model.model_validate({})


@pytest.mark.parametrize("bad", [
    {"type": "array"}, {"type": "object", "additionalProperties": {}},
    schema({}), schema({"type": "object"}), schema({"type": "array"}),
    schema({"type": "string", "format": "uri"}), schema({"type": "string", "pattern": "x"}),
    schema({"type": "number", "minimum": 0}), schema({"$ref": "https://example.invalid"}),
    schema({"type": "string", "minLength": True}), schema({"type": "string", "maxLength": -1}),
    schema({"type": "string", "minLength": 2, "maxLength": 1}), schema({"enum": []}),
    schema({"enum": [1, 1.0]}), schema({"enum": [{}]}), schema({"const": []}),
    schema({"type": []}), schema({"type": ["string", "string"]}),
    schema({"const": math.inf}), schema({"type": "string"}, required=["missing"]),
    schema({"type": "string"}, required=["value", "value"]),
    schema({"type": "string"}, **{"$schema": "http://json-schema.org/draft-07/schema#"}),
    {"type": "object", "anyOf": []}, {"type": "object", "x-validation": True},
])
def test_reject_unsupported(bad):
    with pytest.raises(UnsupportedSchema):
        build_args_model(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), b"a", (1,), {1: "x"}, object()])
def test_non_json(value):
    model = build_args_model({"type": "object"})
    with pytest.raises(ValidationError):
        model.model_validate({"extra": value})


def test_depth_size_nodes_and_cycle():
    deep = {}
    for _ in range(34):
        deep = {"x": deep}
    cyclic = []; cyclic.append(cyclic)
    for value in (deep, [0] * 20_001, "x" * 1_048_577, cyclic):
        with pytest.raises(ValueError):
            check_json(value)


@pytest.mark.parametrize("extra", [None, True, False])
@pytest.mark.parametrize("data", [
    {}, {"field_0": "extra-value"},
    {"field_0": "extra-0", "field_1": "extra-1", "field_2": None},
    {"optional_text": "provided", "field_0": "extra"},
    {"nullable": None, "field_0": "extra-0", "field_1": "extra-1"},
    {"optional_text": None},
    {"optional_text": "provided", "nullable": "text", "field_0": None},
])
def test_internal_name_collisions_preserve_original_json(extra, data):
    source = {"type": "object", "properties": {
        "optional_text": {"type": "string", "default": "never-insert"},
        "nullable": {"type": ["string", "null"]},
        "other": {"type": "integer", "default": 42},
    }}
    if extra is not None:
        source["additionalProperties"] = extra
    model = build_args_model(source)
    oracle = Draft202012Validator(source)
    if not oracle.is_valid(data):
        with pytest.raises(ValidationError):
            model.model_validate(data)
    else:
        dumped = model.model_validate(deepcopy(data)).model_dump(
            mode="python", by_alias=True, exclude_unset=True)
        assert dumped == data
        assert oracle.is_valid(dumped)
        assert set(dumped) == set(data)
