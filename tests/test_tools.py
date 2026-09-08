import pytest

from adbench.harness.errors import (
    ToolArgumentError,
    ToolExecutionError,
    UnknownToolError,
)
from adbench.harness.tools import ToolRegistry, ToolSpec, build_demo_registry

# --- individual tool functions ---

def test_get_weather_known_city():
    registry = build_demo_registry()
    result = registry.call("get_weather", {"city": "Paris"})
    assert result["city"] == "Paris"
    assert "temperature_f" in result and "condition" in result


def test_get_weather_unknown_city_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolExecutionError):
        registry.call("get_weather", {"city": "Atlantis"})


def test_calculator_basic():
    registry = build_demo_registry()
    result = registry.call("calculator", {"expression": "12 * (3 + 4)"})
    assert result["result"] == 84


def test_calculator_disallowed_characters_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculator", {"expression": "__import__('os')"})


def test_calculator_division_by_zero_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculator", {"expression": "1 / 0"})


def test_calculator_malformed_expression_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculator", {"expression": "1 + * 2"})


def test_search_knowledge_base_match():
    registry = build_demo_registry()
    result = registry.call("search_knowledge_base", {"query": "lora"})
    assert "lora" in result["matches"]


def test_search_knowledge_base_empty_query_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("search_knowledge_base", {"query": "   "})


def test_search_knowledge_base_no_match_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolExecutionError):
        registry.call("search_knowledge_base", {"query": "quantum gravity"})


def test_get_current_time_known_timezone():
    registry = build_demo_registry()
    result = registry.call("get_current_time", {"timezone": "utc"})
    assert result["timezone"] == "utc"
    assert result["current_time"]


def test_get_current_time_unknown_timezone_raises():
    registry = build_demo_registry()
    with pytest.raises(ToolExecutionError):
        registry.call("get_current_time", {"timezone": "mars standard time"})


# --- ToolRegistry mechanics ---

def test_registry_describe_all_lists_every_tool():
    registry = build_demo_registry()
    described = registry.describe_all()
    names = {d["name"] for d in described}
    assert names == {"get_weather", "calculator", "search_knowledge_base", "get_current_time"}
    for d in described:
        assert "description" in d and "parameters" in d


def test_registry_list_names_and_has():
    registry = build_demo_registry()
    assert "get_weather" in registry.list_names()
    assert registry.has("get_weather")
    assert not registry.has("does_not_exist")


def test_registry_get_unknown_tool_raises():
    registry = build_demo_registry()
    with pytest.raises(UnknownToolError):
        registry.get("does_not_exist")


def test_registry_call_unknown_tool_raises():
    registry = build_demo_registry()
    with pytest.raises(UnknownToolError):
        registry.call("does_not_exist", {})


def test_registry_wrong_arguments_raise_tool_argument_error():
    """A syntactically valid tool call with the wrong argument name is a very
    plausible model failure mode — must surface as a HarnessError, not a raw
    TypeError that would escape the executor's error handling."""
    registry = build_demo_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("get_weather", {"location": "Paris"})  # wrong kwarg name


def test_registry_register_duplicate_name_raises():
    registry = ToolRegistry()
    spec = ToolSpec(name="dup", description="d", parameters_schema={}, fn=lambda: None)
    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)


def test_two_registries_from_factory_are_independent():
    a = build_demo_registry()
    b = build_demo_registry()
    a.register(ToolSpec(name="extra", description="d", parameters_schema={}, fn=lambda: None))
    assert a.has("extra")
    assert not b.has("extra")
