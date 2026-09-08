import pytest

from adbench.harness.errors import (
    ToolArgumentError,
    ToolExecutionError,
    UnknownToolError,
)
from adbench.harness.tools import (
    ToolRegistry,
    ToolSpec,
    build_demo_registry,
    build_glaive_registry,
)

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


# --------------------------------------------------------------------------
# Glaive-derived tool set (build_glaive_registry) — pinned to the canonical
# argument names in configs/data.yaml, and each deterministic (same input ->
# same output every time, since eval runs must be reproducible).
# --------------------------------------------------------------------------

def test_glaive_registry_has_all_eight_tools():
    registry = build_glaive_registry()
    assert set(registry.list_names()) == {
        "calculate_distance", "convert_currency", "get_stock_price",
        "calculate_discount", "calculate_bmi", "calculate_tip",
        "calculate_age", "generate_random_number",
    }


def test_calculate_distance_is_deterministic_and_symmetric():
    registry = build_glaive_registry()
    a = registry.call("calculate_distance", {"origin": "Paris", "destination": "Tokyo"})
    b = registry.call("calculate_distance", {"origin": "Paris", "destination": "Tokyo"})
    c = registry.call("calculate_distance", {"origin": "Tokyo", "destination": "Paris"})
    assert a["distance_km"] == b["distance_km"] == c["distance_km"]


def test_calculate_distance_empty_place_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculate_distance", {"origin": "  ", "destination": "Tokyo"})


def test_convert_currency_iso_codes():
    registry = build_glaive_registry()
    result = registry.call("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "USD"})
    assert result["converted_amount"] == 100.0


def test_convert_currency_natural_language_names_resolve():
    """Real dataset examples call this with values like "Euros" / "US
    dollars", not just ISO codes — found by actually running the tool
    against the real prepared data, not just checking the tool name exists."""
    registry = build_glaive_registry()
    result = registry.call("convert_currency", {"amount": 100, "from_currency": "US dollars", "to_currency": "Euros"})
    assert result["converted_amount"] > 0


def test_convert_currency_unknown_name_falls_back_instead_of_erroring():
    registry = build_glaive_registry()
    result = registry.call("convert_currency", {"amount": 100, "from_currency": "Dogecoin", "to_currency": "USD"})
    assert isinstance(result["converted_amount"], float)


def test_convert_currency_negative_amount_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("convert_currency", {"amount": -5, "from_currency": "USD", "to_currency": "EUR"})


def test_get_stock_price_known_ticker():
    registry = build_glaive_registry()
    result = registry.call("get_stock_price", {"symbol": "AAPL"})
    assert result["price"] > 0


def test_get_stock_price_unknown_ticker_falls_back_instead_of_erroring():
    registry = build_glaive_registry()
    result = registry.call("get_stock_price", {"symbol": "ZZZZ"})
    assert result["price"] > 0


def test_get_stock_price_invalid_symbol_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("get_stock_price", {"symbol": "123"})


def test_calculate_discount_formula():
    registry = build_glaive_registry()
    result = registry.call("calculate_discount", {"original_price": 100, "discount_percentage": 20})
    assert result["discounted_price"] == 80.0


def test_calculate_discount_out_of_range_percentage_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculate_discount", {"original_price": 100, "discount_percentage": 150})


def test_calculate_bmi_formula():
    registry = build_glaive_registry()
    result = registry.call("calculate_bmi", {"height": 2.0, "weight": 80})
    assert result["bmi"] == 20.0


def test_calculate_bmi_non_positive_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculate_bmi", {"height": 0, "weight": 80})


def test_calculate_tip_formula():
    registry = build_glaive_registry()
    result = registry.call("calculate_tip", {"bill_amount": 50, "tip_percentage": 20})
    assert result["tip_amount"] == 10.0
    assert result["total_amount"] == 60.0


def test_calculate_tip_negative_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculate_tip", {"bill_amount": -1, "tip_percentage": 20})


def test_calculate_age_formula():
    registry = build_glaive_registry()
    result = registry.call("calculate_age", {"birthdate": "1990-05-15"})
    assert result["age_years"] == 36  # relative to the fixed reference date 2026-09-08


def test_calculate_age_future_date_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculate_age", {"birthdate": "2099-01-01"})


def test_calculate_age_bad_format_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("calculate_age", {"birthdate": "05/15/1990"})


def test_generate_random_number_within_bounds_and_deterministic():
    registry = build_glaive_registry()
    a = registry.call("generate_random_number", {"min": 1, "max": 100})
    b = registry.call("generate_random_number", {"min": 1, "max": 100})
    assert a["value"] == b["value"]
    assert 1 <= a["value"] <= 100


def test_generate_random_number_min_gte_max_raises():
    registry = build_glaive_registry()
    with pytest.raises(ToolArgumentError):
        registry.call("generate_random_number", {"min": 100, "max": 1})
