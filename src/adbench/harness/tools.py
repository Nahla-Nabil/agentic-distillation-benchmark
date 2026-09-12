"""Tool registry: the set of callable tools the harness exposes to a model
during a task, plus the schema each tool call is validated against.

Each tool is a plain Python function wrapped in a ToolSpec, registered here
so both the harness (to execute it) and the prompt-construction code (to
describe it to the model) share one definition — no drift between "what the
model is told" and "what actually runs".

Tools are deterministic and side-effect-free by design, so eval runs are
reproducible: e.g. get_weather() is backed by a fixed lookup table, not a
live API call.

The DEMO_REGISTRY built below is a small, self-contained tool set (6 tools,
each with a documented failure mode) used by this package's own unit tests
and as the tool vocabulary for tasks.py's synthetic multi-step eval tasks —
see tasks.py's module docstring for why eval uses this fixed vocabulary
(kept constant across chain lengths 1/3/5) rather than the real
glaive-function-calling-v2 data data/prepare.py produces (single-step only;
used for training, and optionally for a real-data chain_length=1 data point
via tasks.load_tasks(source="glaive_test") — see build_glaive_registry()
below for that tool set).

Originally 4 tools (get_weather, calculator, search_knowledge_base,
get_current_time); convert_temperature and compare_numbers were added
specifically because they consume a NUMBER as input, which lets a
multi-step task have a later step genuinely depend on an earlier step's
result (e.g. "check the weather, then convert that reading to Celsius") —
the original 4 tools all take place names/free-text/timezone-code arguments
that don't naturally chain from one tool's output to another's input. The
4-tool set was also numerically thin for 30-50 *distinct* multi-step
scenarios: only 4*3*2=24 length-3 permutations without repetition,
comfortably short of even the low end of that range at length 5. 6 tools
gives 6*5*4=120 length-3 and 6*5*4*3*2=720 length-5 permutations.
"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from adbench.harness.errors import (
    ToolArgumentError,
    ToolExecutionError,
    UnknownToolError,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    """Holds a set of ToolSpecs, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} is already registered.")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(f"No tool named {name!r} in registry.") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> list[str]:
        return list(self._tools)

    def describe_all(self) -> list[dict[str, Any]]:
        """Tool specs in a form suitable for the model's system prompt
        (name/description/schema), for injection into the chat template."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters_schema,
            }
            for spec in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Look up and execute a tool by name. Raises UnknownToolError if the
        name isn't registered, or whatever ToolExecutionError subclass the
        tool itself raises on bad input. A well-formed tool call with the
        wrong argument names/count (a very plausible model failure mode)
        raises a plain TypeError from the underlying function call — caught
        here and re-raised as ToolArgumentError so it's a HarnessError the
        executor's retry/error-handling path actually catches, instead of an
        uncaught TypeError blowing up the whole task."""
        spec = self.get(name)
        try:
            return spec.fn(**arguments)
        except TypeError as e:
            raise ToolArgumentError(f"Invalid arguments for tool {name!r}: {e}") from e


# --------------------------------------------------------------------------
# Demo tool set — deterministic, side-effect-free, each with one documented
# failure mode so the executor's error-handling path has something real to
# exercise in tests.
# --------------------------------------------------------------------------

WEATHER_TABLE = {
    "paris": {"temperature_f": 59, "condition": "cloudy"},
    "san francisco": {"temperature_f": 62, "condition": "foggy"},
    "tokyo": {"temperature_f": 71, "condition": "clear"},
    "cairo": {"temperature_f": 95, "condition": "sunny"},
    "london": {"temperature_f": 55, "condition": "rainy"},
    "berlin": {"temperature_f": 52, "condition": "overcast"},
    "sydney": {"temperature_f": 68, "condition": "sunny"},
    "mumbai": {"temperature_f": 88, "condition": "humid"},
    "toronto": {"temperature_f": 48, "condition": "windy"},
    "dubai": {"temperature_f": 102, "condition": "clear"},
}
# Public (not _-prefixed): tasks.py's synthetic task generator reads this
# directly to build weather-based scenarios, so the tool's real vocabulary
# and the eval tasks that exercise it can never drift apart.


def get_weather(city: str) -> dict[str, Any]:
    """Failure mode: unknown city -> ToolExecutionError."""
    key = city.strip().lower()
    if key not in WEATHER_TABLE:
        raise ToolExecutionError(f"No weather data for city {city!r}.")
    return {"city": city, **WEATHER_TABLE[key]}


_ALLOWED_CALC_CHARS = set("0123456789+-*/(). ")


def calculator(expression: str) -> dict[str, Any]:
    """Failure mode: invalid/unsafe expression, or div-by-zero -> ToolArgumentError."""
    if not expression or not set(expression) <= _ALLOWED_CALC_CHARS:
        raise ToolArgumentError(f"Expression contains disallowed characters: {expression!r}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ToolArgumentError(f"Division by zero in expression: {expression!r}") from None
    except SyntaxError:
        raise ToolArgumentError(f"Malformed expression: {expression!r}") from None
    return {"expression": expression, "result": result}


KNOWLEDGE_BASE = {
    "unsloth": "Unsloth is a library for fast, memory-efficient LLM fine-tuning.",
    "distillation": "Knowledge distillation trains a smaller student model to match a larger teacher's outputs.",
    "lora": "LoRA fine-tunes a frozen base model via small low-rank adapter matrices.",
    "quantization": "Quantization reduces model weight precision (e.g. 4-bit) to shrink memory footprint with minimal accuracy loss.",
    "perplexity": "Perplexity measures how well a language model predicts a sample of text; lower is better.",
    "tokenizer": "A tokenizer converts raw text into the subword tokens a language model actually operates on.",
    "fine-tuning": "Fine-tuning further trains a pretrained model on task-specific data to specialize its behavior.",
}


def search_knowledge_base(query: str) -> dict[str, Any]:
    """Failure mode: empty query, or no matches -> ToolExecutionError."""
    q = query.strip().lower()
    if not q:
        raise ToolArgumentError("Query must not be empty.")
    matches = {k: v for k, v in KNOWLEDGE_BASE.items() if q in k or q in v.lower()}
    if not matches:
        raise ToolExecutionError(f"No knowledge base entries matched {query!r}.")
    return {"query": query, "matches": matches}


TIME_TABLE = {
    "utc": "2026-09-08T12:00:00Z",
    "est": "2026-09-08T08:00:00-04:00",
    "jst": "2026-09-08T21:00:00+09:00",
    "pst": "2026-09-08T04:00:00-08:00",
    "cet": "2026-09-08T13:00:00+01:00",
    "ist": "2026-09-08T17:30:00+05:30",
    "aest": "2026-09-08T22:00:00+10:00",
    "gmt": "2026-09-08T12:00:00+00:00",
}


def get_current_time(timezone: str) -> dict[str, Any]:
    """Failure mode: unknown timezone -> ToolExecutionError.

    Deterministic fixed timestamps (not wall-clock time) so eval runs are
    reproducible.
    """
    key = timezone.strip().lower()
    if key not in TIME_TABLE:
        raise ToolExecutionError(f"Unknown timezone {timezone!r}.")
    return {"timezone": timezone, "current_time": TIME_TABLE[key]}


_TEMPERATURE_UNIT_ALIASES = {
    "f": "f", "fahrenheit": "f",
    "c": "c", "celsius": "c",
    "k": "k", "kelvin": "k",
}


def _temperature_to_celsius(value: float, unit: str) -> float:
    if unit == "f":
        return (value - 32) * 5 / 9
    if unit == "k":
        return value - 273.15
    return value


def _celsius_to(value_c: float, unit: str) -> float:
    if unit == "f":
        return value_c * 9 / 5 + 32
    if unit == "k":
        return value_c + 273.15
    return value_c


def convert_temperature(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Failure mode: unrecognized unit, or a value below absolute zero -> ToolArgumentError.

    Added specifically to be chainable: its input is a plain number, so a
    later step can genuinely consume an earlier step's numeric result (e.g.
    get_weather's temperature_f) rather than every tool only accepting
    free-text/place-name arguments that can't flow between tools.
    """
    f_unit = _TEMPERATURE_UNIT_ALIASES.get(from_unit.strip().lower())
    t_unit = _TEMPERATURE_UNIT_ALIASES.get(to_unit.strip().lower())
    if f_unit is None or t_unit is None:
        raise ToolArgumentError(f"Unknown temperature unit: {from_unit!r} or {to_unit!r} (use F, C, or K).")
    celsius = _temperature_to_celsius(value, f_unit)
    if celsius < -273.15:
        raise ToolArgumentError(f"{value}{from_unit} is below absolute zero.")
    converted = _celsius_to(celsius, t_unit)
    return {
        "value": value, "from_unit": from_unit, "to_unit": to_unit,
        "converted_value": round(converted, 2),
    }


def compare_numbers(a: float, b: float) -> dict[str, Any]:
    """Failure mode: non-finite input (NaN/infinity) -> ToolArgumentError.

    Added alongside convert_temperature for the same reason: a plain
    numeric tool that a later step can feed from two *different* earlier
    steps' results (e.g. two get_weather calls, or two calculator calls),
    demonstrating dependence on more than just the immediately preceding
    step.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ToolArgumentError(f"a and b must be finite numbers, got a={a!r}, b={b!r}.")
    if a == b:
        larger = "equal"
    else:
        larger = "a" if a > b else "b"
    return {"a": a, "b": b, "larger": larger, "difference": round(abs(a - b), 4)}


def build_demo_registry() -> ToolRegistry:
    """Factory so tests/tasks.py each get a fresh, independent registry
    rather than sharing mutable global state."""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        fn=get_weather,
    ))
    registry.register(ToolSpec(
        name="calculator",
        description="Evaluate a basic arithmetic expression (+ - * / and parentheses).",
        parameters_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        fn=calculator,
    ))
    registry.register(ToolSpec(
        name="search_knowledge_base",
        description="Search a small internal knowledge base for a query term.",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=search_knowledge_base,
    ))
    registry.register(ToolSpec(
        name="get_current_time",
        description="Get the current time in a given timezone (e.g. utc, est, jst, pst, cet, ist, aest, gmt).",
        parameters_schema={
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
        fn=get_current_time,
    ))
    registry.register(ToolSpec(
        name="convert_temperature",
        description="Convert a temperature value between Fahrenheit, Celsius, and Kelvin.",
        parameters_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        fn=convert_temperature,
    ))
    registry.register(ToolSpec(
        name="compare_numbers",
        description="Compare two numbers and report which is larger and by how much.",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        fn=compare_numbers,
    ))
    return registry


# Module-level default instance for convenience (e.g. interactive use);
# tests should prefer build_demo_registry() for isolation.
DEMO_REGISTRY = build_demo_registry()


# --------------------------------------------------------------------------
# Glaive-derived tool set — deterministic mock implementations of the 8 tool
# types selected from glaiveai/glaive-function-calling-v2 by
# data/prepare.py (see configs/data.yaml:subset.selected_tools). Kept
# separate from DEMO_REGISTRY: these argument names are pinned to each
# tool's *canonical* signature as found in the real dataset (the same tool
# name is called with inconsistent argument names across different raw
# examples — prepare.py drops anything that doesn't match the signature
# fixed here; see configs/data.yaml's header comment for the full writeup).
#
# All are pure calculation or small fixed lookup tables — no network I/O —
# and either always succeed for well-typed input (the calculation tools) or
# fail deterministically on a documented bad-input case, matching the demo
# tools' style above.
# --------------------------------------------------------------------------

_TODAY = date(2026, 9, 8)  # fixed reference "today", for reproducible eval runs


def _deterministic_hash_int(*parts: str) -> int:
    digest = hashlib.md5("|".join(parts).encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16)


def calculate_distance(origin: str, destination: str) -> dict[str, Any]:
    """Failure mode: empty origin/destination -> ToolArgumentError.

    No fixed lookup table — real inputs are arbitrary place names pulled
    from the dataset's user utterances, so this derives a deterministic
    pseudo-distance from the (order-independent) place-name pair instead of
    requiring an exhaustive table.
    """
    if not origin.strip() or not destination.strip():
        raise ToolArgumentError("origin and destination must be non-empty.")
    key = "|".join(sorted([origin.strip().lower(), destination.strip().lower()]))
    distance_km = 50 + (_deterministic_hash_int(key) % 4951)  # range: 50-5000
    return {"origin": origin, "destination": destination, "distance_km": distance_km}


_EXCHANGE_RATES_PER_USD = {
    "usd": 1.0, "eur": 0.92, "gbp": 0.79, "jpy": 147.0,
    "cad": 1.36, "aud": 1.52, "cny": 7.2, "inr": 83.0,
}

# The raw dataset doesn't consistently use ISO codes for currency VALUES
# even once the argument *keys* are canonicalized — e.g. real kept examples
# call this with from_currency="Euros" or "US dollars", not just "EUR"/"USD"
# (found by tests/test_prepare_real_output.py actually calling the tool
# against the real prepared data, not just checking the tool name exists).
# Normalize common natural-language names to a code before the rate lookup.
_CURRENCY_ALIASES = {
    "dollar": "usd", "dollars": "usd", "us dollar": "usd", "us dollars": "usd",
    "euro": "eur", "euros": "eur",
    "pound": "gbp", "pounds": "gbp", "british pound": "gbp", "british pounds": "gbp", "sterling": "gbp",
    "yen": "jpy", "japanese yen": "jpy",
    "canadian dollar": "cad", "canadian dollars": "cad",
    "australian dollar": "aud", "australian dollars": "aud",
    "yuan": "cny", "chinese yuan": "cny", "renminbi": "cny",
    "rupee": "inr", "rupees": "inr", "indian rupees": "inr",
}


def _resolve_currency_rate(currency: str) -> float:
    key = currency.strip().lower()
    key = _CURRENCY_ALIASES.get(key, key)
    if key in _EXCHANGE_RATES_PER_USD:
        return _EXCHANGE_RATES_PER_USD[key]
    # Unrecognized currency name: deterministic hash-derived rate rather than
    # failing, consistent with calculate_distance/get_stock_price's fallback
    # style — this tool's job is a plausible deterministic calculation, not
    # real-world FX accuracy.
    return 0.5 + (_deterministic_hash_int(key) % 150) / 100  # range: 0.5-2.0


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Failure mode: negative amount, or empty currency name -> ToolArgumentError."""
    if amount < 0:
        raise ToolArgumentError(f"amount must be non-negative, got {amount!r}.")
    if not from_currency.strip() or not to_currency.strip():
        raise ToolArgumentError("from_currency and to_currency must be non-empty.")
    converted = amount * (_resolve_currency_rate(to_currency) / _resolve_currency_rate(from_currency))
    return {
        "amount": amount, "from_currency": from_currency, "to_currency": to_currency,
        "converted_amount": round(converted, 2),
    }


_STOCK_PRICE_TABLE = {
    "aapl": 189.50, "msft": 415.20, "googl": 171.30,
    "amzn": 178.90, "tsla": 242.70, "nvda": 118.10, "meta": 502.30,
}


def get_stock_price(symbol: str) -> dict[str, Any]:
    """Failure mode: empty/non-alphabetic symbol -> ToolArgumentError.

    Known tickers use a fixed table; any other well-formed symbol falls back
    to a deterministic hash-derived price so arbitrary valid tickers from
    real data still succeed, rather than requiring an exhaustive table.
    """
    key = symbol.strip().lower()
    if not key or not key.isalpha():
        raise ToolArgumentError(f"symbol must be a non-empty alphabetic ticker, got {symbol!r}.")
    if key in _STOCK_PRICE_TABLE:
        price = _STOCK_PRICE_TABLE[key]
    else:
        price = round(10 + (_deterministic_hash_int(key) % 49000) / 100, 2)  # 10.00-500.00
    return {"symbol": symbol, "price": price}


def calculate_discount(original_price: float, discount_percentage: float) -> dict[str, Any]:
    """Failure mode: negative price, or percentage outside [0, 100] -> ToolArgumentError."""
    if original_price < 0:
        raise ToolArgumentError(f"original_price must be non-negative, got {original_price!r}.")
    if not 0 <= discount_percentage <= 100:
        raise ToolArgumentError(f"discount_percentage must be in [0, 100], got {discount_percentage!r}.")
    discounted_price = round(original_price * (1 - discount_percentage / 100), 2)
    return {
        "original_price": original_price, "discount_percentage": discount_percentage,
        "discounted_price": discounted_price,
    }


def calculate_bmi(height: float, weight: float) -> dict[str, Any]:
    """Failure mode: non-positive height/weight -> ToolArgumentError.

    height in meters, weight in kg (the dominant convention in the raw
    dataset's sample values, e.g. height=1.75, weight=70).
    """
    if height <= 0 or weight <= 0:
        raise ToolArgumentError(f"height and weight must be positive, got height={height!r}, weight={weight!r}.")
    bmi = round(weight / (height ** 2), 1)
    return {"height": height, "weight": weight, "bmi": bmi}


def calculate_tip(bill_amount: float, tip_percentage: float) -> dict[str, Any]:
    """Failure mode: negative bill_amount or tip_percentage -> ToolArgumentError."""
    if bill_amount < 0 or tip_percentage < 0:
        raise ToolArgumentError(
            f"bill_amount and tip_percentage must be non-negative, got "
            f"bill_amount={bill_amount!r}, tip_percentage={tip_percentage!r}."
        )
    tip_amount = round(bill_amount * tip_percentage / 100, 2)
    return {
        "bill_amount": bill_amount, "tip_percentage": tip_percentage,
        "tip_amount": tip_amount, "total_amount": round(bill_amount + tip_amount, 2),
    }


def calculate_age(birthdate: str) -> dict[str, Any]:
    """Failure mode: unparseable date, or a date after the fixed reference
    "today" -> ToolArgumentError.

    Age is computed relative to a fixed reference date (not wall-clock time)
    so eval runs are reproducible.
    """
    try:
        year, month, day = (int(p) for p in birthdate.strip().split("-"))
        birth = date(year, month, day)
    except (ValueError, AttributeError):
        raise ToolArgumentError(f"birthdate must be in YYYY-MM-DD format, got {birthdate!r}.") from None
    if birth > _TODAY:
        raise ToolArgumentError(f"birthdate {birthdate!r} is after the reference date {_TODAY.isoformat()}.")
    age_years = _TODAY.year - birth.year - ((_TODAY.month, _TODAY.day) < (birth.month, birth.day))
    return {"birthdate": birthdate, "age_years": age_years}


def generate_random_number(min: float, max: float) -> dict[str, Any]:
    """Failure mode: min >= max -> ToolArgumentError.

    Deterministic (hash-seeded by min/max), not true randomness, so eval
    runs are reproducible.
    """
    if min >= max:
        raise ToolArgumentError(f"min must be less than max, got min={min!r}, max={max!r}.")
    span = max - min
    fraction = (_deterministic_hash_int(str(min), str(max)) % 10_000_000) / 10_000_000
    value = min + fraction * span
    if isinstance(min, int) and isinstance(max, int):
        value = int(min + fraction * (span + 1))
        value = min if value > max else value
    return {"min": min, "max": max, "value": value}


def build_glaive_registry() -> ToolRegistry:
    """Factory for the 8-tool registry used to validate and (eventually) run
    the glaive-derived task set. Fresh instance per call, like
    build_demo_registry()."""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="calculate_distance",
        description="Calculate the distance between two places.",
        parameters_schema={
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["origin", "destination"],
        },
        fn=calculate_distance,
    ))
    registry.register(ToolSpec(
        name="convert_currency",
        description="Convert an amount from one currency to another.",
        parameters_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
        fn=convert_currency,
    ))
    registry.register(ToolSpec(
        name="get_stock_price",
        description="Get the current price of a stock by ticker symbol.",
        parameters_schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        fn=get_stock_price,
    ))
    registry.register(ToolSpec(
        name="calculate_discount",
        description="Calculate the discounted price given an original price and a discount percentage.",
        parameters_schema={
            "type": "object",
            "properties": {
                "original_price": {"type": "number"},
                "discount_percentage": {"type": "number"},
            },
            "required": ["original_price", "discount_percentage"],
        },
        fn=calculate_discount,
    ))
    registry.register(ToolSpec(
        name="calculate_bmi",
        description="Calculate body mass index from height (m) and weight (kg).",
        parameters_schema={
            "type": "object",
            "properties": {"height": {"type": "number"}, "weight": {"type": "number"}},
            "required": ["height", "weight"],
        },
        fn=calculate_bmi,
    ))
    registry.register(ToolSpec(
        name="calculate_tip",
        description="Calculate a tip amount and total given a bill amount and tip percentage.",
        parameters_schema={
            "type": "object",
            "properties": {
                "bill_amount": {"type": "number"},
                "tip_percentage": {"type": "number"},
            },
            "required": ["bill_amount", "tip_percentage"],
        },
        fn=calculate_tip,
    ))
    registry.register(ToolSpec(
        name="calculate_age",
        description="Calculate age in years from a birthdate (YYYY-MM-DD).",
        parameters_schema={
            "type": "object",
            "properties": {"birthdate": {"type": "string"}},
            "required": ["birthdate"],
        },
        fn=calculate_age,
    ))
    registry.register(ToolSpec(
        name="generate_random_number",
        description="Generate a number between min and max (inclusive).",
        parameters_schema={
            "type": "object",
            "properties": {"min": {"type": "number"}, "max": {"type": "number"}},
            "required": ["min", "max"],
        },
        fn=generate_random_number,
    ))
    return registry


GLAIVE_REGISTRY = build_glaive_registry()
