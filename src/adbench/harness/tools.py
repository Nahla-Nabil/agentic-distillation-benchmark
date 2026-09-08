"""Tool registry: the set of callable tools the harness exposes to a model
during a task, plus the schema each tool call is validated against.

Each tool is a plain Python function wrapped in a ToolSpec, registered here
so both the harness (to execute it) and the prompt-construction code (to
describe it to the model) share one definition — no drift between "what the
model is told" and "what actually runs".

Tools are deterministic and side-effect-free by design, so eval runs are
reproducible: e.g. get_weather() is backed by a fixed lookup table, not a
live API call.

The DEMO_REGISTRY built below is a small, self-contained tool set (4 tools,
each with a documented failure mode) used by this package's own unit tests
and for harness development before the real dataset is wired in. Once
data/prepare.py has run, tasks.py's real task loader will build tool
vocabulary from the filtered glaive-function-calling-v2 subset instead —
DEMO_REGISTRY is not that; see tasks.py for the distinction.
"""

from collections.abc import Callable
from dataclasses import dataclass
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

_WEATHER_TABLE = {
    "paris": {"temperature_f": 59, "condition": "cloudy"},
    "san francisco": {"temperature_f": 62, "condition": "foggy"},
    "tokyo": {"temperature_f": 71, "condition": "clear"},
    "cairo": {"temperature_f": 95, "condition": "sunny"},
}


def get_weather(city: str) -> dict[str, Any]:
    """Failure mode: unknown city -> ToolExecutionError."""
    key = city.strip().lower()
    if key not in _WEATHER_TABLE:
        raise ToolExecutionError(f"No weather data for city {city!r}.")
    return {"city": city, **_WEATHER_TABLE[key]}


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


_KNOWLEDGE_BASE = {
    "unsloth": "Unsloth is a library for fast, memory-efficient LLM fine-tuning.",
    "distillation": "Knowledge distillation trains a smaller student model to match a larger teacher's outputs.",
    "lora": "LoRA fine-tunes a frozen base model via small low-rank adapter matrices.",
}


def search_knowledge_base(query: str) -> dict[str, Any]:
    """Failure mode: empty query, or no matches -> ToolExecutionError."""
    q = query.strip().lower()
    if not q:
        raise ToolArgumentError("Query must not be empty.")
    matches = {k: v for k, v in _KNOWLEDGE_BASE.items() if q in k or q in v.lower()}
    if not matches:
        raise ToolExecutionError(f"No knowledge base entries matched {query!r}.")
    return {"query": query, "matches": matches}


_TIME_TABLE = {
    "utc": "2026-09-08T12:00:00Z",
    "est": "2026-09-08T08:00:00-04:00",
    "jst": "2026-09-08T21:00:00+09:00",
}


def get_current_time(timezone: str) -> dict[str, Any]:
    """Failure mode: unknown timezone -> ToolExecutionError.

    Deterministic fixed timestamps (not wall-clock time) so eval runs are
    reproducible.
    """
    key = timezone.strip().lower()
    if key not in _TIME_TABLE:
        raise ToolExecutionError(f"Unknown timezone {timezone!r}.")
    return {"timezone": timezone, "current_time": _TIME_TABLE[key]}


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
        description="Get the current time in a given timezone (utc, est, or jst).",
        parameters_schema={
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
        fn=get_current_time,
    ))
    return registry


# Module-level default instance for convenience (e.g. interactive use);
# tests should prefer build_demo_registry() for isolation.
DEMO_REGISTRY = build_demo_registry()
