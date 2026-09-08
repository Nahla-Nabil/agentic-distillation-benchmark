"""Tool registry: the fixed set of callable tools the harness exposes to a
model during a task, plus the schema each tool call is validated against.

Design intent (fill in once configs/data.yaml's `selected_tools` is final):
  - Each tool is a plain Python function with a JSON-schema-describable
    signature, registered here so both the harness (to execute it) and the
    prompt-construction code (to describe it to the model) share one
    definition — no drift between "what the model is told" and "what
    actually runs".
  - Tools should be deterministic and side-effect-free (or mocked) so eval
    runs are reproducible: e.g. a `get_weather(city)` backed by a fixed
    lookup table, not a live API call.
  - A handful of tools should have documented failure modes (bad arg, empty
    result, simulated timeout) so errors.py has something real to catch —
    this is what lets the harness exercise its error-handling path.

TODO:
  - Define ToolSpec (name, description, json_schema, fn) and a Registry class.
  - Port ~5-10 tool types matching the function names found in the filtered
    glaive-function-calling-v2 subset, so training data and eval tasks speak
    the same tool vocabulary.
  - Add a `register(tool_spec)` / `get(name)` / `describe_all()` API.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """Placeholder shape — refine once the first real tools are ported in."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    """Placeholder. TODO: implement register/get/describe_all."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        raise NotImplementedError

    def get(self, name: str) -> ToolSpec:
        raise NotImplementedError

    def describe_all(self) -> list[dict[str, Any]]:
        """Return tool specs in a form suitable for the model's system prompt
        (name/description/schema), for injection into the chat template."""
        raise NotImplementedError
