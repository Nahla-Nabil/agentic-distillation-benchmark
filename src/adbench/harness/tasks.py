"""Task/scenario definitions: the multi-step agentic tasks the harness runs,
grouped by chain length (1, 3, 5 — configs/experiment.yaml).

A task is a scripted scenario, not a free-form prompt: it specifies the user
goal, the sequence of tool calls a correct agent would make, and (for
inject_errors=True tasks) which step should trigger a tool failure so we can
score recovery behavior specifically.

TODO:
  - Define TaskSpec: id, chain_length, user_goal (str), expected_tool_sequence
    (list[str]), success_criteria (how run_eval.py judges the final state),
    optional injected_error_at_step (int | None).
  - Write a loader that builds TaskSpecs for each chain length out of the
    held-out test split (data/splits/test.jsonl) plus the tool registry —
    keeping eval tasks grounded in the same tool vocabulary the models were
    trained on, without ever training on the test split itself.
  - Chain-length-3 and -5 tasks likely need to be *composed* from several
    single-step examples (since glaive-function-calling-v2 is single-turn),
    stitched into one scenario — this composition logic belongs here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    chain_length: int
    user_goal: str
    expected_tool_sequence: list[str]
    injected_error_at_step: int | None = None


def load_tasks(chain_length: int) -> list[TaskSpec]:
    """TODO: build TaskSpecs for the given chain length from the held-out
    test split + tool registry."""
    raise NotImplementedError
