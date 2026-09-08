"""The execution loop: call tool -> read result -> decide next step ->
handle errors, repeated for a task's configured chain length (1, 3, or 5
steps per configs/experiment.yaml).

The model under test is injected rather than imported, so the exact same
loop drives the base student, the SFT-only student, the distilled student,
and (for layer analysis) the teacher:

    ModelFn = Callable[[list[dict]], str]
        Takes the running chat-message history, returns the model's raw
        text completion for the next turn (a tool call or a final answer).

TODO:
  - Define TaskState: task id, chain length, messages so far, tools called
    so far, per-step outcome (success / recovered / failed), terminal result.
  - Implement run_task(model_fn, task, registry) -> TaskState:
      loop up to task.chain_length steps:
        1. prompt model_fn with history + tool descriptions
        2. parse the completion as a tool call (or final answer)
           -> ProtocolError if unparseable / unknown tool
        3. execute the tool -> ToolExecutionError on failure
        4. append the tool result to history, continue
      stop early on: final answer produced, unrecoverable error, or
      max_retries_per_step exceeded on a single step.
  - Implement run_chain_suite(model_fn, tasks, registry) -> list[TaskState]
    for batches of tasks at a given chain length (used by evaluation/run_eval.py).
  - Error handling policy should follow configs/experiment.yaml's
    `max_retries_per_step` — retry the same step, don't just abort the task.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

ModelFn = Callable[[list[dict[str, Any]]], str]


@dataclass
class StepOutcome:
    step_index: int
    tool_name: str | None
    succeeded: bool
    recovered_from_error: bool = False
    error: str | None = None


@dataclass
class TaskState:
    task_id: str
    chain_length: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[StepOutcome] = field(default_factory=list)
    final_success: bool = False


def run_task(model_fn: ModelFn, task: Any, registry: Any) -> TaskState:
    """TODO: implement the call->read->decide->error-handle loop described
    above. Left unimplemented deliberately — this is the next piece to
    build once the tool registry and task definitions are in place."""
    raise NotImplementedError
