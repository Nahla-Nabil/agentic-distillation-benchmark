"""Scoring for harness runs.

Central metric: success rate by chain length, per condition — this is the
plot the whole research question hinges on (does the base/sft/distilled gap
widen as chain length goes 1 -> 3 -> 5?).

TODO:
  - success_rate(task_states: list[TaskState]) -> float
  - success_rate_by_chain_length(task_states) -> dict[int, float]
  - error_recovery_rate(task_states) -> float
      (of tasks with an injected error, fraction that still finished
      successfully — isolates robustness from raw task-completion skill)
  - protocol_vs_execution_error_breakdown(task_states) -> dict
      (see harness/errors.py's two error classes — keep them separate here)
  - Simple table/CSV writer so run_eval.py's output is easy to drop into the
    README or a notebook plot.
"""

from adbench.harness.executor import TaskState


def success_rate(task_states: list[TaskState]) -> float:
    raise NotImplementedError


def success_rate_by_chain_length(task_states: list[TaskState]) -> dict[int, float]:
    raise NotImplementedError
