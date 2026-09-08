"""Unit tests for the agentic harness (src/adbench/harness/). These run
locally in VS Code with `pytest` — no GPU, no model weights required, since
the harness is exercised with a scripted fake ModelFn rather than a real LLM.

TODO once harness/executor.py and harness/tools.py have real implementations:
  - test a scripted ModelFn that always emits the "correct" tool call ->
    run_task should report final_success=True with no error steps.
  - test a ModelFn that emits a malformed tool call -> expect a ProtocolError
    to be recorded on that step, not a silent skip.
  - test an injected tool failure (ToolExecutionError) -> expect a retry up
    to max_retries_per_step, then either recovery or a recorded failure.
  - test chain_length=1 vs 3 vs 5 task loading returns the right step counts.

Placeholder test below just checks the package imports cleanly, so `pytest`
is meaningful from day one of the repo.
"""

import adbench


def test_package_imports():
    assert adbench.__version__
