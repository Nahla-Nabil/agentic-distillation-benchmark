"""Error types the harness can inject and must handle correctly.

Two distinct failure surfaces matter for this study, and they should stay
distinguishable in eval logs:
  1. Tool-execution errors (the tool itself fails: bad args, empty result,
     simulated timeout) — a competent agent should notice and recover
     (retry, pick a different tool, or ask for clarification).
  2. Model-protocol errors (malformed tool call, hallucinated tool name,
     ignoring the tool result entirely) — this is closer to what we expect
     to degrade under distillation, so it must be scored separately from #1.

TODO:
  - Define an exception hierarchy: ToolExecutionError, ToolArgumentError,
    ToolTimeoutError (for #1) vs ProtocolError, UnknownToolError,
    MalformedCallError (for #2).
  - Define what "recovery" means per error type so metrics.py can score
    recovered-after-error vs failed-after-error, not just terminal success.
"""


class HarnessError(Exception):
    """Base class for all harness-raised errors."""


class ToolExecutionError(HarnessError):
    """Tool ran but failed (bad args, empty/invalid result, timeout)."""


class ProtocolError(HarnessError):
    """Model's output didn't constitute a valid tool call or valid handling
    of a prior tool result (wrong format, unknown tool, ignored result)."""
