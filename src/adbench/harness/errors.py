"""Error types the harness can inject and must handle correctly.

Two distinct failure surfaces matter for this study, and they stay
distinguishable in eval logs (see evaluation/metrics.py's
protocol_vs_execution_error_breakdown):

  1. Tool-execution errors (the tool itself fails: bad args, empty result,
     simulated timeout) — a competent agent should notice and recover
     (retry, pick a different tool, or ask for clarification).
  2. Model-protocol errors (malformed tool call, hallucinated tool name,
     ignoring the tool result entirely) — this is closer to what we expect
     to degrade under distillation, so it must be scored separately from #1.
"""


class HarnessError(Exception):
    """Base class for all harness-raised errors."""


# --- Tool-execution errors: the tool ran (or was invoked) but failed ---

class ToolExecutionError(HarnessError):
    """Tool ran but failed (bad args, empty/invalid result, timeout)."""


class ToolArgumentError(ToolExecutionError):
    """Tool was called with arguments it could not use (missing/invalid
    field, value outside what the tool supports)."""


class ToolTimeoutError(ToolExecutionError):
    """Tool call exceeded its allotted time (simulated, for harness testing
    of timeout-recovery behavior — no real tools do network I/O)."""


# --- Protocol errors: the model's output was not a valid step ---

class ProtocolError(HarnessError):
    """Model's output didn't constitute a valid tool call or valid handling
    of a prior tool result (wrong format, unknown tool, ignored result)."""


class MalformedCallError(ProtocolError):
    """Model emitted a tool-call block that isn't valid JSON, or is missing
    a required field (e.g. "name")."""


class UnknownToolError(ProtocolError):
    """Model called a tool name that isn't in the registry."""


class WrongToolError(ProtocolError):
    """Model called a real, registered tool — just not the one the task's
    expected_tool_sequence called for at this step."""
