"""Mini agentic harness: tool registry, execution loop, error handling, and
cross-step state tracking. Pure Python — no model-loading code here, so it is
fully unit-testable on the local CPU machine (see tests/test_harness.py).

A model is injected as a callable satisfying the ModelFn protocol in
executor.py; the harness itself is model-agnostic (base / SFT / distilled
student, or the teacher, all plug in the same way).
"""
