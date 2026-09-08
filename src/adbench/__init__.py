"""adbench — Agentic Distillation Benchmark.

Research question: when a smaller student model is distilled from a larger
teacher, does its reliability on multi-step agentic tool-use degrade faster
than its general language modeling performance — and if so, at which layers
does that gap originate?

Subpackages:
    harness     — deterministic multi-step tool-use execution engine (no model
                  code lives here; it just calls whatever model you hand it).
    data        — dataset filtering/splitting (glaive-function-calling-v2).
    training    — SFT-only and KD+SFT distillation training scripts (Colab/GPU).
    evaluation  — runs all three conditions through the harness, scores them.
    analysis    — teacher/student activation comparison utilities.
"""

__version__ = "0.1.0"
