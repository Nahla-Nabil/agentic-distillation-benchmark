"""Verify that the teacher and student share a tokenizer/vocab before relying
on logit-level KD in src/adbench/training/losses.py.

Only downloads tokenizer files (a few KB), not model weights. Needs
`transformers` (no torch required just to load a tokenizer) — not part of
requirements.txt's core set since it's a one-off check, not a runtime
dependency of anything else in adbench yet.

    pip install transformers
    python scripts/verify_tokenizer_compatibility.py

RESULT (checked 2026-09, both repos' tokenizer configs as published on HF):
    Qwen/Qwen3-14B and Qwen/Qwen3-4B-Instruct-2507 use the identical
    tokenizer: Qwen2Tokenizer, vocab_size=151643 (len=151669 incl. added
    special tokens), same eos/pad tokens, and identical token-id sequences
    across code, JSON tool-call syntax, and CJK sample strings.
    => logit-level KD (src/adbench/training/losses.py::combined_loss) is
       safe to use as implemented. See that module's docstring for the
       sequence-level-distillation fallback kept in reserve in case a future
       teacher/student pairing does NOT share a tokenizer.
"""

from transformers import AutoTokenizer

TEACHER = "Qwen/Qwen3-14B"
STUDENT = "Qwen/Qwen3-4B-Instruct-2507"

SAMPLES = [
    "Hello, world!",
    "def call_tool(name, **kwargs):\n    return execute(name, kwargs)",
    "The weather in San Francisco is 62F and foggy.",
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
    "分布式蒸馏基准测试",  # non-ASCII sample — tokenizer divergence often shows here first
]

SPECIAL_FIELDS = [
    "bos_token", "eos_token", "pad_token", "unk_token",
    "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
]


def check_tokenizer_compatibility(teacher_id: str = TEACHER, student_id: str = STUDENT) -> bool:
    """Returns True iff vocab size, special tokens, and sample encodings all
    match. Prints a human-readable report either way."""
    tok_t = AutoTokenizer.from_pretrained(teacher_id)
    tok_s = AutoTokenizer.from_pretrained(student_id)

    print(f"teacher vocab_size: {tok_t.vocab_size}  (len(tokenizer)={len(tok_t)})")
    print(f"student vocab_size: {tok_s.vocab_size}  (len(tokenizer)={len(tok_s)})")

    vocab_match = tok_t.vocab_size == tok_s.vocab_size and len(tok_t) == len(tok_s)

    print("\nspecial tokens:")
    specials_match = True
    for f in SPECIAL_FIELDS:
        vt, vs = getattr(tok_t, f, None), getattr(tok_s, f, None)
        specials_match &= vt == vs
        print(f"  {f:16s} teacher={vt!r:20s} student={vs!r:20s} [{'OK' if vt == vs else 'MISMATCH'}]")

    print("\nsample string -> token ids:")
    samples_match = True
    for s in SAMPLES:
        ids_t = tok_t.encode(s, add_special_tokens=False)
        ids_s = tok_s.encode(s, add_special_tokens=False)
        match = ids_t == ids_s
        samples_match &= match
        print(f"  {s[:50]!r} match={match}")

    all_match = vocab_match and specials_match and samples_match
    print(f"\n=> {'SHARED TOKENIZER (logit-level KD OK)' if all_match else 'TOKENIZERS DIFFER (use sequence-level distillation fallback)'}")
    return all_match


if __name__ == "__main__":
    check_tokenizer_compatibility()
