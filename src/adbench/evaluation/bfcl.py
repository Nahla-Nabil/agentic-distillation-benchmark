"""External benchmark subset: Berkeley Function-Calling Leaderboard (BFCL) v3, "simple"
category (huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) — one
user turn, exactly one function offered, exactly one correct call expected.

Every eval elsewhere in this package (harness/tasks.py's synthetic sets, the Glaive-derived
seen-tool set) is a task WE wrote, scored against a tool vocabulary WE built. That is fine for
the chain-length comparison (see tasks.py's module docstring), but it means every number in the
paper could in principle be an artefact of our own task design or prompt phrasing. BFCL is an
external, widely-used benchmark neither authored nor tuned by us: agreement between our
internal findings and BFCL numbers is evidence the findings aren't specific to our synthetic
tasks. It is NOT multi-step — a student trained/evaluated only on 1/3/5-step chains meeting a
single-turn, single-function-call benchmark is itself informative (does a benefit built for
chains transfer to the simplest possible case?).

Scoring is a deliberately SIMPLIFIED version of BFCL's own official checker: right function
name, every ground-truth parameter present with a value BFCL's own possible_answer file marks
acceptable (or validly omitted, when "" is one of that parameter's acceptable values — BFCL's
convention for "optional and left out is fine"), and no parameter outside what the ground truth
recognises. It does not replicate BFCL's full type-coercion/nested-structure rules, so a number
here is a reasonable proxy for exact-match accuracy, not a submittable leaderboard score — see
check_bfcl_call's docstring.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adbench.evaluation.batching import BatchedModelFn
from adbench.harness.errors import HarnessError
from adbench.harness.executor import FinalAnswer, ModelFn, ToolCall, _values_match, parse_model_output

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_CATEGORY = "simple"
TASK_SET_PREFIX = "bfcl_"


@dataclass(frozen=True)
class BFCLExample:
    example_id: str
    user_message: str
    functions: list[dict[str, Any]]                    # BFCL's own "function" list, native schema
    ground_truth: dict[str, dict[str, list[Any]]]       # {func_name: {param: [acceptable values]}}


def _read_jsonl_text(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_bfcl_examples(questions_text: str, answers_text: str) -> list[BFCLExample]:
    """Join BFCL's two parallel JSONL files (the question file and its possible_answer file,
    matched by "id") into one BFCLExample per row. Kept separate from the download so it's
    testable against small fixture strings, no network. Rows with no published ground truth,
    or whose ground truth isn't a single function call (defensive — "simple" should always be
    exactly one), are skipped rather than guessed at."""
    questions = _read_jsonl_text(questions_text)
    answers = {row["id"]: row for row in _read_jsonl_text(answers_text)}
    examples = []
    for q in questions:
        ans = answers.get(q["id"])
        if ans is None or len(ans["ground_truth"]) != 1:
            continue
        turns = q["question"][0]  # "question" is a list of turn-lists (multi-turn compatible); simple has one
        user_messages = [t["content"] for t in turns if t["role"] == "user"]
        examples.append(BFCLExample(
            example_id=q["id"],
            user_message="\n".join(user_messages),
            functions=q["function"],
            ground_truth=ans["ground_truth"][0],
        ))
    return examples


def pick_subset(examples: list[BFCLExample], n: int) -> list[BFCLExample]:
    """Deterministic, evenly spaced subset — same idea as scripts/check_batched_eval.py's
    pick_tasks — so a run's cost is bounded and reproducible run-to-run."""
    if n >= len(examples):
        return examples
    step = len(examples) / n
    return [examples[int(i * step)] for i in range(n)]


def download_bfcl_examples(
    n: int = 100, category: str = BFCL_CATEGORY, cache_dir: str | Path | None = None
) -> list[BFCLExample]:
    """Downloads BFCL_v3_<category>.json and possible_answer/BFCL_v3_<category>.json from the
    public gorilla-llm/Berkeley-Function-Calling-Leaderboard dataset repo (no auth token
    needed) and returns a deterministic n-example subset."""
    from huggingface_hub import hf_hub_download

    q_path = hf_hub_download(
        repo_id=BFCL_REPO, repo_type="dataset", filename=f"BFCL_v3_{category}.json", cache_dir=cache_dir
    )
    a_path = hf_hub_download(
        repo_id=BFCL_REPO, repo_type="dataset",
        filename=f"possible_answer/BFCL_v3_{category}.json", cache_dir=cache_dir,
    )
    examples = parse_bfcl_examples(Path(q_path).read_text(encoding="utf-8"), Path(a_path).read_text(encoding="utf-8"))
    return pick_subset(examples, n)


def build_bfcl_prompt(example: BFCLExample) -> list[dict[str, str]]:
    """One example's chat messages: a system prompt listing this example's own function(s), in
    BFCL's native JSON-schema dialect ("type": "dict", not adbench's own "type": "object") —
    kept as-is rather than rewritten into our harness convention, since testing against an
    external benchmark's own formatting is the point (see module docstring) — plus the user
    question. Mirrors harness.executor.build_system_prompt's protocol wording, so a model
    trained to answer in <tool_call>...</tool_call> style sees the same instruction it was
    trained on, only a different tool list."""
    tool_lines = [
        f"- {fn['name']}: {fn.get('description', '')} (parameters schema: {json.dumps(fn.get('parameters', {}))})"
        for fn in example.functions
    ]
    system = (
        "You are an assistant that can call tools to complete a task.\n"
        "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
        "To call a tool, respond with exactly one block of the form:\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": example.user_message}]


_MISSING = object()


def _value_in_acceptable(actual: Any, acceptable: list[Any]) -> bool:
    """`actual` matches one of `acceptable` (already stripped of any "" placeholder — that
    case is handled by check_bfcl_call). List-valued entries (e.g. an interval [1.0, 3.0]) are
    compared element-wise with the same numeric/string tolerance as a scalar
    (harness.executor._values_match)."""
    for want in acceptable:
        if isinstance(want, list) and isinstance(actual, list):
            if len(want) == len(actual) and all(_values_match(a, w) for a, w in zip(actual, want, strict=True)):
                return True
        elif not isinstance(want, list) and not isinstance(actual, list):
            if _values_match(actual, want):
                return True
    return False


def check_bfcl_call(call: ToolCall | FinalAnswer, ground_truth: dict[str, dict[str, list[Any]]]) -> bool:
    """Whether `call` matches BFCL's own multi-valued ground truth for a "simple"-category
    example: right function name, every ground-truth parameter present and matching (or
    validly omitted — "" is one of its acceptable values), and no parameter the ground truth
    doesn't recognise. A deliberately simplified stand-in for BFCL's own checker — see the
    module docstring."""
    if isinstance(call, FinalAnswer):
        return False
    if call.name not in ground_truth:
        return False
    expected_params = ground_truth[call.name]
    if not set(call.arguments) <= set(expected_params):
        return False
    for param, acceptable in expected_params.items():
        provided = call.arguments.get(param, _MISSING)
        omittable = any(a == "" for a in acceptable)
        if provided is _MISSING:
            if not omittable:
                return False
            continue
        if not _value_in_acceptable(provided, [a for a in acceptable if a != ""]):
            return False
    return True


def task_row(
    condition: str, example: BFCLExample, raw_output: str, category: str = BFCL_CATEGORY
) -> dict[str, Any]:
    """One example's result as a metrics.py-style row (same field shapes as
    evaluation/metrics.py's rows, so it can sit in the same eval_results.jsonl if wanted).
    `task_set` is "bfcl_<category>" (e.g. "bfcl_simple", "bfcl_multiple") so rows from different
    BFCL categories never get pooled together by accident. `error_type` is set only when the
    model's raw text wasn't even a well-formed tool call."""
    error_type = None
    try:
        parsed = parse_model_output(raw_output)
    except HarnessError as e:
        parsed = FinalAnswer(text=raw_output)
        error_type = type(e).__name__
    success = check_bfcl_call(parsed, example.ground_truth)
    return {
        "condition": condition,
        "task_set": f"{TASK_SET_PREFIX}{category}",
        "task_id": example.example_id,
        "success": success,
        "tool_name": parsed.name if isinstance(parsed, ToolCall) else None,
        "error_type": error_type,
    }


def run_bfcl_eval(
    condition: str, model_fn: ModelFn, examples: list[BFCLExample], category: str = BFCL_CATEGORY
) -> list[dict[str, Any]]:
    """Single call per example — BFCL "simple"/"multiple" are one-shot, not a multi-step chain,
    so this does not go through harness.executor.run_task. Runs concurrently if `model_fn` is a
    BatchedModelFn (see evaluation.batching), same convention as
    evaluation.run_eval.run_harness_eval_for_condition."""
    if isinstance(model_fn, BatchedModelFn):
        outputs = model_fn.run_concurrently(lambda ex: model_fn(build_bfcl_prompt(ex)), examples)
    else:
        outputs = [model_fn(build_bfcl_prompt(ex)) for ex in examples]
    return [task_row(condition, ex, raw, category) for ex, raw in zip(examples, outputs, strict=True)]


def summarize_bfcl(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Accuracy per condition, plus how many rows failed to even produce a well-formed tool
    call (error_type is not None)."""
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    return {
        condition: sum(r["success"] for r in crows) / len(crows)
        for condition, crows in by_condition.items()
    }
