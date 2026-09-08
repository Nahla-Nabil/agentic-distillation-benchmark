"""Build the fixed 500-1000 example subset of glaiveai/glaive-function-calling-v2
used across all three conditions, and freeze its 80/20 train/test split.

Run once, locally (CPU-only, no GPU needed):
    python -m adbench.data.prepare --config configs/data.yaml

Pipeline (see configs/data.yaml's header comment for the "why" behind each
of these, discovered by inspecting the raw dataset):
  1. Download the raw dataset (network + `datasets`, only in main()/CLI use —
     everything else here is a pure function over a `chat` string, so it's
     unit-testable without network access; see tests/test_prepare.py).
  2. Parse every example's <functioncall> block (src/adbench/data/_glaive_parsing.py)
     and count every tool name that appears, at every frequency — this is
     the "log which tool names appear and their counts" the report prints,
     independent of which ones end up selected.
  3. Keep only examples that: parsed cleanly, contain exactly one function
     call (see configs/data.yaml — genuine multi-step chains are ~8 examples
     in the whole dataset, not enough to build a subset from), call one of
     configs/data.yaml's 8 selected tools, and use that tool's canonical
     argument-key signature (drops the long tail of inconsistent per-example
     naming for the same tool — see canonicalize_arguments()).
  4. Sample up to `per_tool_target` kept examples per tool (seeded), split
     80/20 per tool so the split stays balanced across tools, write to disk.
     Both files refuse to be overwritten without --force, so a re-run with a
     different seed can't silently swap what's in train vs test.
  5. Each written record is a superset of harness.tasks.TaskSpec's fields
     (task_id, chain_length, user_goal, expected_tool_sequence,
     injected_error_at_step) plus provenance (source_index, arguments,
     arguments_style) — so tasks.py's real load_tasks() can be a thin
     wrapper that just drops the extra fields.

NOT handled here (by design, not oversight):
  - Chain-length-3/5 eval tasks — this dataset is single-step; those must be
    COMPOSED from several of these chain_length=1 records. That composition
    is a task-construction decision, not dataset content, so it belongs in
    tasks.py's real load_tasks(), not here. Every record this module writes
    is chain_length=1.
  - The ~0.9% of <functioncall> blocks that fail to parse (unescaped
    apostrophes breaking the raw dataset's own single-quote string
    boundaries) — skipped and counted, not repaired.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from adbench.data._glaive_parsing import (
    FunctionCallParseError,
    extract_last_user_message_before,
    find_all_functioncalls,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_args_by_tool(config: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """{tool_name: sorted tuple of its required canonical argument keys},
    from configs/data.yaml:subset.selected_tools."""
    return {
        t["name"]: tuple(sorted(t["canonical_args"]))
        for t in config["subset"]["selected_tools"]
    }


def canonicalize_arguments(
    tool_name: str, arguments: dict[str, Any], canonical: dict[str, tuple[str, ...]]
) -> dict[str, Any] | None:
    """Returns `arguments` unchanged if its key signature matches the tool's
    canonical signature, else None (caller drops the example). Doesn't
    attempt to *rename* mismatched keys to the canonical set — the raw
    dataset gives no reliable signal for which alternate name maps to which
    canonical one (e.g. is "start_location" the same slot as "origin", or a
    different tool variant entirely?), so guessing would silently corrupt
    data. Dropping is the safe default; there is ample volume to drop from
    (see configs/data.yaml's per-tool counts)."""
    if tool_name not in canonical:
        return None
    if tuple(sorted(arguments.keys())) != canonical[tool_name]:
        return None
    return arguments


def process_parsed_example(
    calls: list[dict[str, Any]],
    chat: str,
    source_index: int,
    canonical: dict[str, tuple[str, ...]],
) -> tuple[dict[str, Any] | None, str]:
    """Same as process_raw_example() below, but takes already-parsed
    `calls` (find_all_functioncalls(chat)) — lets main() parse each chat
    exactly once and reuse it both for the raw tool-name frequency count and
    for this filtering step, instead of parsing every row twice."""
    if len(calls) == 0:
        return None, "zero_calls"
    if len(calls) > 1:
        return None, "multi_call"

    call = calls[0]
    if call["name"] not in canonical:
        return None, "tool_not_selected"

    arguments = canonicalize_arguments(call["name"], call["arguments"], canonical)
    if arguments is None:
        return None, "non_canonical_args"

    try:
        user_goal = extract_last_user_message_before(chat, call["_char_offset"])
    except FunctionCallParseError:
        return None, "no_user_turn"
    if not user_goal:
        return None, "empty_user_goal"

    record = {
        "task_id": f"glaive-{source_index}",
        "chain_length": 1,
        "user_goal": user_goal,
        "expected_tool_sequence": [call["name"]],
        "injected_error_at_step": None,
        "arguments": arguments,
        "arguments_style": call["arguments_style"],
        "source_index": source_index,
    }
    return record, "ok"


def process_raw_example(
    chat: str, source_index: int, canonical: dict[str, tuple[str, ...]]
) -> tuple[dict[str, Any] | None, str]:
    """Convert one raw dataset row's `chat` text into a processed record, or
    None with a reason if it's dropped. Reasons (for the prepare report):
    "parse_error", "zero_calls", "multi_call", "tool_not_selected",
    "non_canonical_args", "no_user_turn", "empty_user_goal", "ok".

    Pure function — no dataset/network access — so it's directly
    unit-testable against fabricated `chat` strings. main() uses the
    lower-level process_parsed_example() directly to avoid parsing each chat
    twice; this wrapper is what tests and any other one-off caller want.
    """
    try:
        calls = find_all_functioncalls(chat)
    except FunctionCallParseError:
        return None, "parse_error"
    return process_parsed_example(calls, chat, source_index, canonical)


def balance_and_split(
    records: list[dict[str, Any]],
    per_tool_target: int,
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cap each tool's pool at `per_tool_target` (seeded sample), split each
    tool's capped pool `train_fraction`/rest (so the split stays balanced
    across tools, not just overall), then seeded-shuffle each combined split
    so tool order isn't blocked. Deterministic for a fixed seed; disjoint by
    construction (each record is assigned to exactly one of the two lists).
    """
    rng = random.Random(seed)

    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_tool[r["expected_tool_sequence"][0]].append(r)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for tool_name in sorted(by_tool):
        pool = list(by_tool[tool_name])
        rng.shuffle(pool)
        capped = pool[:per_tool_target]
        n_train = round(len(capped) * train_fraction)
        train.extend(capped[:n_train])
        test.extend(capped[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve(config_relative_path: str) -> Path:
    """configs/data.yaml's output paths are relative to the repo root, not
    to wherever the script happens to be invoked from."""
    return REPO_ROOT / config_relative_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing train/test split instead of refusing to run.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Process only the first N raw rows (for a fast local dry run).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = _resolve(config["output"]["train_path"])
    test_path = _resolve(config["output"]["test_path"])
    stats_path = _resolve(config["output"]["stats_path"])

    if not args.force and (train_path.exists() or test_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite an existing split ({train_path} / {test_path}). "
            "The test split is frozen once written so train/test never leak across "
            "experiments — pass --force if you deliberately want to regenerate both."
        )

    from datasets import load_dataset  # deferred: only main() needs network/this dep

    print(f"Loading {config['source']['hf_dataset']} ...")
    ds = load_dataset(config["source"]["hf_dataset"], split="train")
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    print(f"Loaded {len(ds)} raw rows.")

    canonical = canonical_args_by_tool(config)

    all_tool_name_counter: Counter[str] = Counter()  # every tool name seen, any signature
    drop_reason_counter: Counter[str] = Counter()
    kept_records: list[dict[str, Any]] = []

    for i, row in enumerate(ds):
        chat = row["chat"]
        try:
            calls = find_all_functioncalls(chat)
        except FunctionCallParseError:
            drop_reason_counter["parse_error"] += 1
            continue

        for call in calls:
            all_tool_name_counter[call["name"]] += 1

        record, reason = process_parsed_example(calls, chat, i, canonical)
        drop_reason_counter[reason] += 1
        if record is not None:
            kept_records.append(record)

        if i % 20000 == 0:
            print(f"  ...{i}/{len(ds)}")

    per_tool_kept = Counter(r["expected_tool_sequence"][0] for r in kept_records)
    per_tool_target = config["subset"]["per_tool_target"]
    train_records, test_records = balance_and_split(
        kept_records,
        per_tool_target=per_tool_target,
        train_fraction=config["split"]["train_fraction"],
        seed=config["split"]["seed"],
    )
    total = len(train_records) + len(test_records)

    n_min = config["subset"]["n_examples_min"]
    n_max = config["subset"]["n_examples_max"]
    if not n_min <= total <= n_max:
        print(f"WARNING: final subset size {total} is outside the configured "
              f"[{n_min}, {n_max}] range.")

    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    stats = {
        "total_raw_rows": len(ds),
        "drop_reasons": dict(drop_reason_counter),
        "distinct_tool_names_in_raw_data": len(all_tool_name_counter),
        "top_40_tool_names_by_frequency": all_tool_name_counter.most_common(40),
        "selected_tools": sorted(canonical),
        "kept_examples_per_selected_tool_before_capping": {
            name: per_tool_kept.get(name, 0) for name in sorted(canonical)
        },
        "per_tool_target": per_tool_target,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_total": total,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=== prepare.py summary ===")
    print(f"raw rows processed: {len(ds)}")
    print(f"drop reasons: {dict(drop_reason_counter)}")
    print(f"distinct tool names seen in raw data: {len(all_tool_name_counter)}")
    print("kept examples per selected tool (before the per-tool cap):")
    for name in sorted(canonical):
        print(f"  {name}: {per_tool_kept.get(name, 0)}")
    print(f"train: {len(train_records)}   test: {len(test_records)}   total: {total}")
    print(f"wrote: {train_path}")
    print(f"wrote: {test_path}")
    print(f"wrote: {stats_path}")


if __name__ == "__main__":
    main()
