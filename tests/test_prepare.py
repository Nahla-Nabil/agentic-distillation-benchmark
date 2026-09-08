"""Unit tests for adbench.data.prepare's pure functions. None of these touch
the network or `datasets` — main() is the only part of prepare.py that does,
and it's intentionally thin (see prepare.py's module docstring) so everything
worth testing is testable without downloading anything.
"""

from pathlib import Path

import pytest
import yaml

from adbench.data.prepare import (
    balance_and_split,
    canonical_args_by_tool,
    canonicalize_arguments,
    load_config,
    process_raw_example,
    read_jsonl,
    write_jsonl,
)
from adbench.harness.tools import build_glaive_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"


# --- fixtures: a small hand-written canonical map, independent of the real
# config, so most tests aren't coupled to configs/data.yaml's exact contents ---

CANONICAL = {
    "calculate_bmi": ("height", "weight"),
    "calculate_tip": ("bill_amount", "tip_percentage"),
}


def _record(task_id: str, tool: str) -> dict:
    return {
        "task_id": task_id,
        "chain_length": 1,
        "user_goal": "goal",
        "expected_tool_sequence": [tool],
        "injected_error_at_step": None,
        "arguments": {},
        "arguments_style": "raw_object",
        "source_index": int(task_id.split("-")[-1]),
    }


# --- canonicalize_arguments ---

def test_canonicalize_matching_signature_kept():
    args = {"height": 1.75, "weight": 70}
    assert canonicalize_arguments("calculate_bmi", args, CANONICAL) == args


def test_canonicalize_mismatched_signature_dropped():
    args = {"height_cm": 175, "weight_kg": 70}
    assert canonicalize_arguments("calculate_bmi", args, CANONICAL) is None


def test_canonicalize_unselected_tool_dropped():
    assert canonicalize_arguments("some_other_tool", {"a": 1}, CANONICAL) is None


def test_canonicalize_key_order_does_not_matter():
    args = {"weight": 70, "height": 1.75}  # reversed insertion order
    assert canonicalize_arguments("calculate_bmi", args, CANONICAL) == args


# --- process_raw_example: one drop reason per branch ---

def test_process_raw_example_kept_ok():
    chat = (
        "USER: What's my BMI at 70kg and 1.75m?\n\n\n"
        'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": '
        "'{\"height\": 1.75, \"weight\": 70}'} <|endoftext|>"
    )
    record, reason = process_raw_example(chat, source_index=7, canonical=CANONICAL)
    assert reason == "ok"
    assert record["task_id"] == "glaive-7"
    assert record["chain_length"] == 1
    assert record["expected_tool_sequence"] == ["calculate_bmi"]
    assert record["user_goal"] == "What's my BMI at 70kg and 1.75m?"
    assert record["arguments"] == {"height": 1.75, "weight": 70}
    assert record["source_index"] == 7


def test_process_raw_example_parse_error():
    chat = (
        "USER: hi\n\n\n"
        'ASSISTANT: <functioncall> {"name": "x", "arguments": \'{not json and no closing quote} <|endoftext|>'
    )
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "parse_error"


def test_process_raw_example_zero_calls():
    chat = "USER: hi\n\n\nASSISTANT: hello there"
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "zero_calls"


def test_process_raw_example_multi_call():
    chat = (
        "USER: bmi and tip please\n\n\n"
        'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": \'{"height": 1.75, "weight": 70}\'} <|endoftext|>\n\n\n'
        'FUNCTION RESPONSE: {"bmi": 22.9}\n\n\n'
        'ASSISTANT: <functioncall> {"name": "calculate_tip", "arguments": \'{"bill_amount": 50, "tip_percentage": 15}\'} <|endoftext|>'
    )
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "multi_call"


def test_process_raw_example_tool_not_selected():
    chat = (
        "USER: weather?\n\n\n"
        'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": \'{"city": "Paris"}\'} <|endoftext|>'
    )
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "tool_not_selected"


def test_process_raw_example_non_canonical_args():
    chat = (
        "USER: bmi?\n\n\n"
        'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": \'{"height_cm": 175, "weight_kg": 70}\'} <|endoftext|>'
    )
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "non_canonical_args"


def test_process_raw_example_no_user_turn():
    chat = 'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": \'{"height": 1.75, "weight": 70}\'} <|endoftext|>'
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "no_user_turn"


def test_process_raw_example_empty_user_goal():
    chat = (
        "USER:   \n\n\n"
        'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": \'{"height": 1.75, "weight": 70}\'} <|endoftext|>'
    )
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert record is None
    assert reason == "empty_user_goal"


# --- balance_and_split ---

def test_balance_and_split_is_deterministic_for_a_fixed_seed():
    records = [_record(f"t-{i}", "calculate_bmi") for i in range(20)]
    train1, test1 = balance_and_split(records, per_tool_target=20, train_fraction=0.8, seed=42)
    train2, test2 = balance_and_split(records, per_tool_target=20, train_fraction=0.8, seed=42)
    assert [r["task_id"] for r in train1] == [r["task_id"] for r in train2]
    assert [r["task_id"] for r in test1] == [r["task_id"] for r in test2]


def test_balance_and_split_no_overlap_between_train_and_test():
    records = [_record(f"t-{i}", "calculate_bmi" if i % 2 == 0 else "calculate_tip") for i in range(40)]
    train, test = balance_and_split(records, per_tool_target=20, train_fraction=0.8, seed=1)
    train_ids = {r["task_id"] for r in train}
    test_ids = {r["task_id"] for r in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(train) + len(test)  # no internal dupes either


def test_balance_and_split_respects_train_fraction_per_tool():
    records = [_record(f"t-{i}", "calculate_bmi") for i in range(10)]
    train, test = balance_and_split(records, per_tool_target=10, train_fraction=0.8, seed=1)
    assert len(train) == 8
    assert len(test) == 2


def test_balance_and_split_caps_at_per_tool_target():
    records = [_record(f"t-{i}", "calculate_bmi") for i in range(100)]
    train, test = balance_and_split(records, per_tool_target=10, train_fraction=0.8, seed=1)
    assert len(train) + len(test) == 10


def test_balance_and_split_stratifies_across_tools():
    records = (
        [_record(f"bmi-{i}", "calculate_bmi") for i in range(10)]
        + [_record(f"tip-{i}", "calculate_tip") for i in range(10)]
    )
    train, test = balance_and_split(records, per_tool_target=10, train_fraction=0.8, seed=1)
    train_tools = [r["expected_tool_sequence"][0] for r in train]
    test_tools = [r["expected_tool_sequence"][0] for r in test]
    assert train_tools.count("calculate_bmi") == 8
    assert train_tools.count("calculate_tip") == 8
    assert test_tools.count("calculate_bmi") == 2
    assert test_tools.count("calculate_tip") == 2


def test_balance_and_split_handles_pool_smaller_than_target():
    records = [_record(f"t-{i}", "calculate_bmi") for i in range(3)]
    train, test = balance_and_split(records, per_tool_target=100, train_fraction=0.8, seed=1)
    assert len(train) + len(test) == 3


# --- write_jsonl / read_jsonl round-trip ---

def test_jsonl_round_trip(tmp_path):
    records = [_record(f"t-{i}", "calculate_bmi") for i in range(5)]
    path = tmp_path / "out.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded == records


# --- every chain_length metadata is correct for kept ("ok") records ---

def test_kept_record_chain_length_matches_tool_sequence_length():
    chat = (
        "USER: tip please\n\n\n"
        'ASSISTANT: <functioncall> {"name": "calculate_tip", "arguments": '
        "'{\"bill_amount\": 50, \"tip_percentage\": 15}'} <|endoftext|>"
    )
    record, reason = process_raw_example(chat, 0, CANONICAL)
    assert reason == "ok"
    assert record["chain_length"] == len(record["expected_tool_sequence"]) == 1


# --- integration with the real config + the real ToolRegistry (req #6) ---

def test_real_config_loads_and_every_selected_tool_exists_in_registry():
    config = load_config(DATA_CONFIG_PATH)
    canonical = canonical_args_by_tool(config)
    assert 5 <= len(canonical) <= 10  # configs/data.yaml:subset.min/max_distinct_tools

    registry = build_glaive_registry()
    for tool_name in canonical:
        assert registry.has(tool_name), f"{tool_name!r} is selected but not in build_glaive_registry()"


def test_real_config_selected_tools_have_no_duplicate_names():
    with open(DATA_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    names = [t["name"] for t in raw["subset"]["selected_tools"]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("tool_name,args", [
    ("calculate_distance", {"origin": "Paris", "destination": "Tokyo"}),
    ("convert_currency", {"amount": 100, "from_currency": "usd", "to_currency": "eur"}),
    ("get_stock_price", {"symbol": "aapl"}),
    ("calculate_discount", {"original_price": 100, "discount_percentage": 10}),
    ("calculate_bmi", {"height": 1.75, "weight": 70}),
    ("calculate_tip", {"bill_amount": 50, "tip_percentage": 15}),
    ("calculate_age", {"birthdate": "1990-05-15"}),
    ("generate_random_number", {"min": 1, "max": 100}),
])
def test_every_selected_tool_is_actually_callable_with_its_canonical_args(tool_name, args):
    """Not just registered by name — callable with exactly the argument keys
    configs/data.yaml declares as canonical for it."""
    registry = build_glaive_registry()
    result = registry.call(tool_name, args)
    assert isinstance(result, dict)
