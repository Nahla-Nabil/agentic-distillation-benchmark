"""Sanity checks against the *actual* generated data/splits/*.jsonl — not
fabricated fixtures. These are the checks data/prepare.py's requirements
literally asked for: no duplicate examples between train/test, every tool
referenced actually exists in ToolRegistry, chain lengths are correctly
labeled.

Skipped (not failed) if the split hasn't been generated on this machine —
it's gitignored per data/README.md, generated locally by:
    python -m adbench.data.prepare
"""

from pathlib import Path

import pytest

from adbench.data.prepare import load_config, read_jsonl
from adbench.harness.tools import build_glaive_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "configs" / "data.yaml")
TRAIN_PATH = REPO_ROOT / CONFIG["output"]["train_path"]
TEST_PATH = REPO_ROOT / CONFIG["output"]["test_path"]

pytestmark = pytest.mark.skipif(
    not (TRAIN_PATH.exists() and TEST_PATH.exists()),
    reason="data/splits not generated on this machine — run `python -m adbench.data.prepare` first",
)


@pytest.fixture(scope="module")
def train_records():
    return read_jsonl(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_records():
    return read_jsonl(TEST_PATH)


def test_no_duplicate_examples_within_or_across_splits(train_records, test_records):
    train_ids = [r["source_index"] for r in train_records]
    test_ids = [r["source_index"] for r in test_records]
    assert len(train_ids) == len(set(train_ids)), "duplicate source_index within train"
    assert len(test_ids) == len(set(test_ids)), "duplicate source_index within test"
    assert set(train_ids).isdisjoint(test_ids), "an example appears in both train and test"


def test_every_referenced_tool_exists_in_registry(train_records, test_records):
    registry = build_glaive_registry()
    for record in train_records + test_records:
        for tool_name in record["expected_tool_sequence"]:
            assert registry.has(tool_name), (
                f"{record['task_id']!r} references {tool_name!r}, "
                "which is not in build_glaive_registry()"
            )


def test_chain_lengths_are_correctly_labeled(train_records, test_records):
    for record in train_records + test_records:
        assert record["chain_length"] == len(record["expected_tool_sequence"]), (
            f"{record['task_id']!r}: chain_length={record['chain_length']} but "
            f"expected_tool_sequence has {len(record['expected_tool_sequence'])} entries"
        )
        # This dataset is single-step end to end (see prepare.py's module
        # docstring) — every record prepare.py writes should be chain_length=1.
        assert record["chain_length"] == 1


def test_every_referenced_tool_is_actually_callable(train_records, test_records):
    """Not just present by name — callable with the exact arguments prepare.py
    kept for it (i.e. matching its canonical signature)."""
    registry = build_glaive_registry()
    for record in train_records + test_records:
        tool_name = record["expected_tool_sequence"][0]
        result = registry.call(tool_name, record["arguments"])
        assert isinstance(result, dict)


def test_total_size_within_configured_bounds(train_records, test_records):
    total = len(train_records) + len(test_records)
    assert CONFIG["subset"]["n_examples_min"] <= total <= CONFIG["subset"]["n_examples_max"]


def test_distinct_tool_count_within_configured_bounds(train_records, test_records):
    tools = {r["expected_tool_sequence"][0] for r in train_records + test_records}
    assert CONFIG["subset"]["min_distinct_tools"] <= len(tools) <= CONFIG["subset"]["max_distinct_tools"]
