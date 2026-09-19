"""Unit tests for evaluation/run_eval.py's testable core — config-driven
dispatch, the model_fn wiring, and the harness-run loop — using fake/
scripted models exactly like tests/test_train.py. load_condition_model is
the only function that needs Unsloth/a GPU/a real checkpoint and isn't
exercised here; everything it feeds into is.
"""

import json

import pytest
import torch

from adbench.evaluation.run_eval import (
    make_harness_model_fn,
    run_harness_eval_for_condition,
    write_eval_outputs,
)
from adbench.harness.tools import build_demo_registry
from adbench.training.train import REPO_ROOT, load_experiment_config

# --- make_harness_model_fn: fake model + tokenizer ---

class _StubBatchEncoding(dict):
    """Minimal stand-in for HF's BatchEncoding — just enough of its
    interface (dict-style item access + a `.to(device)` that's a no-op
    here) for make_harness_model_fn's `encoded["input_ids"]` /
    `.to(model.device)` calls."""

    def to(self, device):
        return self


class _StubTokenizer:
    """Same word-level pseudo-tokenizer pattern as tests/test_train.py, with
    a decode() that just joins ids back to their original words."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}

    def _token_id(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._reverse[idx] = word
        return self._vocab[word]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages]
        text = " ".join(parts)
        if add_generation_prompt:
            text += " <assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [self._token_id(w) for w in text.split()]
        if return_tensors == "pt":
            return _StubBatchEncoding(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            )
        return {"input_ids": ids}

    def decode(self, token_ids):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return " ".join(self._reverse[i] for i in token_ids)


class _StubGenerationConfig:
    max_length: int | None = 262144


class _FakeGenerateModel:
    """Fake model.generate(): ignores the actual prompt content and just
    appends a fixed, pre-tokenized completion after it — mimicking HF's
    convention of returning [prompt_tokens + new_tokens] concatenated."""

    def __init__(self, completion_ids: list[int]):
        self.completion_ids = completion_ids
        self.device = "cpu"
        self.generation_config = _StubGenerationConfig()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=256):
        completion = torch.tensor([self.completion_ids[:max_new_tokens]])
        return torch.cat([input_ids, completion], dim=1)


def test_make_harness_model_fn_returns_only_new_tokens():
    tokenizer = _StubTokenizer()
    # pre-register the completion's words so decode() can reverse them
    completion_text = '<tool_call>{"name":'
    completion_ids = [tokenizer._token_id(w) for w in completion_text.split()]
    model = _FakeGenerateModel(completion_ids)

    model_fn = make_harness_model_fn(model, tokenizer, max_new_tokens=10)
    result = model_fn([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi there"}])

    # the echoed-back prompt tokens must NOT appear in the result
    assert "sys" not in result
    assert "hi" not in result
    assert "there" not in result
    assert result == completion_text


def test_make_harness_model_fn_respects_max_new_tokens():
    tokenizer = _StubTokenizer()
    completion_ids = [tokenizer._token_id(w) for w in ["one", "two", "three", "four", "five"]]
    model = _FakeGenerateModel(completion_ids)

    model_fn = make_harness_model_fn(model, tokenizer, max_new_tokens=2)
    result = model_fn([{"role": "user", "content": "go"}])

    assert result == "one two"


# --- run_harness_eval_for_condition: real load_tasks + a correct scripted model ---

DEFAULT_ARGS_BY_TOOL = {
    "get_weather": {"city": "Paris"},
    "calculator": {"expression": "1 + 1"},
    "search_knowledge_base": {"query": "lora"},
    "get_current_time": {"timezone": "utc"},
    "convert_temperature": {"value": 100, "from_unit": "F", "to_unit": "C"},
    "compare_numbers": {"a": 5, "b": 3},
}


def _correct_model_fn(expected_sequence):
    def model_fn(messages):
        success_count = sum(
            1 for m in messages if m["role"] == "tool" and not m["content"].startswith("ERROR:")
        )
        name = expected_sequence[success_count]
        return f'<tool_call>{json.dumps({"name": name, "arguments": DEFAULT_ARGS_BY_TOOL[name]})}</tool_call>'
    return model_fn


def test_run_harness_eval_for_condition_produces_one_row_per_task():
    """Sanity check on task_state_to_row()/run_task() composed per-task
    (each given its own correctly-scripted model_fn) before the next test
    checks run_harness_eval_for_condition's own loop with a single,
    less-than-omniscient model_fn."""
    registry = build_demo_registry()
    from adbench.harness.tasks import build_synthetic_tasks
    expected_task_count = len(build_synthetic_tasks(1))

    tasks = build_synthetic_tasks(1)
    rows = []
    for task in tasks:
        model_fn = _correct_model_fn(task.expected_tool_sequence)
        from adbench.harness.executor import run_task
        state = run_task(model_fn, task, registry)
        from adbench.evaluation.metrics import task_state_to_row
        rows.append(task_state_to_row("base", task, state))

    assert len(rows) == expected_task_count
    assert all(r["condition"] == "base" for r in rows)
    assert all(r["success"] for r in rows)


def test_run_harness_eval_for_condition_integration_chain_length_one():
    """A real (not per-task-hand-rolled) call to
    run_harness_eval_for_condition, proving its own loop — not just
    run_task() in isolation — wires load_tasks -> run_task ->
    task_state_to_row correctly end to end.

    Uses a model_fn that only ever calls get_weather: build_system_prompt()
    describes the whole registry regardless of which task is running, so a
    model_fn with no other way to know which single tool a given
    chain_length=1 task actually wants can't be "universally correct" here
    by design — that's fine, since this test checks the function's wiring
    (row count, condition/chain_length tags, a real mix of success and
    WrongToolError failure), not full-suite success.
    """
    registry = build_demo_registry()

    def get_weather_only_model_fn(messages):
        return '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'

    rows = run_harness_eval_for_condition(
        "sft_only", get_weather_only_model_fn, chain_lengths=[1], registry=registry
    )

    from adbench.harness.tasks import build_synthetic_tasks
    assert len(rows) == len(build_synthetic_tasks(1))
    assert all(r["condition"] == "sft_only" for r in rows)
    assert all(r["chain_length"] == 1 for r in rows)
    assert any(r["success"] for r in rows)      # the weather-only tasks succeeded
    assert any(not r["success"] for r in rows)  # everything else correctly failed
    assert any(r["failure_error_type"] == "WrongToolError" for r in rows if not r["success"])


def test_run_harness_eval_for_condition_multiple_chain_lengths():
    registry = build_demo_registry()

    def model_fn(messages):
        return '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'

    rows = run_harness_eval_for_condition("base", model_fn, chain_lengths=[1, 3], registry=registry)

    from adbench.harness.tasks import build_synthetic_tasks
    expected = len(build_synthetic_tasks(1)) + len(build_synthetic_tasks(3))
    assert len(rows) == expected
    assert {r["chain_length"] for r in rows} == {1, 3}


# --- write_eval_outputs ---

def _sample_rows():
    return [
        {
            "condition": "base", "chain_length": 1, "task_id": "t1", "success": True,
            "num_steps_attempted": 1, "injected_error_at_step": None,
            "failure_step_index": None, "failure_error_type": None,
            "steps": [{"step_index": 0, "tool_name": "get_weather", "succeeded": True,
                       "recovered_from_error": False, "error": None, "error_type": None}],
        },
        {
            "condition": "sft_only", "chain_length": 1, "task_id": "t2", "success": False,
            "num_steps_attempted": 1, "injected_error_at_step": None,
            "failure_step_index": 0, "failure_error_type": "WrongToolError",
            "steps": [{"step_index": 0, "tool_name": "calculator", "succeeded": False,
                       "recovered_from_error": False, "error": "bad", "error_type": "WrongToolError"}],
        },
    ]


def test_write_eval_outputs_creates_all_three_files(tmp_path):
    output = write_eval_outputs(_sample_rows(), {"base": 15.2, "sft_only": 18.7}, tmp_path)

    assert (tmp_path / "eval_results.jsonl").exists()
    assert (tmp_path / "eval_results.csv").exists()
    assert (tmp_path / "eval_summary.json").exists()

    assert output["perplexity_by_condition"] == {"base": 15.2, "sft_only": 18.7}
    conditions_in_summary = {e["condition"] for e in output["summary"]}
    assert conditions_in_summary == {"base", "sft_only"}


def test_write_eval_outputs_summary_includes_perplexity_per_entry(tmp_path):
    output = write_eval_outputs(_sample_rows(), {"base": 15.2, "sft_only": 18.7}, tmp_path)
    for entry in output["summary"]:
        assert entry["perplexity"] == pytest.approx({"base": 15.2, "sft_only": 18.7}[entry["condition"]])


def test_write_eval_outputs_jsonl_round_trips(tmp_path):
    from adbench.evaluation.metrics import read_results_jsonl
    rows = _sample_rows()
    write_eval_outputs(rows, {"base": 1.0, "sft_only": 1.0}, tmp_path)
    assert read_results_jsonl(tmp_path / "eval_results.jsonl") == rows


def test_write_eval_outputs_summary_json_is_valid_json(tmp_path):
    write_eval_outputs(_sample_rows(), {"base": 1.0, "sft_only": 1.0}, tmp_path)
    with open(tmp_path / "eval_summary.json", encoding="utf-8") as f:
        json.load(f)  # must not raise


# --- config sanity: run_eval.py reads the real config's harness block ---

def test_real_experiment_config_has_harness_chain_lengths_and_retries():
    config = load_experiment_config(REPO_ROOT / "configs" / "experiment.yaml")
    assert config["harness"]["chain_lengths"] == [1, 3, 5]
    assert isinstance(config["harness"]["max_retries_per_step"], int)


# --- v2: greedy decoding, seen-tool evaluation ---

def test_make_harness_model_fn_switches_the_model_to_greedy_decoding():
    tokenizer = _StubTokenizer()
    model = _FakeGenerateModel([tokenizer._token_id("x")])
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20

    make_harness_model_fn(model, tokenizer)

    cfg = model.generation_config
    assert cfg.do_sample is False
    assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None


def _seen_tool_tasks(n):
    from adbench.harness.tasks import TaskSpec
    return [TaskSpec(task_id=f"g-{i}", chain_length=1, user_goal="What is my BMI?",
                     expected_tool_sequence=["calculate_bmi"]) for i in range(n)]


def test_run_seen_tool_eval_tags_rows_reads_glaive_test_and_limits_tasks(monkeypatch):
    from adbench.evaluation import run_eval

    requested = {}

    def fake_load_tasks(chain_length, source="synthetic", config_path=None):
        requested["source"] = source
        return _seen_tool_tasks(5)

    monkeypatch.setattr(run_eval, "load_tasks", fake_load_tasks)
    correct_call = '<tool_call>{"name": "calculate_bmi", "arguments": {"height": 1.75, "weight": 70}}</tool_call>'

    rows = run_eval.run_seen_tool_eval_for_condition("base", lambda messages: correct_call, n_tasks=3)

    assert requested["source"] == "glaive_test"
    assert len(rows) == 3
    assert {r["task_set"] for r in rows} == {"seen_tools"}
    assert all(r["success"] for r in rows)


def test_run_seen_tool_eval_is_skipped_when_n_is_zero(monkeypatch):
    from adbench.evaluation import run_eval

    def must_not_load(*args, **kwargs):
        raise AssertionError("tasks should not be loaded when the seen-tool eval is disabled")

    monkeypatch.setattr(run_eval, "load_tasks", must_not_load)
    assert run_eval.run_seen_tool_eval_for_condition("base", lambda m: "", n_tasks=0) == []


def test_real_experiment_config_defines_a_seen_tool_eval_size():
    config = load_experiment_config(REPO_ROOT / "configs" / "experiment.yaml")
    assert isinstance(config["harness"]["seen_tool_eval_n"], int)


def test_evaluate_seen_only_loads_one_model_and_runs_only_seen_tasks(monkeypatch):
    from adbench.evaluation import run_eval

    monkeypatch.setattr(run_eval, "load_condition_model", lambda c, e, m: (_FakeGenerateModel([1]), _StubTokenizer()))
    seen_calls = {}

    def fake_seen(condition, model_fn, n_tasks, max_retries):
        seen_calls.update(condition=condition, n=n_tasks, retries=max_retries)
        return [{"condition": condition, "task_set": "seen_tools"}]

    monkeypatch.setattr(run_eval, "run_seen_tool_eval_for_condition", fake_seen)
    config = {"harness": {"seen_tool_eval_n": 7, "max_retries_per_step": 2}}

    rows = run_eval.evaluate_seen_only("sft_early", config, {})

    assert rows == [{"condition": "sft_early", "task_set": "seen_tools"}]
    assert seen_calls == {"condition": "sft_early", "n": 7, "retries": 2}
