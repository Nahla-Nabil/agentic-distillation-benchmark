"""Tests for evaluation/batching.py with a scripted generate_batch — no GPU."""

import json
import threading
import time

import pytest

from adbench.evaluation.batching import BatchedModelFn, configure_greedy
from adbench.evaluation.run_eval import run_harness_eval_for_condition
from adbench.harness.tasks import TaskSpec, load_tasks

ARGS = {
    "get_weather": {"city": "Paris"},
    "calculator": {"expression": "1 + 1"},
    "search_knowledge_base": {"query": "lora"},
    "get_current_time": {"timezone": "utc"},
    "convert_temperature": {"value": 100, "from_unit": "F", "to_unit": "C"},
    "compare_numbers": {"a": 5, "b": 3},
}


def _next_call(messages):
    """The correct next tool call for a task whose user goal is 'tool_a|tool_b|...'."""
    sequence = messages[1]["content"].split("|")
    done = sum(1 for m in messages if m["role"] == "tool" and not m["content"].startswith("ERROR:"))
    name = sequence[done]
    return f'<tool_call>{json.dumps({"name": name, "arguments": ARGS[name]})}</tool_call>'


def _task(i, sequence):
    return TaskSpec(task_id=f"t{i}", chain_length=len(sequence), user_goal="|".join(sequence),
                    expected_tool_sequence=list(sequence))


SEQUENCES = [["get_weather"], ["get_weather", "calculator"], ["calculator", "compare_numbers", "get_current_time"],
             ["search_knowledge_base"], ["get_weather", "convert_temperature", "compare_numbers", "calculator", "get_current_time"]]


def _make(max_batch, sizes=None, delay=0.0):
    def generate_batch(batch_messages):
        if sizes is not None:
            sizes.append(len(batch_messages))
        if delay:
            time.sleep(delay)
        return [_next_call(m) for m in batch_messages]

    return BatchedModelFn(generate_batch, max_batch=max_batch, max_wait_s=0.5)


def test_batched_run_gives_each_caller_its_own_answer_in_input_order():
    tasks = [_task(i, SEQUENCES[i % len(SEQUENCES)]) for i in range(11)]
    batched = _make(max_batch=4)
    try:
        from adbench.harness.executor import run_task
        from adbench.harness.tools import build_demo_registry

        registry = build_demo_registry()
        states = batched.run_concurrently(lambda t: run_task(batched, t, registry), tasks)
    finally:
        batched.close()

    assert [s.task_id for s in states] == [t.task_id for t in tasks]
    assert all(s.final_success for s in states)
    assert [len(s.steps) for s in states] == [t.chain_length for t in tasks]


def test_requests_really_are_batched_and_never_exceed_max_batch():
    sizes = []
    tasks = [_task(i, ["get_weather", "calculator", "compare_numbers"]) for i in range(8)]
    batched = _make(max_batch=4, sizes=sizes, delay=0.01)
    try:
        from adbench.harness.executor import run_task
        from adbench.harness.tools import build_demo_registry

        registry = build_demo_registry()
        batched.run_concurrently(lambda t: run_task(batched, t, registry), tasks)
    finally:
        batched.close()

    assert max(sizes) > 1
    assert max(sizes) <= 4
    assert sum(sizes) == 8 * 3          # one request per step, none lost or duplicated


def test_a_small_run_does_not_wait_for_a_full_batch():
    tasks = [_task(0, ["get_weather"]), _task(1, ["calculator"])]
    batched = _make(max_batch=64)       # far more slots than tasks
    start = time.monotonic()
    try:
        from adbench.harness.executor import run_task
        from adbench.harness.tools import build_demo_registry

        registry = build_demo_registry()
        batched.run_concurrently(lambda t: run_task(batched, t, registry), tasks)
    finally:
        batched.close()
    assert time.monotonic() - start < 0.4    # would be >= 0.5 s if it waited out max_wait_s


def test_batched_rows_equal_sequential_rows(monkeypatch):
    from adbench.evaluation import run_eval

    tasks = [_task(i, SEQUENCES[i % len(SEQUENCES)]) for i in range(9)]
    monkeypatch.setattr(run_eval, "load_tasks", lambda chain_length, source="synthetic", config_path=None: tasks)

    sequential = run_harness_eval_for_condition("base", _next_call, [1])
    batched_fn = _make(max_batch=4)
    try:
        batched = run_harness_eval_for_condition("base", batched_fn, [1])
    finally:
        batched_fn.close()

    assert batched == sequential


def test_a_failing_generate_batch_releases_every_waiting_caller_with_the_error():
    def broken(batch_messages):
        raise RuntimeError("CUDA out of memory")

    batched = BatchedModelFn(broken, max_batch=4)
    errors = []

    def call():
        try:
            batched([{"role": "system", "content": "s"}, {"role": "user", "content": "get_weather"}])
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=call) for _ in range(3)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
    finally:
        batched.close()
    assert errors == ["CUDA out of memory"] * 3


def test_a_wrong_number_of_texts_is_an_error_not_a_silent_mixup():
    batched = BatchedModelFn(lambda batch: [], max_batch=4, max_wait_s=0.05)   # always the wrong count
    results = []

    def call():
        try:
            batched([{"role": "user", "content": "x"}])
        except RuntimeError as e:
            results.append("error")

    threads = [threading.Thread(target=call) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
    finally:
        batched.close()
    assert results == ["error", "error"]


def test_max_batch_must_be_positive():
    with pytest.raises(ValueError):
        BatchedModelFn(lambda b: [], max_batch=0)


def test_configure_greedy_clears_sampling_settings():
    class Cfg:
        max_length = 262144
        do_sample = True
        temperature = 0.7
        top_p = 0.8
        top_k = 20

    class M:
        generation_config = Cfg()

    configure_greedy(M)
    cfg = M.generation_config
    assert (cfg.max_length, cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k) == (None, False, None, None, None)


def test_real_synthetic_tasks_run_the_same_batched_and_sequential():
    """Not a toy: the actual 121-task set, scripted correct model (chains 1, 3 and 5)."""
    def script(messages):
        # look the expected sequence up from the real tasks by user goal
        goal = messages[1]["content"]
        task = next(t for cl in (1, 3, 5) for t in load_tasks(cl) if t.user_goal == goal)
        done = sum(1 for m in messages if m["role"] == "tool" and not m["content"].startswith("ERROR:"))
        name = task.expected_tool_sequence[done]
        return f'<tool_call>{json.dumps({"name": name, "arguments": ARGS[name]})}</tool_call>'

    sequential = run_harness_eval_for_condition("base", script, [1, 3, 5], max_retries_per_step=0)
    fn = BatchedModelFn(lambda batch: [script(m) for m in batch], max_batch=16, max_wait_s=0.2)
    try:
        batched = run_harness_eval_for_condition("base", fn, [1, 3, 5], max_retries_per_step=0)
    finally:
        fn.close()
    assert batched == sequential
    assert len(batched) == 121
