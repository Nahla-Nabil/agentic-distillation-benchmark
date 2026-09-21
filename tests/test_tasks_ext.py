"""Extended synthetic task set (source="synthetic_ext") and its six extra tools — no GPU."""

import json

import pytest

from adbench.evaluation.run_eval import EXT_TASK_SET, run_ext_eval_for_condition
from adbench.harness.errors import ToolArgumentError, ToolExecutionError
from adbench.harness.executor import run_task
from adbench.harness.tasks import build_synthetic_tasks, load_tasks
from adbench.harness.tools import (
    average_numbers,
    build_demo_registry,
    build_extended_registry,
    build_glaive_registry,
    convert_length,
    get_capital,
    get_elevation,
    get_population,
    round_number,
)

ARGS = {
    "get_weather": {"city": "Paris"},
    "calculator": {"expression": "1 + 1"},
    "search_knowledge_base": {"query": "lora"},
    "get_current_time": {"timezone": "utc"},
    "convert_temperature": {"value": 100, "from_unit": "F", "to_unit": "C"},
    "compare_numbers": {"a": 5, "b": 3},
    "get_population": {"country": "France"},
    "get_capital": {"country": "France"},
    "get_elevation": {"place": "Everest"},
    "convert_length": {"value": 5, "from_unit": "km", "to_unit": "mi"},
    "average_numbers": {"a": 4, "b": 8},
    "round_number": {"value": 3.14159, "decimals": 2},
}
EXPECTED_STEPS = "expected_tool_sequence"


def _all_ext():
    return {cl: load_tasks(cl, source="synthetic_ext") for cl in (1, 3, 5)}


def _template(task_id: str) -> str:
    return task_id.rsplit("-", 1)[0]


def test_extended_set_is_large_and_diverse():
    tasks = _all_ext()
    assert (len(tasks[1]), len(tasks[3]), len(tasks[5])) == (30, 64, 64)
    templates = {_template(t.task_id) for ts in tasks.values() for t in ts}
    assert len(templates) == 38
    # the original set has 21 templates; together they must give far more chain-3/5 templates
    assert len({_template(t.task_id) for cl in (3, 5) for t in tasks[cl]}) == 32


def test_task_ids_are_unique_and_disjoint_from_the_original_set():
    ext_ids = [t.task_id for ts in _all_ext().values() for t in ts]
    original_ids = [t.task_id for cl in (1, 3, 5) for t in build_synthetic_tasks(cl)]
    assert len(ext_ids) == len(set(ext_ids))
    assert not set(ext_ids) & set(original_ids)
    assert len(original_ids) == 121          # the original set must not change


def test_chain_length_matches_sequence_and_every_task_uses_known_tools():
    registry = build_extended_registry()
    for chain_length, tasks in _all_ext().items():
        for t in tasks:
            assert t.chain_length == chain_length == len(t.expected_tool_sequence), t.task_id
            assert all(registry.has(n) for n in t.expected_tool_sequence), t.task_id
            assert t.user_goal.strip() and "None" not in t.user_goal and "{" not in t.user_goal, t.task_id


def test_every_extended_template_is_a_new_tool_sequence():
    original = {tuple(t.expected_tool_sequence) for cl in (1, 3, 5) for t in build_synthetic_tasks(cl)}
    ext = {tuple(t.expected_tool_sequence) for ts in _all_ext().values() for t in ts if t.chain_length > 1}
    assert not original & ext


def test_new_tools_are_not_training_tools_and_the_demo_registry_is_untouched():
    new_tools = set(build_extended_registry().list_names()) - set(build_demo_registry().list_names())
    assert new_tools == {"get_population", "get_capital", "get_elevation", "convert_length", "average_numbers", "round_number"}
    assert not new_tools & set(build_glaive_registry().list_names())
    # the original tasks' prompt lists the demo tools only, exactly as before
    assert len(build_demo_registry().list_names()) == 6


def test_a_correct_scripted_model_completes_every_extended_task():
    registry = build_extended_registry()

    def make_model_fn(task):
        def model_fn(messages):
            done = sum(1 for m in messages if m["role"] == "tool" and not m["content"].startswith("ERROR:"))
            name = task.expected_tool_sequence[done]
            return f'<tool_call>{json.dumps({"name": name, "arguments": ARGS[name]})}</tool_call>'
        return model_fn

    for tasks in _all_ext().values():
        for t in tasks:
            state = run_task(make_model_fn(t), t, registry, max_retries_per_step=2)
            assert state.final_success, t.task_id


def test_injected_errors_are_assigned_like_the_original_set():
    for tasks in _all_ext().values():
        injected = [t for t in tasks if t.injected_error_at_step is not None]
        assert len(injected) == len(range(0, len(tasks), 4))
        assert all(0 <= t.injected_error_at_step < t.chain_length for t in injected)


def test_unsupported_chain_length_is_rejected():
    with pytest.raises(ValueError):
        load_tasks(2, source="synthetic_ext")


def test_run_ext_eval_tags_rows_with_the_extended_task_set():
    def script(messages):
        goal = messages[1]["content"]
        task = next(t for cl in (1, 3, 5) for t in load_tasks(cl, source="synthetic_ext") if t.user_goal == goal)
        done = sum(1 for m in messages if m["role"] == "tool" and not m["content"].startswith("ERROR:"))
        name = task.expected_tool_sequence[done]
        return f'<tool_call>{json.dumps({"name": name, "arguments": ARGS[name]})}</tool_call>'

    rows = run_ext_eval_for_condition("base", script, [1, 3, 5], max_retries_per_step=2)
    assert len(rows) == 158
    assert {r["task_set"] for r in rows} == {EXT_TASK_SET}
    assert all(r["success"] for r in rows)
    assert all("transcript" not in r for r in rows)


def test_transcripts_are_stored_only_on_request(monkeypatch):
    def script(messages):
        return '<tool_call>{"name": "get_capital", "arguments": {"country": "France"}}</tool_call>'

    monkeypatch.setenv("ADBENCH_STORE_TRANSCRIPTS", "1")
    from adbench.evaluation.run_eval import run_harness_eval_for_condition

    rows = run_harness_eval_for_condition(
        "base", script, [1], build_extended_registry(), 0, task_source="synthetic_ext", task_set=EXT_TASK_SET, max_tasks=2
    )
    transcript = rows[0]["transcript"]
    assert transcript[0]["role"] == "user" and all(m["role"] != "system" for m in transcript)
    assert any(m["role"] == "assistant" and "get_capital" in m["content"] for m in transcript)


# ---- the six new tools ----------------------------------------------------

def test_population_capital_elevation_lookups_and_failures():
    assert get_population("France")["population_millions"] == 68
    assert get_capital("Japan")["capital"] == "tokyo"
    assert get_elevation("Everest")["elevation_m"] == 8849
    for fn, bad in ((get_population, "atlantis"), (get_capital, "atlantis"), (get_elevation, "olympus mons")):
        with pytest.raises(ToolExecutionError):
            fn(bad)


def test_capitals_used_in_chains_exist_in_the_weather_table():
    from adbench.harness.tasks import _CAPITAL_COUNTRIES
    from adbench.harness.tools import WEATHER_TABLE

    for country in _CAPITAL_COUNTRIES:
        assert get_capital(country)["capital"] in WEATHER_TABLE


def test_convert_length():
    assert convert_length(1, "km", "m")["converted_value"] == 1000
    assert convert_length(1, "mi", "km")["converted_value"] == 1.61
    assert convert_length(3048, "m", "ft")["converted_value"] == 10000
    with pytest.raises(ToolArgumentError):
        convert_length(1, "parsec", "m")
    with pytest.raises(ToolArgumentError):
        convert_length(-1, "m", "km")


def test_average_and_round():
    assert average_numbers(4, 9)["average"] == 6.5
    assert round_number(3.14159, 2)["rounded"] == 3.14
    assert round_number(2.5, 0)["rounded"] == 2      # banker's rounding, documented Python behaviour
    with pytest.raises(ToolArgumentError):
        average_numbers(float("nan"), 1)
    with pytest.raises(ToolArgumentError):
        round_number(1.0, 11)
    with pytest.raises(ToolArgumentError):
        round_number(1.0, 1.5)


# ---- agreement between a batched re-run and recorded rows -------------------

def _row(task_id, success, tools):
    return {"task_id": task_id, "success": success, "steps": [{"tool_name": t} for t in tools]}


def test_agreement_with_recorded_counts_matching_tasks_only():
    from adbench.evaluation.run_eval import agreement_with_recorded

    recorded = [_row("a", True, ["x", "y"]), _row("b", False, ["x"]), _row("c", True, ["z"])]
    rerun = [_row("a", True, ["x", "y"]), _row("b", True, ["x"]), _row("d", True, ["q"])]
    # "d" has no recorded counterpart and is ignored; "b" flipped success but kept its steps
    assert agreement_with_recorded(rerun, recorded) == {"compared": 2, "same_success": 1, "same_steps": 2}


def test_fast_inference_default_follows_the_environment(monkeypatch):
    from adbench.evaluation.run_eval import fast_inference_default

    monkeypatch.delenv("ADBENCH_FAST_INFERENCE", raising=False)
    assert fast_inference_default() is True
    monkeypatch.setenv("ADBENCH_FAST_INFERENCE", "0")
    assert fast_inference_default() is False
