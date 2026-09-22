"""evaluation/bfcl.py — parsing, prompt building and the simplified BFCL checker. No network:
the "real fixture" below is five verbatim rows copied from the public BFCL_v3_simple.json /
possible_answer/BFCL_v3_simple.json files, so parse_bfcl_examples is tested against the real
format, not a guess at it."""

import json

import pytest

from adbench.evaluation.batching import BatchedModelFn
from adbench.evaluation.bfcl import (
    BFCLExample,
    build_bfcl_prompt,
    check_bfcl_call,
    parse_bfcl_examples,
    pick_subset,
    run_bfcl_eval,
    summarize_bfcl,
    task_row,
)
from adbench.harness.executor import FinalAnswer, ToolCall

QUESTIONS_FIXTURE = "\n".join([
    json.dumps({"id": "simple_0", "question": [[{"role": "user", "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}]], "function": [{"name": "calculate_triangle_area", "description": "Calculate the area of a triangle given its base and height.", "parameters": {"type": "dict", "properties": {"base": {"type": "integer", "description": "The base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle."}, "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}}, "required": ["base", "height"]}}]}),
    json.dumps({"id": "simple_1", "question": [[{"role": "user", "content": "Calculate the factorial of 5 using math functions."}]], "function": [{"name": "math.factorial", "description": "Calculate the factorial of a given number.", "parameters": {"type": "dict", "properties": {"number": {"type": "integer", "description": "The number for which factorial needs to be calculated."}}, "required": ["number"]}}]}),
    json.dumps({"id": "simple_13", "question": [[{"role": "user", "content": "Calculate the area under the curve x**2 between 1 and 3."}]], "function": [{"name": "calculate_area_under_curve", "description": "Calculate the area under a curve.", "parameters": {"type": "dict", "properties": {"function": {"type": "string"}, "interval": {"type": "array"}, "method": {"type": "string"}}, "required": ["function", "interval"]}}]}),
    json.dumps({"id": "simple_no_answer", "question": [[{"role": "user", "content": "no ground truth published for this one"}]], "function": [{"name": "whatever", "description": "", "parameters": {}}]}),
])

ANSWERS_FIXTURE = "\n".join([
    json.dumps({"id": "simple_0", "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}),
    json.dumps({"id": "simple_1", "ground_truth": [{"math.factorial": {"number": [5]}}]}),
    json.dumps({"id": "simple_13", "ground_truth": [{"calculate_area_under_curve": {"function": ["x**2", "lambda x: x**2", "y=x**2"], "interval": [[1.0, 3.0]], "method": ["", "trapezoidal"]}}]}),
])


def test_parse_bfcl_examples_matches_the_real_file_format():
    examples = parse_bfcl_examples(QUESTIONS_FIXTURE, ANSWERS_FIXTURE)
    ids = [e.example_id for e in examples]
    assert ids == ["simple_0", "simple_1", "simple_13"]   # "simple_no_answer" dropped: no ground truth
    e0 = examples[0]
    assert e0.user_message == "Find the area of a triangle with a base of 10 units and height of 5 units."
    assert e0.functions[0]["name"] == "calculate_triangle_area"
    assert e0.ground_truth == {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}


def test_pick_subset_is_deterministic_and_evenly_spaced():
    examples = [BFCLExample(str(i), "", [], {}) for i in range(10)]
    assert [e.example_id for e in pick_subset(examples, 5)] == ["0", "2", "4", "6", "8"]
    assert pick_subset(examples, 100) == examples          # n >= len: no-op


def test_build_bfcl_prompt_lists_only_this_examples_own_functions_in_bfcl_schema():
    examples = parse_bfcl_examples(QUESTIONS_FIXTURE, ANSWERS_FIXTURE)
    messages = build_bfcl_prompt(examples[1])   # math.factorial
    assert messages[0]["role"] == "system"
    assert "math.factorial" in messages[0]["content"]
    assert '"type": "dict"' in messages[0]["content"]        # BFCL's own schema dialect, unmodified
    assert "calculate_triangle_area" not in messages[0]["content"]   # not this example's function
    assert messages[1] == {"role": "user", "content": "Calculate the factorial of 5 using math functions."}


# ---- check_bfcl_call ------------------------------------------------------

GT_SIMPLE0 = {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
GT_INTERVAL = {"calculate_area_under_curve": {"function": ["x**2", "lambda x: x**2"], "interval": [[1.0, 3.0]], "method": ["", "trapezoidal"]}}


def test_exact_match_including_the_optional_parameter():
    call = ToolCall("calculate_triangle_area", {"base": 10, "height": 5, "unit": "units"})
    assert check_bfcl_call(call, GT_SIMPLE0) is True


def test_optional_parameter_may_be_omitted_when_empty_string_is_acceptable():
    call = ToolCall("calculate_triangle_area", {"base": 10, "height": 5})
    assert check_bfcl_call(call, GT_SIMPLE0) is True


def test_required_parameter_missing_fails():
    call = ToolCall("calculate_triangle_area", {"base": 10})
    assert check_bfcl_call(call, GT_SIMPLE0) is False


def test_numeric_tolerance_and_string_case_insensitivity_carry_over():
    assert check_bfcl_call(ToolCall("calculate_triangle_area", {"base": 10.0000001, "height": 5, "unit": "UNITS"}), GT_SIMPLE0) is True


def test_wrong_value_fails():
    assert check_bfcl_call(ToolCall("calculate_triangle_area", {"base": 11, "height": 5}), GT_SIMPLE0) is False


def test_wrong_function_name_fails():
    assert check_bfcl_call(ToolCall("wrong_name", {"base": 10, "height": 5}), GT_SIMPLE0) is False


def test_unexpected_extra_argument_fails():
    call = ToolCall("calculate_triangle_area", {"base": 10, "height": 5, "extra_made_up_param": 1})
    assert check_bfcl_call(call, GT_SIMPLE0) is False


def test_a_final_answer_instead_of_a_tool_call_fails():
    assert check_bfcl_call(FinalAnswer("the area is 25"), GT_SIMPLE0) is False


def test_list_valued_acceptable_answer_matches_elementwise_with_tolerance():
    call = ToolCall("calculate_area_under_curve", {"function": "x**2", "interval": [1.0000001, 3.0]})
    assert check_bfcl_call(call, GT_INTERVAL) is True
    call_wrong = ToolCall("calculate_area_under_curve", {"function": "x**2", "interval": [1.0, 3.5]})
    assert check_bfcl_call(call_wrong, GT_INTERVAL) is False


def test_list_valued_parameter_may_also_be_omitted_when_optional():
    # "method" accepts "" (omittable) alongside "trapezoidal"
    call = ToolCall("calculate_area_under_curve", {"function": "x**2", "interval": [1.0, 3.0]})
    assert check_bfcl_call(call, GT_INTERVAL) is True


# ---- task_row / run_bfcl_eval / summarize_bfcl ----------------------------

def test_task_row_scores_a_well_formed_correct_call():
    ex = BFCLExample("simple_0", "goal", [], GT_SIMPLE0)
    raw = '<tool_call>{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}</tool_call>'
    row = task_row("distilled", ex, raw)
    assert row == {
        "condition": "distilled", "task_set": "bfcl_simple", "task_id": "simple_0",
        "success": True, "tool_name": "calculate_triangle_area", "error_type": None,
    }


def test_task_row_flags_a_malformed_call_without_crashing():
    ex = BFCLExample("simple_0", "goal", [], GT_SIMPLE0)
    row = task_row("base", ex, "<tool_call>{not: valid json}</tool_call>")
    assert row["success"] is False
    assert row["error_type"] == "MalformedCallError"
    assert row["tool_name"] is None


def test_run_bfcl_eval_sequential():
    examples = [BFCLExample("simple_0", "g0", [], GT_SIMPLE0), BFCLExample("simple_1", "g1", [], {"f": {"x": [1]}})]

    def script(messages):
        goal = messages[1]["content"]
        if goal == "g0":
            return '<tool_call>{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}</tool_call>'
        return '<tool_call>{"name": "f", "arguments": {"x": 2}}</tool_call>'   # wrong value -> fails

    rows = run_bfcl_eval("sft_only", script, examples)
    assert [r["task_id"] for r in rows] == ["simple_0", "simple_1"]
    assert [r["success"] for r in rows] == [True, False]
    assert summarize_bfcl(rows) == {"sft_only": 0.5}


def test_run_bfcl_eval_batched_keeps_input_order():
    examples = [BFCLExample(str(i), f"g{i}", [], {"f": {"x": [i]}}) for i in range(6)]

    def generate_batch(batch_messages):
        return [
            f'<tool_call>{{"name": "f", "arguments": {{"x": {m[1]["content"][1:]}}}}}</tool_call>'
            for m in batch_messages
        ]

    batched = BatchedModelFn(generate_batch, max_batch=4, max_wait_s=0.1)
    try:
        rows = run_bfcl_eval("distilled", batched, examples)
    finally:
        batched.close()
    assert [r["task_id"] for r in rows] == [str(i) for i in range(6)]
    assert all(r["success"] for r in rows)


def test_summarize_bfcl_groups_by_condition():
    rows = [
        {"condition": "a", "success": True}, {"condition": "a", "success": False},
        {"condition": "b", "success": True},
    ]
    assert summarize_bfcl(rows) == {"a": 0.5, "b": 1.0}
