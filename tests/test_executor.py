import json

import pytest

from adbench.harness.errors import MalformedCallError
from adbench.harness.executor import (
    FinalAnswer,
    ToolCall,
    build_system_prompt,
    parse_model_output,
    run_task,
)
from adbench.harness.tasks import TaskSpec
from adbench.harness.tools import build_demo_registry

# --- parse_model_output ---

def test_parse_plain_text_is_final_answer():
    parsed = parse_model_output("The answer is 42.")
    assert isinstance(parsed, FinalAnswer)
    assert parsed.text == "The answer is 42."


def test_parse_valid_tool_call():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    parsed = parse_model_output(text)
    assert isinstance(parsed, ToolCall)
    assert parsed.name == "get_weather"
    assert parsed.arguments == {"city": "Paris"}


def test_parse_tool_call_defaults_arguments_to_empty_dict():
    text = '<tool_call>{"name": "get_current_time"}</tool_call>'
    parsed = parse_model_output(text)
    assert isinstance(parsed, ToolCall)
    assert parsed.arguments == {}


def test_parse_tool_call_across_multiple_lines():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    parsed = parse_model_output(text)
    assert isinstance(parsed, ToolCall)
    assert parsed.name == "get_weather"


def test_parse_invalid_json_raises_malformed_call_error():
    with pytest.raises(MalformedCallError):
        parse_model_output("<tool_call>{not valid json}</tool_call>")


def test_parse_missing_name_field_raises():
    with pytest.raises(MalformedCallError):
        parse_model_output('<tool_call>{"arguments": {}}</tool_call>')


def test_parse_arguments_not_object_raises():
    with pytest.raises(MalformedCallError):
        parse_model_output('<tool_call>{"name": "x", "arguments": "oops"}</tool_call>')


# --- build_system_prompt ---

def test_system_prompt_lists_every_tool_and_the_protocol():
    registry = build_demo_registry()
    prompt = build_system_prompt(registry)
    for name in registry.list_names():
        assert name in prompt
    assert "<tool_call>" in prompt


# --- run_task: helpers ---

def _tool_call_text(name: str, arguments: dict) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def make_sequential_model_fn(expected_sequence, arguments_by_position):
    """Scripted model: emits the tool call for whichever step comes next,
    determined by counting *successful* prior tool results in the running
    message history — so a retry after an error naturally re-attempts the
    same step rather than advancing."""

    def model_fn(messages):
        success_count = sum(
            1 for m in messages
            if m["role"] == "tool" and not m["content"].startswith("ERROR:")
        )
        name = expected_sequence[success_count]
        args = arguments_by_position[success_count]
        return _tool_call_text(name, args)

    return model_fn


# --- run_task: single step ---

def test_run_task_single_step_success():
    registry = build_demo_registry()
    task = TaskSpec(
        task_id="t1", chain_length=1, user_goal="weather?",
        expected_tool_sequence=["get_weather"],
    )
    model_fn = make_sequential_model_fn(["get_weather"], [{"city": "Paris"}])

    state = run_task(model_fn, task, registry)

    assert state.final_success is True
    assert len(state.steps) == 1
    assert state.steps[0].succeeded is True
    assert state.steps[0].tool_name == "get_weather"
    assert state.steps[0].recovered_from_error is False
    assert state.steps[0].error_type is None


# --- run_task: multi-step success ---

def test_run_task_three_step_success():
    registry = build_demo_registry()
    sequence = ["get_weather", "calculator", "get_current_time"]
    args = [{"city": "Tokyo"}, {"expression": "71 * 2"}, {"timezone": "jst"}]
    task = TaskSpec(
        task_id="t3", chain_length=3, user_goal="do three things",
        expected_tool_sequence=sequence,
    )
    model_fn = make_sequential_model_fn(sequence, args)

    state = run_task(model_fn, task, registry)

    assert state.final_success is True
    assert [s.tool_name for s in state.steps] == sequence
    assert all(s.succeeded for s in state.steps)
    # a tool result should have been appended to the conversation per step
    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert len(tool_messages) == 3


# --- run_task: recovery from an injected tool-execution error ---

def test_run_task_recovers_from_injected_tool_execution_error():
    registry = build_demo_registry()
    sequence = ["get_weather", "calculator", "search_knowledge_base"]
    args = [{"city": "Cairo"}, {"expression": "95 - 10"}, {"query": "lora"}]
    task = TaskSpec(
        task_id="t3-err", chain_length=3, user_goal="do three things, one will fail once",
        expected_tool_sequence=sequence,
        injected_error_at_step=1,
    )
    model_fn = make_sequential_model_fn(sequence, args)

    state = run_task(model_fn, task, registry)

    assert state.final_success is True
    assert len(state.steps) == 3
    assert state.steps[1].tool_name == "calculator"
    assert state.steps[1].succeeded is True
    assert state.steps[1].recovered_from_error is True
    # the other steps were never forced to fail
    assert state.steps[0].recovered_from_error is False
    assert state.steps[2].recovered_from_error is False


# --- run_task: recovery from a protocol error (unknown tool, then correct) ---

def test_run_task_recovers_from_protocol_error():
    registry = build_demo_registry()
    calls = {"n": 0}

    def flaky_then_correct(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_call_text("does_not_exist", {})
        return _tool_call_text("get_weather", {"city": "Paris"})

    task = TaskSpec(
        task_id="t1-protocol-err", chain_length=1, user_goal="weather?",
        expected_tool_sequence=["get_weather"],
    )

    state = run_task(flaky_then_correct, task, registry)

    assert state.final_success is True
    assert state.steps[0].succeeded is True
    assert state.steps[0].recovered_from_error is True
    assert state.steps[0].error_type is None  # cleared once the step ultimately succeeds
    assert calls["n"] == 2


def test_run_task_recovers_from_premature_final_answer():
    registry = build_demo_registry()
    calls = {"n": 0}

    def answers_then_calls(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Sure, let me think about that."
        return _tool_call_text("get_weather", {"city": "Paris"})

    task = TaskSpec(
        task_id="t1-final-answer-first", chain_length=1, user_goal="weather?",
        expected_tool_sequence=["get_weather"],
    )

    state = run_task(answers_then_calls, task, registry)

    assert state.final_success is True
    assert state.steps[0].recovered_from_error is True


# --- run_task: unrecoverable failures ---

def test_run_task_fails_after_exhausting_retries_on_wrong_tool():
    registry = build_demo_registry()

    def always_wrong_tool(messages):
        return _tool_call_text("calculator", {"expression": "1 + 1"})

    task = TaskSpec(
        task_id="t1-wrong-tool", chain_length=1, user_goal="weather?",
        expected_tool_sequence=["get_weather"],
    )

    state = run_task(always_wrong_tool, task, registry, max_retries_per_step=2)

    assert state.final_success is False
    assert len(state.steps) == 1
    assert state.steps[0].succeeded is False
    assert state.steps[0].error is not None
    assert state.steps[0].error_type == "WrongToolError"


def test_run_task_stops_at_first_unrecoverable_step():
    """A task with chain_length=3 whose first step never succeeds should
    record exactly one (failed) step, not run steps 2 and 3."""
    registry = build_demo_registry()

    def always_malformed(messages):
        return "<tool_call>{not json}</tool_call>"

    task = TaskSpec(
        task_id="t3-stops-early", chain_length=3, user_goal="do three things",
        expected_tool_sequence=["get_weather", "calculator", "get_current_time"],
    )

    state = run_task(always_malformed, task, registry, max_retries_per_step=1)

    assert state.final_success is False
    assert len(state.steps) == 1
    assert state.steps[0].error_type == "MalformedCallError"


def test_run_task_exhausts_retries_on_persistent_injected_error():
    registry = build_demo_registry()
    sequence = ["get_weather"]
    args = [{"city": "Paris"}]
    task = TaskSpec(
        task_id="t1-persistent-err", chain_length=1, user_goal="weather?",
        expected_tool_sequence=sequence,
        injected_error_at_step=0,
    )
    model_fn = make_sequential_model_fn(sequence, args)

    # max_retries_per_step=0 means only the forced-fail attempt ever runs
    state = run_task(model_fn, task, registry, max_retries_per_step=0)

    assert state.final_success is False
    assert state.steps[0].succeeded is False
    assert state.steps[0].error_type == "ToolExecutionError"
