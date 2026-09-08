import pytest

from adbench.data._glaive_parsing import (
    FunctionCallParseError,
    extract_last_user_message_before,
    find_all_functioncalls,
)

# Real examples (lightly trimmed) from glaiveai/glaive-function-calling-v2,
# used to keep these tests anchored to the actual raw format rather than an
# idealized one.

SINGLE_QUOTED_CHAT = """USER: Can you tell me the latest news headlines for the United States?


ASSISTANT: <functioncall> {"name": "get_news_headlines", "arguments": '{"country": "United States"}'} <|endoftext|>


FUNCTION RESPONSE: {"headlines": ["A", "B"]}


ASSISTANT: Here are the headlines. <|endoftext|>"""

MULTILINE_ARGUMENTS_CHAT = """USER: I spent $200 on groceries on 1st.


ASSISTANT: <functioncall> {"name": "track_expenses", "arguments": '{
  "category": "groceries",
  "amount": 200,
  "date": "2022-04-01"
}'} <|endoftext|>


FUNCTION RESPONSE: {"status": "success"}"""

RAW_OBJECT_ARGUMENTS_CHAT = (
    'USER: What is the BMI for 70kg and 1.75m?\n\n\n'
    'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": {"weight": 70, "height": 1.75}} <|endoftext|>'
)

MISSING_ARGUMENTS_CHAT = (
    "USER: What time is it?\n\n\n"
    'ASSISTANT: <functioncall> {"name": "get_current_time"} <|endoftext|>'
)

# A real failure mode: an unescaped apostrophe inside the single-quoted
# arguments string breaks the naive string boundary.
UNESCAPED_APOSTROPHE_CHAT = (
    "USER: Email my boss.\n\n\n"
    'ASSISTANT: <functioncall> {"name": "send_email", "arguments": '
    "'{\"message\": \"Regards, [User's Name]\"}'} <|endoftext|>"
)

MULTI_CALL_CHAT = """USER: What's the weather in Tokyo?


ASSISTANT: <functioncall> {"name": "get_weather", "arguments": '{"city": "Tokyo"}'} <|endoftext|>


FUNCTION RESPONSE: {"temp": 71}


ASSISTANT: It's 71F. <|endoftext|>


USER: What about Paris?


ASSISTANT: <functioncall> {"name": "get_weather", "arguments": '{"city": "Paris"}'} <|endoftext|>"""


def test_single_quoted_arguments_parses():
    calls = find_all_functioncalls(SINGLE_QUOTED_CHAT)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_news_headlines"
    assert calls[0]["arguments"] == {"country": "United States"}
    assert calls[0]["arguments_style"] == "string_single_quoted"


def test_multiline_arguments_parses():
    calls = find_all_functioncalls(MULTILINE_ARGUMENTS_CHAT)
    assert len(calls) == 1
    assert calls[0]["arguments"] == {"category": "groceries", "amount": 200, "date": "2022-04-01"}


def test_raw_object_arguments_parses():
    calls = find_all_functioncalls(RAW_OBJECT_ARGUMENTS_CHAT)
    assert len(calls) == 1
    assert calls[0]["arguments"] == {"weight": 70, "height": 1.75}
    assert calls[0]["arguments_style"] == "raw_object"


def test_missing_arguments_defaults_to_empty_dict():
    calls = find_all_functioncalls(MISSING_ARGUMENTS_CHAT)
    assert len(calls) == 1
    assert calls[0]["arguments"] == {}
    assert calls[0]["arguments_style"] == "missing"


def test_unescaped_apostrophe_raises_parse_error():
    """This is a genuine data artifact (~0.9% of real function calls), not a
    parser bug — documented in prepare.py as a skip, not a repair."""
    with pytest.raises(FunctionCallParseError):
        find_all_functioncalls(UNESCAPED_APOSTROPHE_CHAT)


def test_multiple_functioncalls_found_in_order():
    calls = find_all_functioncalls(MULTI_CALL_CHAT)
    assert len(calls) == 2
    assert [c["name"] for c in calls] == ["get_weather", "get_weather"]
    assert calls[0]["arguments"]["city"] == "Tokyo"
    assert calls[1]["arguments"]["city"] == "Paris"
    assert calls[0]["_char_offset"] < calls[1]["_char_offset"]


def test_no_functioncall_returns_empty_list():
    assert find_all_functioncalls("USER: hi\n\nASSISTANT: hello") == []


# --- extract_last_user_message_before ---

def test_extract_user_message_immediately_before_call():
    offset = SINGLE_QUOTED_CHAT.find("<functioncall>")
    goal = extract_last_user_message_before(SINGLE_QUOTED_CHAT, offset)
    assert goal == "Can you tell me the latest news headlines for the United States?"


def test_extract_picks_last_user_turn_when_clarification_happened():
    chat = (
        "USER: I need a password.\n\n\n"
        "ASSISTANT: How long?\n\n\n"
        "USER: 12 characters, with symbols.\n\n\n"
        'ASSISTANT: <functioncall> {"name": "generate_password", "arguments": '
        "'{\"length\": 12}'} <|endoftext|>"
    )
    offset = chat.find("<functioncall>")
    goal = extract_last_user_message_before(chat, offset)
    assert goal == "12 characters, with symbols."


def test_extract_second_call_gets_the_user_turn_right_before_it():
    offset = MULTI_CALL_CHAT.rfind("<functioncall>")
    goal = extract_last_user_message_before(MULTI_CALL_CHAT, offset)
    assert goal == "What about Paris?"


def test_extract_no_preceding_user_turn_raises():
    chat = 'ASSISTANT: <functioncall> {"name": "x", "arguments": \'{}\'} <|endoftext|>'
    with pytest.raises(FunctionCallParseError):
        extract_last_user_message_before(chat, chat.find("<functioncall>"))
