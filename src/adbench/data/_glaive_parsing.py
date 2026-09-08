"""Parsing for glaiveai/glaive-function-calling-v2's raw "chat" text field.

The raw dataset is NOT structured turns — "chat" is one free-text blob using
USER:/ASSISTANT:/FUNCTION RESPONSE: markers, and a function call is a
<functioncall> block whose "arguments" value is a JSON-encoded STRING
wrapped in single quotes (e.g. `'arguments': '{"a": 1}'`), not a nested JSON
object. That string's content can itself contain nested braces and span
multiple lines, so a naive regex (`\\{.*?\\}`) matches only up to the first
`}` it finds — usually inside the arguments string — and fails on the vast
majority of real examples. This module does real brace-counting and
explicit string-boundary scanning instead.

Empirically (checked against the full 112,960-row train split): this parses
99.1% of <functioncall> blocks. The remaining ~0.9% are genuine data
artifacts — an unescaped apostrophe inside the single-quoted arguments
string (e.g. a message body containing "[User's Name]") that makes the
string's own closing quote ambiguous. Those are treated as unparseable and
skipped by prepare.py, not specially unescaped — the volume needed is tiny
relative to the size of this drop.
"""

import ast
import json


class FunctionCallParseError(Exception):
    """A <functioncall> block (or the USER: text preceding it) didn't parse."""


def _find_matching_brace(text: str, open_pos: int) -> int:
    """text[open_pos] must be '{'. Returns the index of the matching '}'."""
    depth = 0
    i = open_pos
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise FunctionCallParseError("Unbalanced braces: no matching '}' found.")


def _find_matching_quote(text: str, open_pos: int) -> int:
    """text[open_pos] is a quote character (' or "). Returns the index of
    the matching *unescaped* closing quote of the same kind."""
    quote_char = text[open_pos]
    i = open_pos + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote_char:
            return i
        i += 1
    raise FunctionCallParseError(f"Unterminated string starting at position {open_pos}.")


def _extract_functioncall_block(chat: str, tag_end_pos: int) -> tuple[str, int]:
    """Given the index right after a '<functioncall>' tag, skip whitespace,
    require a '{', and return (raw_block_text, index_after_block)."""
    i = tag_end_pos
    while i < len(chat) and chat[i].isspace():
        i += 1
    if i >= len(chat) or chat[i] != "{":
        raise FunctionCallParseError(
            f"Expected '{{' after <functioncall> at position {tag_end_pos}, "
            f"found {chat[i:i + 30]!r}."
        )
    end = _find_matching_brace(chat, i)
    return chat[i : end + 1], end + 1


def _parse_functioncall_block(raw_block: str) -> dict:
    """Parse one raw block (e.g. `{"name": "x", "arguments": '{"a": 1}'}`)
    into {"name": str, "arguments": dict, "arguments_style": str}.

    arguments_style records how the raw dataset encoded the value —
    "string_single_quoted" (the dominant convention), "string_double_quoted",
    "raw_object" (already a nested object, no reformatting needed), or
    "missing" (no arguments field at all) — purely for prepare.py's report
    on how much reformatting the raw data actually needed.
    """
    name_pos = raw_block.find('"name"')
    if name_pos == -1:
        raise FunctionCallParseError('No "name" key found.')
    colon_pos = raw_block.find(":", name_pos)
    q1 = raw_block.find('"', colon_pos)
    q2 = _find_matching_quote(raw_block, q1)
    name = raw_block[q1 + 1 : q2]

    args_pos = raw_block.find('"arguments"')
    if args_pos == -1:
        return {"name": name, "arguments": {}, "arguments_style": "missing"}

    colon_pos = raw_block.find(":", args_pos)
    i = colon_pos + 1
    while i < len(raw_block) and raw_block[i].isspace():
        i += 1
    if i >= len(raw_block):
        raise FunctionCallParseError('"arguments" value is empty.')

    ch = raw_block[i]
    if ch in ("'", '"'):
        end_q = _find_matching_quote(raw_block, i)
        inner = raw_block[i + 1 : end_q]
        style = "string_single_quoted" if ch == "'" else "string_double_quoted"
        try:
            arguments = json.loads(inner)
        except json.JSONDecodeError:
            try:
                arguments = ast.literal_eval(inner)
            except (ValueError, SyntaxError) as e:
                raise FunctionCallParseError(
                    f"Could not parse arguments string as JSON or a Python literal: "
                    f"{inner[:200]!r} ({e})"
                ) from e
    elif ch == "{":
        end_b = _find_matching_brace(raw_block, i)
        inner = raw_block[i : end_b + 1]
        style = "raw_object"
        arguments = json.loads(inner)
    else:
        raise FunctionCallParseError(f"Unexpected arguments value start: {raw_block[i:i + 30]!r}")

    if not isinstance(arguments, dict):
        raise FunctionCallParseError(f"Parsed arguments is not an object: {type(arguments).__name__}")

    return {"name": name, "arguments": arguments, "arguments_style": style}


def find_all_functioncalls(chat: str) -> list[dict]:
    """Find and parse every <functioncall> block in a chat string, in order.
    Each result also carries "_char_offset" (the block's position in `chat`,
    used to locate the user message that preceded it). Raises
    FunctionCallParseError on the first block that doesn't parse — callers
    decide whether that means dropping the whole example."""
    results = []
    pos = 0
    while True:
        tag_pos = chat.find("<functioncall>", pos)
        if tag_pos == -1:
            break
        tag_end = tag_pos + len("<functioncall>")
        raw_block, after = _extract_functioncall_block(chat, tag_end)
        parsed = _parse_functioncall_block(raw_block)
        parsed["_char_offset"] = tag_pos
        results.append(parsed)
        pos = after
    return results


def extract_last_user_message_before(chat: str, char_offset: int) -> str:
    """The USER: message text immediately preceding a given position in
    `chat` — used as a function call's user_goal. If a request needed
    clarification (multiple USER: turns before the actual tool call), this
    returns the *last* one: the final, fully-specified request, not the
    opening turn.
    """
    search_region = chat[:char_offset]
    user_pos = search_region.rfind("USER:")
    if user_pos == -1:
        raise FunctionCallParseError("No preceding USER: turn found for this function call.")
    start = user_pos + len("USER:")
    # The turn ends at the next role marker or end of the searched region.
    end_candidates = [
        p for p in (
            search_region.find("ASSISTANT:", start),
            search_region.find("FUNCTION RESPONSE:", start),
            search_region.find("USER:", start),
        ) if p != -1
    ]
    end = min(end_candidates) if end_candidates else len(search_region)
    return search_region[start:end].strip()
