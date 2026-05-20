"""Tests for the ReAct text-protocol parser."""

from __future__ import annotations

import pytest

from src.react_agent.parser import ParseError, parse_step


def test_parse_tool_step() -> None:
    raw = (
        "Thought: I need to compute 7 * 6.\n"
        "Action: calculator\n"
        "Action Input: {\"expression\": \"7 * 6\"}"
    )
    step = parse_step(raw)
    assert step.is_final is False
    assert step.action == "calculator"
    assert step.action_input == {"expression": "7 * 6"}


def test_parse_final_answer() -> None:
    raw = "Thought: The product is 42.\nFinal Answer: 42"
    step = parse_step(raw)
    assert step.is_final is True
    assert step.final_answer == "42"


def test_parse_step_strips_json_fences() -> None:
    raw = (
        "Thought: Use the calculator.\n"
        "Action: calculator\n"
        "Action Input: ```json\n{\"expression\": \"1+1\"}\n```"
    )
    step = parse_step(raw)
    assert step.action_input == {"expression": "1+1"}


def test_missing_thought_raises() -> None:
    with pytest.raises(ParseError):
        parse_step("Action: x\nAction Input: {}")


def test_missing_action_and_final_raises() -> None:
    with pytest.raises(ParseError):
        parse_step("Thought: just thinking")


def test_invalid_json_action_input_raises() -> None:
    raw = "Thought: t\nAction: x\nAction Input: not-json"
    with pytest.raises(ParseError):
        parse_step(raw)
