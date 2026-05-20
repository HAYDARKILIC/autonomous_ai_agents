"""Parser for the ReAct text protocol.

The agent prompts the LLM to emit a structured trace in this form:

    Thought: <free text>
    Action: <tool_name>
    Action Input: <JSON object>

…or, when finished:

    Thought: <free text>
    Final Answer: <free text>

A robust parser is essential: small format deviations should not break
the loop. The implementation here forgives whitespace, accepts loose
JSON for action inputs, and surfaces a structured :class:`ParseError`
that the agent uses to inject a corrective message back into the LLM
context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class ParseError(ValueError):
    """Raised when a ReAct step cannot be parsed."""


@dataclass(slots=True)
class Step:
    thought: str
    is_final: bool
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None


_THOUGHT = re.compile(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
_ACTION = re.compile(r"Action:\s*(.+?)(?=\n)", re.DOTALL)
_ACTION_INPUT = re.compile(r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\Z)", re.DOTALL)
_FINAL = re.compile(r"Final Answer:\s*(.+)\Z", re.DOTALL)


def parse_step(text: str) -> Step:
    """Parse a single LLM turn into a :class:`Step` object."""
    thought_match = _THOUGHT.search(text)
    if thought_match is None:
        raise ParseError("Missing `Thought:` field.")
    thought = thought_match.group(1).strip()

    final_match = _FINAL.search(text)
    if final_match is not None:
        return Step(thought=thought, is_final=True, final_answer=final_match.group(1).strip())

    action_match = _ACTION.search(text)
    if action_match is None:
        raise ParseError("Missing `Action:` and `Final Answer:` — at least one required.")
    action = action_match.group(1).strip()

    input_match = _ACTION_INPUT.search(text)
    if input_match is None:
        raise ParseError(f"Action {action!r} present but no `Action Input:` provided.")

    raw_input = input_match.group(1).strip()
    raw_input = raw_input.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        action_input = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Action Input is not valid JSON: {exc}") from exc
    if not isinstance(action_input, dict):
        raise ParseError("Action Input must be a JSON object.")

    return Step(thought=thought, is_final=False, action=action, action_input=action_input)
