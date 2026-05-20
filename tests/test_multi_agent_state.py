"""Tests for multi-agent state and loop detection."""

from __future__ import annotations

from src.multi_agent.state import AgentState


def test_fingerprint_is_deterministic() -> None:
    a = AgentState(task="t", code="print(1)", last_error="oops")
    b = AgentState(task="t", code="print(1)", last_error="oops")
    assert a.fingerprint() == b.fingerprint()


def test_detect_loop_returns_true_on_repeat() -> None:
    state = AgentState(task="t", code="print(1)", last_error="boom")
    assert state.detect_loop() is False
    # Same code + same error → loop.
    assert state.detect_loop() is True


def test_detect_loop_returns_false_on_progress() -> None:
    state = AgentState(task="t", code="print(1)", last_error="boom")
    state.detect_loop()
    state.code = "print(2)"   # the Coder revised the code
    assert state.detect_loop() is False


def test_history_appends() -> None:
    state = AgentState(task="t")
    state.append("coder", "first code")
    state.append("critic", "looks wrong")
    assert len(state.history) == 2
    assert state.history[1].role == "critic"
