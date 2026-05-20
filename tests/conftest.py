"""Shared pytest fixtures.

A deterministic :class:`FakeLLM` replaces the real provider in unit tests
so the suite is hermetic, free, and reproducible.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.utils.llm_client import Message


class FakeLLM:
    """A scripted LLM: returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses.")
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory() -> Iterator[type[FakeLLM]]:
    yield FakeLLM
