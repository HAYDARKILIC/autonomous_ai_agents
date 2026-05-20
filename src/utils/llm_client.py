"""Provider-agnostic LLM client.

A thin abstraction that hides whether the underlying provider is OpenAI or
Anthropic. The agent and re-ranking code depend only on the :class:`LLMClient`
Protocol; tests inject a deterministic fake.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    """Minimal interface every concrete LLM client must satisfy."""

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant's response as a plain string."""
        ...


class AnthropicClient:
    """Thin wrapper around `anthropic.Anthropic`."""

    def __init__(self, model: str = "claude-opus-4-7", api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        system = next((m.content for m in messages if m.role == "system"), None)
        api_msgs = [{"role": m.role, "content": m.content}
                    for m in messages if m.role != "system"]
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_msgs,
        }
        if system is not None:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text


class OpenAIClient:
    """Thin wrapper around `openai.OpenAI`."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return resp.choices[0].message.content or ""


def build_client(provider: str, model: str | None = None) -> LLMClient:
    """Factory: returns a configured client by provider name."""
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-opus-4-7")
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o")
    raise ValueError(f"Unknown LLM provider: {provider!r}")
