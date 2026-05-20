"""Shared state for multi-agent collaboration.

The state is a structured object passed between agents and stamped after
each turn. A SHA-256 hash of the content is computed each round; if the
same hash repeats, the system has entered a non-productive cycle and
the orchestrator aborts. This is a simple but effective guard against
the most common multi-agent failure mode: two agents oscillating
between two contradictory positions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Turn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Mutable blackboard shared across the multi-agent loop."""

    task: str
    history: list[Turn] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)
    code: str | None = None
    test_passed: bool = False
    last_error: str | None = None
    iteration: int = 0

    def append(self, role: str, content: str, **meta: Any) -> None:
        self.history.append(Turn(role=role, content=content, metadata=meta))

    def fingerprint(self) -> str:
        """Hash of the *productive* content (code + last error) for loop detection."""
        material = f"{self.code or ''}|{self.last_error or ''}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def detect_loop(self) -> bool:
        """Return True if the current fingerprint has been seen before."""
        fp = self.fingerprint()
        if fp in self.seen_hashes:
            return True
        self.seen_hashes.add(fp)
        return False

    def summary(self) -> str:
        return "\n".join(f"[{t.role}] {t.content[:200]}" for t in self.history[-5:])
