"""Role-specialized agents for the self-correcting coder topology.

The three roles:

- :class:`CoderAgent`    — writes / revises a Python solution.
- :class:`ExecutorAgent` — runs the code in a subprocess and captures
                           stdout / stderr / exit code. *Not* an LLM.
- :class:`CriticAgent`   — inspects an error trace and proposes a fix
                           strategy in natural language.

Subprocess execution
--------------------
The Executor uses ``subprocess.run`` with a hard timeout. This is not
a hardened sandbox — for genuine isolation, replace with a container
runtime (e.g. ``docker``, ``firejail``, or a cloud sandbox). The
interface is designed so the swap is local.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.utils.llm_client import LLMClient, Message


_CODE_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Pull the largest Python code block out of an LLM response."""
    matches = _CODE_FENCE.findall(text)
    if not matches:
        return text.strip()
    return max(matches, key=len).strip()


# ---------------------------------------------------------------------- #
# Coder                                                                   #
# ---------------------------------------------------------------------- #

_CODER_SYS_INITIAL = """You are an expert Python engineer. Write a complete,
self-contained Python program that solves the task described by the user.
Constraints:
- The program must be runnable as `python solution.py` with no arguments.
- It must print a clear success indicator (e.g., `print("OK: ...")`) on success.
- Include enough self-check / assertions to validate correctness.
- Wrap the FINAL program in a single fenced ```python ... ``` code block."""

_CODER_SYS_REVISE = """You are an expert Python engineer fixing a failing program.
You will receive:
  - the previous attempt (Python source)
  - the captured stderr / error message
  - a critic's diagnosis
Produce a corrected program that addresses the failure.
Wrap the FINAL program in a single fenced ```python ... ``` code block."""


@dataclass
class CoderAgent:
    llm: LLMClient

    def initial(self, task: str) -> str:
        text = self.llm.complete(
            [Message("system", _CODER_SYS_INITIAL), Message("user", task)],
            max_tokens=1500, temperature=0.2,
        )
        return _extract_code(text)

    def revise(self, task: str, code: str, error: str, critique: str) -> str:
        user = (
            f"TASK:\n{task}\n\n"
            f"PREVIOUS ATTEMPT:\n```python\n{code}\n```\n\n"
            f"ERROR:\n{error}\n\n"
            f"CRITIQUE:\n{critique}\n"
        )
        text = self.llm.complete(
            [Message("system", _CODER_SYS_REVISE), Message("user", user)],
            max_tokens=1500, temperature=0.3,
        )
        return _extract_code(text)


# ---------------------------------------------------------------------- #
# Executor (not an LLM)                                                   #
# ---------------------------------------------------------------------- #

@dataclass(slots=True)
class ExecutionResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int


class ExecutorAgent:
    """Runs Python code in a subprocess with a hard timeout."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout = timeout_seconds

    def run(self, code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solution.py"
            path.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    capture_output=True, text=True,
                    timeout=self.timeout, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    passed=False,
                    stdout=exc.stdout or "",
                    stderr=f"TIMEOUT: process exceeded {self.timeout}s",
                    returncode=-1,
                )
            return ExecutionResult(
                passed=(proc.returncode == 0),
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )


# ---------------------------------------------------------------------- #
# Critic                                                                  #
# ---------------------------------------------------------------------- #

_CRITIC_SYS = """You are a senior code reviewer. You receive a failing Python
program and its error output. Diagnose the *root cause* (not just the symptom)
and propose a concise fix strategy in 1-3 sentences. Do not write code."""


@dataclass
class CriticAgent:
    llm: LLMClient

    def diagnose(self, code: str, error: str) -> str:
        user = (
            f"CODE:\n```python\n{code}\n```\n\nERROR:\n{error}\n"
        )
        return self.llm.complete(
            [Message("system", _CRITIC_SYS), Message("user", user)],
            max_tokens=400, temperature=0.2,
        )
