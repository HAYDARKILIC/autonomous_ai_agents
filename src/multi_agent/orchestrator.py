"""Self-correcting code-generation orchestrator.

Topology
--------

    User task
        │
        ▼
    ┌─────────┐
    │  Coder  │◀──────────────────┐
    └────┬────┘                   │
         │ code                   │ revise(code, error, critique)
         ▼                        │
    ┌──────────┐                  │
    │ Executor │── result (pass/fail, stderr)
    └────┬─────┘                  │
         │ fail                   │
         ▼                        │
    ┌──────────┐                  │
    │  Critic  │── critique ──────┘
    └──────────┘
         │ pass
         ▼
       SUCCESS

Termination guarantees
----------------------
- ``max_iterations`` is a hard budget on Coder revisions.
- ``AgentState.detect_loop`` halts on repeated (code, error) fingerprints,
  catching the failure mode where the Critic and Coder oscillate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.multi_agent.agents import CoderAgent, CriticAgent, ExecutorAgent
from src.multi_agent.state import AgentState
from src.utils.llm_client import build_client
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class CoderRunResult:
    final_code: str
    success: bool
    iterations: int
    halted_reason: str  # success | max_iterations | loop_detected


class SelfCorrectingCoder:
    """Coordinates the Coder–Executor–Critic loop."""

    def __init__(
        self,
        max_iterations: int = 5,
        timeout_seconds: float = 10.0,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        provider = llm_provider or os.environ.get("LLM_PROVIDER", "anthropic")
        llm = build_client(provider, llm_model)
        self.coder = CoderAgent(llm)
        self.executor = ExecutorAgent(timeout_seconds=timeout_seconds)
        self.critic = CriticAgent(llm)
        self.max_iterations = max_iterations

    def solve(self, task: str) -> CoderRunResult:
        state = AgentState(task=task)
        code = self.coder.initial(task)
        state.code = code
        state.append("coder", code, role_event="initial")

        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration
            log.info("multi_agent_iteration", iteration=iteration)

            result = self.executor.run(state.code or "")
            state.append("executor", f"passed={result.passed} rc={result.returncode}")

            if result.passed:
                return CoderRunResult(
                    final_code=state.code or "",
                    success=True,
                    iterations=iteration,
                    halted_reason="success",
                )

            error = result.stderr or result.stdout or "no output"
            state.last_error = error

            if state.detect_loop():
                log.warning("multi_agent_loop_detected", iteration=iteration)
                return CoderRunResult(
                    final_code=state.code or "",
                    success=False,
                    iterations=iteration,
                    halted_reason="loop_detected",
                )

            critique = self.critic.diagnose(state.code or "", error)
            state.append("critic", critique)

            revised = self.coder.revise(task, state.code or "", error, critique)
            state.code = revised
            state.append("coder", revised, role_event="revise")

        return CoderRunResult(
            final_code=state.code or "",
            success=False,
            iterations=self.max_iterations,
            halted_reason="max_iterations",
        )
