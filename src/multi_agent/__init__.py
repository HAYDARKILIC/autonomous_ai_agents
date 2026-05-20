"""Week 5 — Multi-agent systems and state management.

Three primitives:

- :class:`AgentState` — shared blackboard, hashed for loop detection.
- :class:`Agent` — a role-specialized LLM wrapper.
- :class:`Orchestrator` — drives the conversation between agents and
  enforces termination (bounded retries, state-hash loop detection,
  explicit success criteria).
"""

from src.multi_agent.agents import CoderAgent, CriticAgent, ExecutorAgent
from src.multi_agent.orchestrator import SelfCorrectingCoder
from src.multi_agent.state import AgentState

__all__ = ["CoderAgent", "CriticAgent", "ExecutorAgent", "SelfCorrectingCoder", "AgentState"]
