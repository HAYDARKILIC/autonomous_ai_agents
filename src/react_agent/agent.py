"""The ReAct agent: a finite-loop Thought-Action-Observation runner.

This is a pure-Python implementation. No agent framework. The control
flow is a `while` loop that:

    1. asks the LLM for the next step,
    2. parses the response into a :class:`Step`,
    3. executes the tool if the step is non-final,
    4. appends the observation to the conversation,
    5. terminates on :class:`Step.is_final` or when ``max_steps`` is
       exhausted.

Failure modes and mitigations
-----------------------------
- **Unparseable LLM output** → :class:`ParseError` caught; a corrective
  user message is injected and the loop retries with a small budget.
- **Tool error** → the exception is converted to an Observation so the
  LLM can recover. Tool *crashes* (programmer errors) propagate.
- **Infinite loops** → bounded by ``max_steps``.
- **Action hallucination** (action name not in registry) → handled like
  a tool error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mcp_core.tool_registry import Tool, ToolRegistry
from src.react_agent.parser import ParseError, Step, parse_step
from src.utils.llm_client import LLMClient, Message
from src.utils.logging import get_logger

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are a careful reasoning agent that solves problems by
interleaving Thought, Action, and Observation steps.

You have access to the following tools:
{tool_block}

You must reply in EXACTLY one of the following two formats:

FORMAT A (use a tool):
Thought: <your reasoning>
Action: <tool name from the list above>
Action Input: <a single JSON object matching the tool's input schema>

FORMAT B (you are done):
Thought: <your reasoning>
Final Answer: <your final answer to the user>

Hard rules:
- Do not write anything outside these two formats.
- Action Input must be valid JSON, on a single line if possible.
- Never invent tool names. Use only the tools listed above.
- When you have enough information to answer, switch to FORMAT B.
"""


def _format_tool_block(tools: ToolRegistry) -> str:
    lines = []
    for descriptor in tools.list_all():
        lines.append(f"- {descriptor['name']}: {descriptor['description']}")
        lines.append(f"  input schema: {descriptor['inputSchema']}")
    return "\n".join(lines)


@dataclass(slots=True)
class TraceEntry:
    """One step of the agent's execution trace."""

    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    is_final: bool
    final_answer: str | None = None


@dataclass(slots=True)
class AgentRun:
    """The full result of a single ``run`` invocation."""

    answer: str
    trace: list[TraceEntry] = field(default_factory=list)
    steps: int = 0
    halted_reason: str = "final_answer"  # final_answer | max_steps | parse_failure


class ReActAgent:
    """A pure-Python ReAct agent."""

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        max_steps: int = 10,
        parse_retry_budget: int = 2,
    ) -> None:
        self.llm = llm
        self.tools = ToolRegistry()
        for tool in tools:
            self.tools.register(tool)
        self.max_steps = max_steps
        self.parse_retry_budget = parse_retry_budget

    def run(self, task: str) -> AgentRun:
        system = _SYSTEM_PROMPT.format(tool_block=_format_tool_block(self.tools))
        history: list[Message] = [Message("system", system), Message("user", task)]
        trace: list[TraceEntry] = []

        for step_idx in range(1, self.max_steps + 1):
            try:
                step = self._llm_step(history)
            except ParseError as exc:
                log.warning("react_parse_error", step=step_idx, error=str(exc))
                return AgentRun(
                    answer=f"[agent halted: {exc}]",
                    trace=trace,
                    steps=step_idx,
                    halted_reason="parse_failure",
                )

            log.info(
                "react_step",
                step=step_idx,
                action=step.action,
                is_final=step.is_final,
            )

            if step.is_final:
                trace.append(
                    TraceEntry(
                        thought=step.thought, action=None, action_input=None,
                        observation=None, is_final=True,
                        final_answer=step.final_answer,
                    )
                )
                return AgentRun(
                    answer=step.final_answer or "",
                    trace=trace,
                    steps=step_idx,
                    halted_reason="final_answer",
                )

            assert step.action is not None and step.action_input is not None
            observation = self._execute_action(step.action, step.action_input)
            trace.append(
                TraceEntry(
                    thought=step.thought, action=step.action,
                    action_input=step.action_input,
                    observation=observation, is_final=False,
                )
            )

            # Append the assistant turn and the observation to the chat history.
            history.append(Message("assistant", self._render_step(step)))
            history.append(Message("user", f"Observation: {observation}"))

        return AgentRun(
            answer="[agent halted: max_steps exceeded]",
            trace=trace,
            steps=self.max_steps,
            halted_reason="max_steps",
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _llm_step(self, history: list[Message]) -> Step:
        last_error: ParseError | None = None
        for attempt in range(self.parse_retry_budget + 1):
            text = self.llm.complete(
                history if attempt == 0 else history + [
                    Message("user", f"Your last reply could not be parsed: {last_error}. "
                                    "Reply again in the required format.")
                ],
                max_tokens=1024,
                temperature=0.2,
                stop=["Observation:"],
            )
            try:
                return parse_step(text)
            except ParseError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _execute_action(self, action: str, action_input: dict) -> str:
        if action not in self.tools:
            return f"ERROR: unknown tool {action!r}. Available: {list(self.tools._tools.keys())}"
        try:
            result = self.tools.get(action).invoke(action_input)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        import json
        return json.dumps(result, ensure_ascii=False, default=str)

    @staticmethod
    def _render_step(step: Step) -> str:
        import json
        return (
            f"Thought: {step.thought}\n"
            f"Action: {step.action}\n"
            f"Action Input: {json.dumps(step.action_input)}"
        )
