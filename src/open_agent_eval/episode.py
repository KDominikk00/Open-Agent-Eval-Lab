"""Multi-step agent episodes: agents act, the runner executes tools and records a trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from open_agent_eval.schemas import TaskSpec
from open_agent_eval.tools import Environment, ToolError


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswer:
    text: str


@dataclass(frozen=True)
class TraceStep:
    """One executed tool call and what the sandbox returned."""

    tool: str
    args: dict[str, Any]
    observation: str
    is_error: bool = False


Action = ToolCall | FinalAnswer


class EpisodeAgent(Protocol):
    """Agents that choose one action at a time given the trace so far."""

    @property
    def name(self) -> str:
        """Display name used in reports."""

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        """Return the next tool call, or the final answer."""


@dataclass(frozen=True)
class EpisodeResult:
    response: str
    trace: list[TraceStep]
    steps: int
    final_files: dict[str, str]
    truncated: bool

    def trace_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self.trace]


def run_episode(task: TaskSpec, agent: EpisodeAgent) -> EpisodeResult:
    """Drive ``agent`` against the task's sandbox until it answers or runs out of steps."""

    env = Environment(task.environment, tools=task.tools)
    trace: list[TraceStep] = []
    steps = 0

    while steps < task.max_steps:
        action = agent.decide(task, list(trace))
        steps += 1

        if isinstance(action, FinalAnswer):
            return EpisodeResult(action.text, trace, steps, dict(env.files), truncated=False)

        try:
            observation = env.call(action.name, action.args)
            trace.append(TraceStep(action.name, action.args, observation))
        except ToolError as exc:
            trace.append(TraceStep(action.name, action.args, f"error: {exc}", is_error=True))

    return EpisodeResult("", trace, steps, dict(env.files), truncated=True)
