"""Baseline agent implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from open_agent_eval.schemas import TaskSpec


@dataclass(frozen=True)
class AgentOutput:
    """Structured output returned by any benchmarked agent."""

    response: str
    tool_calls: list[str]
    steps: int


class Agent(Protocol):
    """Protocol implemented by all agent backends."""

    name: str

    def run(self, task: TaskSpec) -> AgentOutput:
        """Execute one task and return the agent output."""


@dataclass(frozen=True)
class EchoReferenceAgent:
    """Upper-bound baseline that returns the expected reference answer."""

    name: str = "echo-reference"

    def run(self, task: TaskSpec) -> AgentOutput:
        steps = min(task.max_steps, max(1, len(task.required_tools) + 1))
        return AgentOutput(
            response=task.expected_response,
            tool_calls=task.required_tools,
            steps=steps,
        )


@dataclass(frozen=True)
class InstructionSnippetAgent:
    """Weak baseline that copies a snippet from instructions."""

    name: str = "instruction-snippet"

    def run(self, task: TaskSpec) -> AgentOutput:
        snippet = task.instructions.split(".", maxsplit=1)[0].strip()
        if not snippet:
            snippet = task.instructions.strip()
        return AgentOutput(response=snippet, tool_calls=[], steps=1)


def get_agent(agent_name: str) -> Agent:
    """Return a baseline agent by name."""

    registry: dict[str, Agent] = {
        "echo-reference": EchoReferenceAgent(),
        "instruction-snippet": InstructionSnippetAgent(),
    }

    try:
        return registry[agent_name]
    except KeyError as exc:
        options = ", ".join(sorted(registry))
        raise ValueError(f"Unknown agent '{agent_name}'. Available agents: {options}") from exc
