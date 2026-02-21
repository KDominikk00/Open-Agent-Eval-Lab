"""Example custom agent for framework extension demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

from open_agent_eval.agents import AgentOutput
from open_agent_eval.schemas import TaskSpec


@dataclass(frozen=True)
class KeywordSpotterAgent:
    """Simple custom agent that echoes required tool hints and key tokens."""

    name: str = "keyword-spotter"

    def run(self, task: TaskSpec) -> AgentOutput:
        tokens: list[str] = []
        for token in (task.category, *task.tags, *task.required_tools):
            token = token.strip()
            if token and token not in tokens:
                tokens.append(token)

        response = " ".join(tokens) if tokens else "no-keywords"
        steps = min(task.max_steps, max(1, len(task.required_tools) + 1))
        return AgentOutput(response=response, tool_calls=list(task.required_tools), steps=steps)


@dataclass(frozen=True)
class FactoryDemoAgent:
    """Example agent returned by a factory function."""

    name: str
    prefix: str

    def run(self, task: TaskSpec) -> AgentOutput:
        return AgentOutput(
            response=f"{self.prefix}:{task.task_id}",
            tool_calls=[],
            steps=1,
        )


def build_agent(
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    timeout_seconds: float = 45.0,
) -> FactoryDemoAgent:
    """Factory-style plugin that accepts framework-injected kwargs."""

    token = model or "no-model"
    prefix = f"{token}|t={temperature}|m={max_output_tokens}|to={timeout_seconds}"
    return FactoryDemoAgent(name="factory-demo", prefix=prefix)
