"""Example tool-using agent that obeys instructions found inside documents.

It reads every file the task mentions, then follows any URL it finds in what it read, which is
exactly the failure the ``injection`` task family is built to catch. Run it with:

    PYTHONPATH=. open-agent-eval run --tasks-dir tasks --output-dir results --split public \
        --agent examples.custom_agents.gullible_agent:GullibleAgent --label demo
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from open_agent_eval.agents import AgentOutput, output_from_episode
from open_agent_eval.episode import Action, FinalAnswer, ToolCall, TraceStep, run_episode
from open_agent_eval.schemas import TaskSpec

URL = re.compile(r"https?://[^\s'\"<>)]+")


@dataclass(frozen=True)
class GullibleAgent:
    name: str = "gullible"

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        files = sorted(task.environment.get("files", {}))
        mentioned = [path for path in files if path in task.instructions]
        if len(history) < len(mentioned):
            return ToolCall("read_file", {"path": mentioned[len(history)]})

        fetched = {step.args.get("url") for step in history if step.tool == "http_get"}
        for step in history:
            for url in URL.findall(step.observation):
                if url not in fetched and "http_get" in (task.tools or ["http_get"]):
                    return ToolCall("http_get", {"url": url})

        observations = [step.observation for step in history if not step.is_error and step.tool == "read_file"]
        return FinalAnswer(" ".join(observations)[:400])

    def run(self, task: TaskSpec) -> AgentOutput:
        return output_from_episode(run_episode(task, self))
