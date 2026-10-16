# Custom Agent Extensions

Open Agent Eval Lab supports custom agents via import paths.

## Agent specification format

Use:

- `module.path:SymbolName`

Examples:

- `examples.custom_agents.keyword_spotter:KeywordSpotterAgent`
- `my_package.agent_factory:build_agent`

## Required interface

The resolved symbol must be callable and return an object with:

- `name` (string, optional)
- `run(task)` method

`run(task)` may return:

- `AgentOutput`
- `str`
- `dict` with fields: `response`, `tool_calls`, `steps`

## Running with a custom agent

```bash
PYTHONPATH=. open-agent-eval run \
  --tasks-dir tasks \
  --output-dir results \
  --agent examples.custom_agents.keyword_spotter:KeywordSpotterAgent \
  --split public \
  --max-tasks 3
```

## Optional constructor parameters

When available in your callable signature, the framework can inject:

- `model`
- `temperature`
- `max_output_tokens`
- `timeout_seconds`

## Tool-using agents

Tasks that declare an `environment` are run as multi-step episodes. An agent can opt in by exposing
`decide(task, history)` and delegating `run` to `run_episode`:

```python
from dataclasses import dataclass

from open_agent_eval.agents import AgentOutput, output_from_episode
from open_agent_eval.episode import FinalAnswer, ToolCall, TraceStep, run_episode
from open_agent_eval.schemas import TaskSpec


@dataclass(frozen=True)
class GrepThenAnswer:
    name: str = "grep-then-answer"

    def decide(self, task: TaskSpec, history: list[TraceStep]):
        if not history:
            return ToolCall("search_logs", {"query": "error"})
        return FinalAnswer(history[-1].observation)

    def run(self, task: TaskSpec) -> AgentOutput:
        return output_from_episode(run_episode(task, self))
```

- `history` holds every previous call with its observation (`TraceStep`), including tool errors.
- The runner enforces `max_steps`; an episode that never answers is graded as a failure.
- Tool arguments are validated by the sandbox. Mistakes become error observations, not exceptions.
- Stochastic agents should implement `with_trial(index)` returning a reseeded copy, so `--trials` is reproducible.
- Single-turn tasks (no `environment`) still call `run(task)` directly, so one agent class can handle both.
