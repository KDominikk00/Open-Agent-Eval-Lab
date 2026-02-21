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
PYTHONPATH=src:. python -m open_agent_eval.cli run \
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
