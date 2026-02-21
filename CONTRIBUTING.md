# Contributing to Open Agent Eval Lab

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run tests before submitting:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Adding a benchmark task

1. Copy `docs/TASK_TEMPLATE.json` into the appropriate split/category folder:
   - `tasks/public/<category>/`
   - `tasks/hidden/<category>/`
2. Set a globally unique `task_id`.
3. Choose the correct grader type for the expected output.
4. Keep instructions realistic and concise.
5. Validate the dataset:

```bash
PYTHONPATH=src python -m open_agent_eval.cli validate --tasks-dir tasks --split all
```

## Adding or changing agents

1. Implement agent logic in `src/open_agent_eval/agents.py`.
2. Register the agent name in the internal registry.
3. Ensure deterministic behavior for baseline comparability.
4. Add or update tests when behavior changes.

## Pull request expectations

- All tests pass locally.
- Dataset validation passes.
- Changes include documentation updates when interfaces change.
- Commit messages should be clear and human-readable.

## Issue templates

Use the repository issue templates when opening:

- Bug reports
- Task proposals
- Feature requests
