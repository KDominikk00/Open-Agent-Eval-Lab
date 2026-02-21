# Release Checklist

Use this checklist before publishing a new release.

## 1. Verify code and tests

- Run unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

- Validate benchmark tasks:

```bash
PYTHONPATH=src python -m open_agent_eval.cli validate --tasks-dir tasks --split all
```

- Run local smoke profile:

```bash
PYTHONPATH=src python -m open_agent_eval.cli run-profile \
  --profile profiles/local-smoke.json \
  --timestamp 20260221-200000
```

## 2. Verify packaging metadata

- Ensure `pyproject.toml` version matches `src/open_agent_eval/__init__.py`.
- If you add `project.urls`, confirm repository links are correct.

## 3. Regenerate public artifacts

- Build leaderboard from latest results:

```bash
PYTHONPATH=src python -m open_agent_eval.cli leaderboard \
  --results-dir results \
  --output-dir leaderboard
```

## 4. Documentation pass

- Ensure README examples still run.
- Update `CHANGELOG.md` with release notes.
- Update `ROADMAP.md` if priorities changed.

## 5. Prepare release commit/tag

- Commit release updates with a clear message.
- Create a git tag (example):

```bash
git tag v0.4.0
```

- Push commits and tags from your local machine when ready.
