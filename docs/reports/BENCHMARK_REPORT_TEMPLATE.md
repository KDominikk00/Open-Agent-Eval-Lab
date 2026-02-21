# Benchmark Report: <Title>

## 1. Executive Summary

- Date: <YYYY-MM-DD>
- Repository commit: `<commit-hash>`
- Dataset split: `<public|hidden|all>`
- Overall winner: `<agent/model>`
- Key result: `<one-sentence result>`

## 2. Run Configuration

| Field | Value |
| --- | --- |
| Profile file | `<path>` |
| Tasks directory | `<path>` |
| Results directory | `<path>` |
| Leaderboard directory | `<path>` |
| Max tasks | `<value>` |

### Model configurations

| Label | Agent | Model | Temperature | Max output tokens | Timeout (s) |
| --- | --- | --- | --- | --- | --- |
| Model A | `<agent>` | `<model>` | `<value>` | `<value>` | `<value>` |
| Model B | `<agent>` | `<model>` | `<value>` | `<value>` | `<value>` |

## 3. Quantitative Results

### Aggregate leaderboard

| Rank | Run ID | Agent | Model | Split | Pass Rate | Avg Score | Avg Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `<run-id>` | `<agent>` | `<model>` | `<split>` | `<xx.xx%>` | `<x.xx>` | `<x.xxx>` |
| 2 | `<run-id>` | `<agent>` | `<model>` | `<split>` | `<xx.xx%>` | `<x.xx>` | `<x.xxx>` |

### Category performance

| Category | Model A Pass Rate | Model B Pass Rate | Delta |
| --- | --- | --- | --- |
| api_usage | `<xx.xx%>` | `<xx.xx%>` | `<+/-xx.xx%>` |
| debugging | `<xx.xx%>` | `<xx.xx%>` | `<+/-xx.xx%>` |
| security | `<xx.xx%>` | `<xx.xx%>` | `<+/-xx.xx%>` |

## 4. Error Analysis

### Top failure patterns

1. `<pattern>`
2. `<pattern>`
3. `<pattern>`

### Example failures

- Task ID: `<task-id>`
  - Expected: `<expected>`
  - Actual: `<actual>`
  - Grade reason: `<reason>`
- Task ID: `<task-id>`
  - Expected: `<expected>`
  - Actual: `<actual>`
  - Grade reason: `<reason>`

## 5. Operational Notes

- Any API rate limits observed: `<yes/no + details>`
- Any transient failures/retries: `<details>`
- Any config caveats: `<details>`

## 6. Conclusions and Next Actions

1. `<next action 1>`
2. `<next action 2>`
3. `<next action 3>`

## 7. Reproducibility Appendix

```bash
# Example command set used for this report
export OPENAI_API_KEY="<set-me>"
export ANTHROPIC_API_KEY="<set-me>"
PYTHONPATH=src python -m open_agent_eval.cli run-profile \
  --profile profiles/real-models-public.json
```

Artifacts referenced:

- `results/<run-id>/summary.json`
- `results/<run-id>/categories.json`
- `leaderboard/leaderboard.json`
- `leaderboard/leaderboard.html`
