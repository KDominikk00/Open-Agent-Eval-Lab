"""Leaderboard generation utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LeaderboardRow:
    run_id: str
    agent_name: str
    split: str
    total_tasks: int
    pass_rate: float
    average_score: float
    average_latency_ms: float
    started_at_utc: str


@dataclass(frozen=True)
class AggregateRow:
    agent_name: str
    split: str
    runs: int
    mean_pass_rate: float
    best_pass_rate: float
    mean_score: float
    mean_latency_ms: float


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    return default


def _build_row(summary: dict[str, Any]) -> LeaderboardRow:
    return LeaderboardRow(
        run_id=str(summary.get("run_id", "unknown-run")),
        agent_name=str(summary.get("agent_name", "unknown-agent")),
        split=str(summary.get("split", "unknown")),
        total_tasks=_to_int(summary.get("total_tasks"), default=0),
        pass_rate=_to_float(summary.get("pass_rate"), default=0.0),
        average_score=_to_float(summary.get("average_score"), default=0.0),
        average_latency_ms=_to_float(summary.get("average_latency_ms"), default=0.0),
        started_at_utc=str(summary.get("started_at_utc", "")),
    )


def _build_aggregates(rows: list[LeaderboardRow]) -> list[AggregateRow]:
    buckets: dict[tuple[str, str], list[LeaderboardRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.agent_name, row.split)].append(row)

    aggregates: list[AggregateRow] = []
    for (agent_name, split), bucket in buckets.items():
        runs = len(bucket)
        pass_rates = [row.pass_rate for row in bucket]
        scores = [row.average_score for row in bucket]
        latencies = [row.average_latency_ms for row in bucket]

        aggregates.append(
            AggregateRow(
                agent_name=agent_name,
                split=split,
                runs=runs,
                mean_pass_rate=round(sum(pass_rates) / runs, 4),
                best_pass_rate=round(max(pass_rates), 4),
                mean_score=round(sum(scores) / runs, 4),
                mean_latency_ms=round(sum(latencies) / runs, 3),
            )
        )

    return sorted(
        aggregates,
        key=lambda row: (row.mean_pass_rate, row.mean_score, -row.mean_latency_ms),
        reverse=True,
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def _render_markdown(rows: list[LeaderboardRow], aggregates: list[AggregateRow]) -> str:
    run_rows = [
        [
            row.run_id,
            row.agent_name,
            row.split,
            str(row.total_tasks),
            f"{row.pass_rate:.2%}",
            f"{row.average_score:.2f}",
            f"{row.average_latency_ms:.3f}",
            row.started_at_utc,
        ]
        for row in rows
    ]

    aggregate_rows = [
        [
            row.agent_name,
            row.split,
            str(row.runs),
            f"{row.mean_pass_rate:.2%}",
            f"{row.best_pass_rate:.2%}",
            f"{row.mean_score:.2f}",
            f"{row.mean_latency_ms:.3f}",
        ]
        for row in aggregates
    ]

    sections = [
        "# Open Agent Eval Leaderboard",
        "",
        "## Run Leaderboard",
        _markdown_table(
            [
                "Run ID",
                "Agent",
                "Split",
                "Tasks",
                "Pass Rate",
                "Avg Score",
                "Avg Latency (ms)",
                "Started (UTC)",
            ],
            run_rows,
        ),
        "",
        "## Agent Aggregate",
        _markdown_table(
            [
                "Agent",
                "Split",
                "Runs",
                "Mean Pass Rate",
                "Best Pass Rate",
                "Mean Score",
                "Mean Latency (ms)",
            ],
            aggregate_rows,
        ),
        "",
    ]
    return "\n".join(sections)


def _render_html(rows: list[LeaderboardRow], aggregates: list[AggregateRow]) -> str:
    run_rows = "\n".join(
        "<tr>"
        f"<td>{row.run_id}</td>"
        f"<td>{row.agent_name}</td>"
        f"<td>{row.split}</td>"
        f"<td>{row.total_tasks}</td>"
        f"<td>{row.pass_rate:.2%}</td>"
        f"<td>{row.average_score:.2f}</td>"
        f"<td>{row.average_latency_ms:.3f}</td>"
        f"<td>{row.started_at_utc}</td>"
        "</tr>"
        for row in rows
    )

    aggregate_rows = "\n".join(
        "<tr>"
        f"<td>{row.agent_name}</td>"
        f"<td>{row.split}</td>"
        f"<td>{row.runs}</td>"
        f"<td>{row.mean_pass_rate:.2%}</td>"
        f"<td>{row.best_pass_rate:.2%}</td>"
        f"<td>{row.mean_score:.2f}</td>"
        f"<td>{row.mean_latency_ms:.3f}</td>"
        "</tr>"
        for row in aggregates
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Open Agent Eval Leaderboard</title>
  <style>
    :root {{
      --bg: #f5f1ea;
      --fg: #1f2933;
      --card: #fffdf8;
      --accent: #b45309;
      --border: #d3c8b5;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #fffdf8 0%, var(--bg) 70%);
      color: var(--fg);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.5;
      padding: 2rem 1rem 3rem;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    h1 {{
      margin-top: 0;
      letter-spacing: 0.02em;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 1rem;
      overflow-x: auto;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    th, td {{
      text-align: left;
      padding: 0.55rem 0.65rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.95rem;
    }}
    th {{
      color: var(--accent);
      font-weight: 700;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .stamp {{
      color: #4b5563;
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Open Agent Eval Leaderboard</h1>
    <p class=\"stamp\">Generated from local run artifacts.</p>

    <section class=\"card\">
      <h2>Run Leaderboard</h2>
      <table>
        <thead>
          <tr>
            <th>Run ID</th><th>Agent</th><th>Split</th><th>Tasks</th><th>Pass Rate</th><th>Avg Score</th><th>Avg Latency (ms)</th><th>Started (UTC)</th>
          </tr>
        </thead>
        <tbody>
          {run_rows}
        </tbody>
      </table>
    </section>

    <section class=\"card\">
      <h2>Agent Aggregate</h2>
      <table>
        <thead>
          <tr>
            <th>Agent</th><th>Split</th><th>Runs</th><th>Mean Pass Rate</th><th>Best Pass Rate</th><th>Mean Score</th><th>Mean Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          {aggregate_rows}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def generate_leaderboard(results_dir: Path, output_dir: Path) -> Path:
    """Generate leaderboard artifacts from run summaries."""

    if not results_dir.exists() or not results_dir.is_dir():
        raise ValueError(f"results directory does not exist: {results_dir}")

    run_dirs = _iter_run_dirs(results_dir)
    if not run_dirs:
        raise ValueError(f"no runs found in: {results_dir}")

    rows: list[LeaderboardRow] = []
    for run_dir in run_dirs:
        summary = _load_json(run_dir / "summary.json")
        rows.append(_build_row(summary))

    rows.sort(key=lambda row: (row.pass_rate, row.average_score, -row.average_latency_ms), reverse=True)
    aggregates = _build_aggregates(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [asdict(row) for row in rows],
        "aggregates": [asdict(row) for row in aggregates],
    }

    json_path = output_dir / "leaderboard.json"
    markdown_path = output_dir / "leaderboard.md"
    html_path = output_dir / "leaderboard.html"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown(rows, aggregates), encoding="utf-8")
    html_path.write_text(_render_html(rows, aggregates), encoding="utf-8")

    return output_dir
