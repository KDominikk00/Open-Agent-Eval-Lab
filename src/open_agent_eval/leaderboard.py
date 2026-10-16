"""Leaderboard generation: ranked agent table with confidence intervals, category heatmap, trace viewer."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

SMOKE_LABEL = "smoke"
MAX_TRACE_OBSERVATION_CHARS = 400


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
    trials: int = 1
    label: str | None = None
    pass_rate_ci_low: float = 0.0
    pass_rate_ci_high: float = 1.0
    average_steps: float = 0.0


@dataclass(frozen=True)
class AggregateRow:
    agent_name: str
    split: str
    runs: int
    mean_pass_rate: float
    best_pass_rate: float
    mean_score: float
    mean_latency_ms: float
    headline_run_id: str = ""
    total_tasks: int = 0
    trials: int = 1
    pass_rate_ci_low: float = 0.0
    pass_rate_ci_high: float = 1.0
    average_steps: float = 0.0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _is_smoke(summary: dict[str, Any]) -> bool:
    label = summary.get("label")
    return isinstance(label, str) and label.casefold() == SMOKE_LABEL


def _build_row(summary: dict[str, Any]) -> LeaderboardRow:
    label = summary.get("label")
    return LeaderboardRow(
        run_id=str(summary.get("run_id", "unknown-run")),
        agent_name=str(summary.get("agent_name", "unknown-agent")),
        split=str(summary.get("split", "unknown")),
        total_tasks=_to_int(summary.get("total_tasks"), default=0),
        pass_rate=_to_float(summary.get("pass_rate"), default=0.0),
        average_score=_to_float(summary.get("average_score"), default=0.0),
        average_latency_ms=_to_float(summary.get("average_latency_ms"), default=0.0),
        started_at_utc=str(summary.get("started_at_utc", "")),
        trials=max(1, _to_int(summary.get("trials"), default=1)),
        label=label if isinstance(label, str) else None,
        pass_rate_ci_low=_to_float(summary.get("pass_rate_ci_low"), default=0.0),
        pass_rate_ci_high=_to_float(summary.get("pass_rate_ci_high"), default=1.0),
        average_steps=_to_float(summary.get("average_steps"), default=0.0),
    )


def _build_aggregates(rows: list[LeaderboardRow]) -> list[AggregateRow]:
    buckets: dict[tuple[str, str], list[LeaderboardRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.agent_name, row.split)].append(row)

    aggregates: list[AggregateRow] = []
    for (agent_name, split), bucket in buckets.items():
        runs = len(bucket)
        headline = max(bucket, key=lambda row: row.started_at_utc)
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
                headline_run_id=headline.run_id,
                total_tasks=headline.total_tasks,
                trials=headline.trials,
                pass_rate_ci_low=headline.pass_rate_ci_low,
                pass_rate_ci_high=headline.pass_rate_ci_high,
                average_steps=headline.average_steps,
            )
        )

    return sorted(
        aggregates,
        key=lambda row: (row.mean_pass_rate, row.mean_score, -row.mean_latency_ms),
        reverse=True,
    )


def _ci_text(low: float, high: float) -> str:
    return f"{low:.1%} – {high:.1%}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def _load_categories(results_dir: Path, run_id: str) -> dict[str, float]:
    path = results_dir / run_id / "categories.json"
    if not path.exists():
        return {}
    try:
        items = _load_json(path)
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        str(item["category"]): _to_float(item.get("pass_rate"))
        for item in items
        if isinstance(item, dict) and "category" in item
    }


def _heatmap_data(
    results_dir: Path, aggregates: list[AggregateRow]
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    per_agent = {
        (row.agent_name, row.split): _load_categories(results_dir, row.headline_run_id)
        for row in aggregates
    }
    categories = sorted({category for values in per_agent.values() for category in values})
    return categories, per_agent


def _render_markdown(
    rows: list[LeaderboardRow],
    aggregates: list[AggregateRow],
    categories: list[str],
    per_agent: dict[tuple[str, str], dict[str, float]],
) -> str:
    aggregate_rows = [
        [
            str(rank),
            row.agent_name,
            row.split,
            f"{row.mean_pass_rate:.1%}",
            _ci_text(row.pass_rate_ci_low, row.pass_rate_ci_high),
            f"{row.mean_score:.2f}",
            f"{row.average_steps:.1f}",
            f"{row.total_tasks} × {row.trials}",
            str(row.runs),
        ]
        for rank, row in enumerate(aggregates, start=1)
    ]
    sections = [
        "# Open Agent Eval Leaderboard",
        "",
        "## Agent Ranking",
        "",
        "Pass rate with 95% Wilson interval (tasks are the independent unit). "
        "`Tasks × Trials` describes the headline (most recent) run.",
        "",
        _markdown_table(
            ["#", "Agent", "Split", "Pass Rate", "95% CI", "Avg Score", "Avg Steps", "Tasks × Trials", "Runs"],
            aggregate_rows,
        ),
        "",
    ]

    if categories:
        heat_rows = [
            [row.agent_name, row.split]
            + [
                f"{per_agent[(row.agent_name, row.split)][c]:.0%}"
                if c in per_agent[(row.agent_name, row.split)]
                else "–"
                for c in categories
            ]
            for row in aggregates
        ]
        sections += [
            "## Pass Rate by Category",
            "",
            _markdown_table(["Agent", "Split", *categories], heat_rows),
            "",
        ]

    run_rows = [
        [
            row.run_id,
            row.agent_name,
            row.split,
            str(row.total_tasks),
            str(row.trials),
            f"{row.pass_rate:.2%}",
            f"{row.average_score:.2f}",
            f"{row.average_latency_ms:.3f}",
            row.started_at_utc,
        ]
        for row in rows
    ]
    sections += [
        "## All Runs",
        "",
        _markdown_table(
            ["Run ID", "Agent", "Split", "Tasks", "Trials", "Pass Rate", "Avg Score", "Latency (ms)", "Started (UTC)"],
            run_rows,
        ),
        "",
    ]
    return "\n".join(sections)


_STYLE = """
    :root { --bg:#fafaf7; --fg:#1c2530; --muted:#5b6673; --card:#fff; --border:#dcdcd4; --accent:#a3470a; --bar:#d9a066; }
    @media (prefers-color-scheme: dark) {
      :root { --bg:#14181d; --fg:#e6e9ec; --muted:#9aa5b1; --card:#1b2128; --border:#2c353f; --accent:#f0a35e; --bar:#a3622c; }
    }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; padding:2rem 1rem 3rem; }
    .container { max-width:1100px; margin:0 auto; }
    h1 { margin:0 0 .25rem; } h2 { margin-top:0; }
    nav a { color:var(--accent); margin-right:1rem; }
    .stamp { color:var(--muted); font-size:.9rem; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; margin:1rem 0; overflow-x:auto; }
    table { width:100%; border-collapse:collapse; min-width:640px; }
    th,td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--border); font-size:.92rem; }
    th { color:var(--accent); }
    tr:last-child td { border-bottom:none; }
    .num { font-variant-numeric:tabular-nums; }
    .ci { position:relative; height:10px; min-width:120px; background:var(--border); border-radius:5px; }
    .ci span { position:absolute; top:0; bottom:0; background:var(--bar); border-radius:5px; }
    .ci i { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--fg); }
    .heat td.cell { text-align:center; color:#1c2530; font-weight:600; }
    details { border-top:1px solid var(--border); padding:.4rem 0; }
    summary { cursor:pointer; }
    .ok { color:#1a7f37; } .bad { color:#c0392b; }
    pre { white-space:pre-wrap; word-break:break-word; background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:.5rem; margin:.3rem 0; font-size:.85rem; }
    .step { margin-left:1rem; } .err { color:#c0392b; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>{_STYLE}</style>
</head>
<body>
  <div class="container">
{body}
  </div>
</body>
</html>
"""


def _heat_color(rate: float) -> str:
    hue = 10 + 110 * rate  # red -> green
    return f"hsl({hue:.0f} 60% {78 - 12 * rate:.0f}%)"


def _render_html(
    rows: list[LeaderboardRow],
    aggregates: list[AggregateRow],
    categories: list[str],
    per_agent: dict[tuple[str, str], dict[str, float]],
    has_traces: bool,
) -> str:
    ranking = "\n".join(
        "<tr>"
        f'<td class="num">{rank}</td>'
        f"<td><strong>{escape(row.agent_name)}</strong></td>"
        f"<td>{escape(row.split)}</td>"
        f'<td class="num">{row.mean_pass_rate:.1%}</td>'
        f'<td><div class="ci" role="img" aria-label="95% CI {_ci_text(row.pass_rate_ci_low, row.pass_rate_ci_high)}">'
        f'<span style="left:{row.pass_rate_ci_low * 100:.1f}%;right:{100 - row.pass_rate_ci_high * 100:.1f}%"></span>'
        f'<i style="left:{row.mean_pass_rate * 100:.1f}%"></i></div></td>'
        f'<td class="num">{_ci_text(row.pass_rate_ci_low, row.pass_rate_ci_high)}</td>'
        f'<td class="num">{row.mean_score:.2f}</td>'
        f'<td class="num">{row.average_steps:.1f}</td>'
        f'<td class="num">{row.total_tasks} × {row.trials}</td>'
        "</tr>"
        for rank, row in enumerate(aggregates, start=1)
    )

    heatmap = ""
    if categories:
        head = "".join(f"<th>{escape(c)}</th>" for c in categories)
        body_rows = []
        for row in aggregates:
            values = per_agent[(row.agent_name, row.split)]
            cells = "".join(
                f'<td class="cell num" style="background:{_heat_color(values[c])}">{values[c]:.0%}</td>'
                if c in values
                else '<td class="cell">–</td>'
                for c in categories
            )
            body_rows.append(
                f"<tr><td><strong>{escape(row.agent_name)}</strong> <span class=\"stamp\">{escape(row.split)}</span></td>{cells}</tr>"
            )
        heatmap = f"""
    <section class="card heat">
      <h2>Pass rate by category</h2>
      <table><thead><tr><th>Agent</th>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>
    </section>"""

    run_rows = "\n".join(
        "<tr>"
        f"<td>{escape(row.run_id)}</td><td>{escape(row.agent_name)}</td><td>{escape(row.split)}</td>"
        f'<td class="num">{row.total_tasks}</td><td class="num">{row.trials}</td>'
        f'<td class="num">{row.pass_rate:.2%}</td><td class="num">{row.average_score:.2f}</td>'
        f'<td class="num">{row.average_latency_ms:.3f}</td><td>{escape(row.started_at_utc)}</td>'
        "</tr>"
        for row in rows
    )
    trace_link = '<a href="traces.html">Trace viewer</a>' if has_traces else ""

    body = f"""    <h1>Open Agent Eval Leaderboard</h1>
    <p class="stamp">Pass rates carry 95% Wilson intervals over tasks. Smoke runs are excluded.</p>
    <nav>{trace_link}</nav>

    <section class="card">
      <h2>Agent ranking</h2>
      <table>
        <thead><tr><th>#</th><th>Agent</th><th>Split</th><th>Pass rate</th><th>95% CI</th><th></th><th>Avg score</th><th>Avg steps</th><th>Tasks × trials</th></tr></thead>
        <tbody>{ranking}</tbody>
      </table>
    </section>
{heatmap}
    <section class="card">
      <details open>
        <summary><strong>All runs</strong></summary>
        <table>
          <thead><tr><th>Run ID</th><th>Agent</th><th>Split</th><th>Tasks</th><th>Trials</th><th>Pass rate</th><th>Avg score</th><th>Latency (ms)</th><th>Started (UTC)</th></tr></thead>
          <tbody>{run_rows}</tbody>
        </table>
      </details>
    </section>"""
    return _page("Open Agent Eval Leaderboard", body)


def _truncate(text: str, limit: int = MAX_TRACE_OBSERVATION_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def _render_trace_task(record: dict[str, Any]) -> str:
    ok = bool(record.get("success"))
    mark = '<span class="ok">✓ pass</span>' if ok else '<span class="bad">✗ fail</span>'
    steps = []
    for index, step in enumerate(record.get("trace") or [], start=1):
        if not isinstance(step, dict):
            continue
        cls = "err" if step.get("is_error") else ""
        steps.append(
            f'<div class="step"><strong>{index}. {escape(str(step.get("tool", "?")))}</strong> '
            f"<code>{escape(json.dumps(step.get('args', {}), ensure_ascii=False))}</code>"
            f'<pre class="{cls}">{escape(_truncate(str(step.get("observation", ""))))}</pre></div>'
        )
    trace_html = "".join(steps) or '<p class="stamp">No tool calls (single-turn task).</p>'
    trial = _to_int(record.get("trial"), 0)
    trial_text = f" · trial {trial + 1}" if trial else ""
    return (
        f"<details><summary>{mark} <strong>{escape(str(record.get('task_id', '?')))}</strong> "
        f'<span class="stamp">{escape(str(record.get("difficulty", "")))}{trial_text}</span></summary>'
        f'<p class="stamp">Grader: {escape(str(record.get("grade_reason", "")))}</p>'
        f"{trace_html}"
        f"<p><strong>Response</strong></p><pre>{escape(_truncate(str(record.get('response', '')), 1000))}</pre>"
        f"<p><strong>Expected</strong></p><pre>{escape(_truncate(str(record.get('expected_response', '')), 1000))}</pre>"
        "</details>"
    )


def _render_traces(results_dir: Path, aggregates: list[AggregateRow]) -> str | None:
    sections = []
    for row in aggregates:
        path = results_dir / row.headline_run_id / "results.jsonl"
        if not path.exists():
            continue
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Failures first: they are what a reader wants to inspect.
        records.sort(key=lambda r: (bool(r.get("success")), str(r.get("task_id"))))
        shown = [r for r in records if _to_int(r.get("trial"), 0) == 0]
        passed = sum(1 for r in shown if r.get("success"))
        sections.append(
            f'<section class="card"><h2>{escape(row.agent_name)} '
            f'<span class="stamp">{escape(row.split)} · {passed}/{len(shown)} passed (trial 1)</span></h2>'
            + "".join(_render_trace_task(r) for r in shown)
            + "</section>"
        )
    if not sections:
        return None
    body = (
        '    <h1>Trace viewer</h1><nav><a href="leaderboard.html">← Leaderboard</a></nav>\n'
        '    <p class="stamp">Every tool call and observation for the headline run of each agent. Failures are listed first.</p>\n'
        + "\n".join(sections)
    )
    return _page("Open Agent Eval Traces", body)


def generate_leaderboard(
    results_dir: Path, output_dir: Path, include_smoke: bool = False
) -> Path:
    """Generate leaderboard artifacts from run summaries."""

    if not results_dir.exists() or not results_dir.is_dir():
        raise ValueError(f"results directory does not exist: {results_dir}")

    run_dirs = _iter_run_dirs(results_dir)
    if not run_dirs:
        raise ValueError(f"no runs found in: {results_dir}")

    rows: list[LeaderboardRow] = []
    for run_dir in run_dirs:
        summary = _load_json(run_dir / "summary.json")
        if not include_smoke and _is_smoke(summary):
            continue
        rows.append(_build_row(summary))

    if not rows:
        raise ValueError(f"no non-smoke runs found in: {results_dir} (use --include-smoke)")

    rows.sort(key=lambda row: (row.pass_rate, row.average_score, -row.average_latency_ms), reverse=True)
    aggregates = _build_aggregates(rows)
    categories, per_agent = _heatmap_data(results_dir, aggregates)
    traces_html = _render_traces(results_dir, aggregates)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [asdict(row) for row in rows],
        "aggregates": [asdict(row) for row in aggregates],
        "categories": {
            f"{agent}|{split}": values for (agent, split), values in per_agent.items()
        },
    }

    (output_dir / "leaderboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "leaderboard.md").write_text(
        _render_markdown(rows, aggregates, categories, per_agent), encoding="utf-8"
    )
    (output_dir / "leaderboard.html").write_text(
        _render_html(rows, aggregates, categories, per_agent, has_traces=traces_html is not None),
        encoding="utf-8",
    )
    traces_path = output_dir / "traces.html"
    if traces_html is not None:
        traces_path.write_text(traces_html, encoding="utf-8")
    elif traces_path.exists():
        traces_path.unlink()

    return output_dir
