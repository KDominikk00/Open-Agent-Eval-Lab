"""Render docs/assets/results.svg (pass rate with 95% CI per agent) from a results directory.

Dependency-free so it runs anywhere the package does.
Usage: python scripts/render_chart.py benchmarks/published docs/assets/results.svg
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

WIDTH, LEFT, RIGHT, ROW, TOP = 760, 190, 90, 38, 58
BAR_W = WIDTH - LEFT - RIGHT


def load(results_dir: Path) -> list[dict[str, object]]:
    rows = [json.loads((d / "summary.json").read_text()) for d in sorted(results_dir.iterdir()) if d.is_dir()]
    return sorted(rows, key=lambda r: float(r["pass_rate"]), reverse=True)  # type: ignore[arg-type]


def render(rows: list[dict[str, object]]) -> str:
    height = TOP + ROW * len(rows) + 52
    x = lambda v: LEFT + BAR_W * float(v)  # noqa: E731
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" role="img" '
        'aria-labelledby="t d" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
        '<title id="t">Pass rate by agent on the public split</title>',
        '<desc id="d">Horizontal bars show pass rate; whiskers show the 95% Wilson interval over tasks.</desc>',
        f'<rect width="{WIDTH}" height="{height}" rx="10" fill="#ffffff" stroke="#d8dce0"/>',
        f'<text x="20" y="30" font-size="16" font-weight="700" fill="#1c2530">Pass rate by agent (public split, {rows[0]["total_tasks"]} tasks)</text>',
        '<text x="20" y="48" font-size="12" fill="#5b6673">Bar = pass rate. Whisker = 95% Wilson interval over tasks.</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        gx = x(tick)
        out.append(f'<line x1="{gx:.1f}" y1="{TOP - 6}" x2="{gx:.1f}" y2="{TOP + ROW * len(rows)}" stroke="#eceff1"/>')
        out.append(
            f'<text x="{gx:.1f}" y="{TOP + ROW * len(rows) + 18}" font-size="11" text-anchor="middle" '
            f'fill="#5b6673">{tick:.0%}</text>'
        )
    for i, row in enumerate(rows):
        y = TOP + i * ROW
        mid, rate = y + ROW / 2, float(row["pass_rate"])  # type: ignore[arg-type]
        low, high = float(row["pass_rate_ci_low"]), float(row["pass_rate_ci_high"])  # type: ignore[arg-type]
        trials = int(row["trials"])  # type: ignore[call-overload]
        label = escape(str(row["agent_name"])) + (f" ({trials} trials)" if trials > 1 else "")
        out += [
            f'<text x="{LEFT - 10}" y="{mid + 4:.1f}" font-size="13" text-anchor="end" fill="#1c2530">{label}</text>',
            f'<rect x="{LEFT}" y="{mid - 9:.1f}" width="{max(BAR_W * rate, 1.5):.1f}" height="18" rx="3" fill="#c2641b"/>',
            f'<line x1="{x(low):.1f}" y1="{mid:.1f}" x2="{x(high):.1f}" y2="{mid:.1f}" stroke="#1c2530" stroke-width="2"/>',
            f'<line x1="{x(low):.1f}" y1="{mid - 6:.1f}" x2="{x(low):.1f}" y2="{mid + 6:.1f}" stroke="#1c2530" stroke-width="2"/>',
            f'<line x1="{x(high):.1f}" y1="{mid - 6:.1f}" x2="{x(high):.1f}" y2="{mid + 6:.1f}" stroke="#1c2530" stroke-width="2"/>',
            f'<text x="{WIDTH - RIGHT + 12}" y="{mid + 4:.1f}" font-size="13" font-weight="600" fill="#1c2530">{rate:.1%}</text>',
        ]
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    results_dir, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(load(results_dir)), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
