#!/usr/bin/env bash
# Regenerate the committed baseline results in benchmarks/published/.
# Fully offline: no API keys, no network. Stochastic agents use 5 seeded trials.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmarks/published
rm -rf "$OUT" benchmarks/site && mkdir -p "$OUT"
CLI="python -m open_agent_eval.cli"
export PYTHONPATH=src

$CLI validate --tasks-dir tasks --split all

run() { # agent trials
  $CLI run --tasks-dir tasks --output-dir "$OUT" --agent "$1" --split public \
    --trials "$2" --run-id "baseline-$1-public"
}

run oracle 1
run keyword-retriever 1
run random 5
run echo-reference 1
run tool-hint 1
run instruction-snippet 1

$CLI leaderboard --results-dir "$OUT" --output-dir benchmarks/site
python scripts/render_chart.py "$OUT" docs/assets/results.svg
