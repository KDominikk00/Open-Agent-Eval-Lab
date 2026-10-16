"""Fail if benchmarks/published no longer matches what the current code and tasks produce.

Latency and timestamps vary run to run; task ids, trials, outcomes and tool traces must not.
Usage: python scripts/check_snapshot.py <committed_dir> <rebuilt_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def digest(root: Path) -> dict[str, list[tuple[object, ...]]]:
    out: dict[str, list[tuple[object, ...]]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        out[run_dir.name] = [(r["task_id"], r["trial"], r["success"], json.dumps(r["trace"])) for r in rows]
    return out


def main() -> int:
    committed, rebuilt = digest(Path(sys.argv[1])), digest(Path(sys.argv[2]))
    if committed != rebuilt:
        changed = sorted(k for k in committed.keys() | rebuilt.keys() if committed.get(k) != rebuilt.get(k))
        print(f"benchmarks/published is stale ({', '.join(changed)}). Run scripts/publish_baselines.sh and commit.")
        return 1
    print(f"snapshot current ({len(committed)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
