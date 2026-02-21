"""CLI entrypoint for Open Agent Eval Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_agent_eval.runner import run_from_directory
from open_agent_eval.validators import validate_tasks_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-agent-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate task files")
    validate_parser.add_argument("--tasks-dir", required=True, help="Directory with JSON task files")

    run_parser = subparsers.add_parser("run", help="Run agent evaluation")
    run_parser.add_argument("--tasks-dir", required=True, help="Directory with JSON task files")
    run_parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")
    run_parser.add_argument(
        "--agent",
        default="echo-reference",
        help="Baseline agent to run (default: echo-reference)",
    )
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run identifier",
    )

    return parser


def _cmd_validate(tasks_dir: Path) -> int:
    report = validate_tasks_dir(tasks_dir)

    if report.is_valid:
        print(f"Validated {len(report.tasks)} task(s) successfully from {tasks_dir}")
        return 0

    for issue in report.issues:
        print(f"[ERROR] {issue.path}: {issue.message}")

    print(f"Validation failed with {len(report.issues)} issue(s)")
    return 1


def _cmd_run(tasks_dir: Path, output_dir: Path, agent_name: str, run_id: str | None) -> int:
    try:
        run_dir, summary = run_from_directory(
            tasks_dir=tasks_dir,
            output_dir=output_dir,
            agent_name=agent_name,
            run_id=run_id,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    print(f"Run completed: {summary.passed_tasks}/{summary.total_tasks} tasks passed")
    print(f"Pass rate: {summary.pass_rate:.2%}")
    print(f"Average latency: {summary.average_latency_ms:.3f} ms")
    print(f"Artifacts: {run_dir}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        raise SystemExit(_cmd_validate(Path(args.tasks_dir)))

    if args.command == "run":
        raise SystemExit(
            _cmd_run(
                tasks_dir=Path(args.tasks_dir),
                output_dir=Path(args.output_dir),
                agent_name=args.agent,
                run_id=args.run_id,
            )
        )

    raise SystemExit(parser.format_help())


if __name__ == "__main__":
    main()
