"""CLI entrypoint for Open Agent Eval Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_agent_eval.agents import list_available_agents
from open_agent_eval.leaderboard import generate_leaderboard
from open_agent_eval.profiles import execute_profile, load_profile
from open_agent_eval.runner import run_from_directory
from open_agent_eval.validators import validate_tasks_dir


def _split_arg(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-agent-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate task files")
    validate_parser.add_argument("--tasks-dir", required=True, help="Dataset root directory")
    validate_parser.add_argument(
        "--split",
        default="all",
        choices=["public", "hidden", "all"],
        help="Validate one split or all splits (default: all)",
    )

    run_parser = subparsers.add_parser("run", help="Run agent evaluation")
    run_parser.add_argument("--tasks-dir", required=True, help="Dataset root directory")
    run_parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")
    run_parser.add_argument(
        "--agent",
        default="echo-reference",
        choices=list_available_agents(),
        help="Baseline agent to run",
    )
    run_parser.add_argument(
        "--split",
        default="public",
        choices=["public", "hidden", "all"],
        help="Task split to run (default: public)",
    )
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run identifier",
    )
    run_parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optionally limit the number of tasks after sorting by task_id",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for API-backed agents",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for API-backed agents (default: 0.0)",
    )
    run_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Maximum output tokens for API-backed agents (default: 512)",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="HTTP timeout for API-backed agents (default: 45)",
    )

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Build leaderboard artifacts")
    leaderboard_parser.add_argument("--results-dir", required=True, help="Directory containing run folders")
    leaderboard_parser.add_argument(
        "--output-dir",
        default="leaderboard",
        help="Directory where leaderboard artifacts are written",
    )

    run_profile_parser = subparsers.add_parser("run-profile", help="Run a benchmark profile JSON")
    run_profile_parser.add_argument("--profile", required=True, help="Path to a benchmark profile JSON file")
    run_profile_parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp suffix used for generated run IDs",
    )
    run_profile_parser.add_argument(
        "--tasks-dir",
        default=None,
        help="Optional override for tasks directory",
    )
    run_profile_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for run artifacts directory",
    )
    run_profile_parser.add_argument(
        "--leaderboard-output-dir",
        default=None,
        help="Optional override for leaderboard output directory",
    )
    run_profile_parser.add_argument(
        "--skip-leaderboard",
        action="store_true",
        help="Skip leaderboard regeneration after profile runs",
    )

    agents_parser = subparsers.add_parser("agents", help="List available baseline agents")
    agents_parser.set_defaults(command="agents")

    return parser


def _cmd_validate(tasks_dir: Path, split: str | None) -> int:
    report = validate_tasks_dir(tasks_dir, only_split=split)

    if report.is_valid:
        split_label = split or "all"
        print(f"Validated {len(report.tasks)} task(s) successfully from {tasks_dir} ({split_label})")
        return 0

    for issue in report.issues:
        print(f"[ERROR] {issue.path}: {issue.message}")

    print(f"Validation failed with {len(report.issues)} issue(s)")
    return 1


def _cmd_run(
    tasks_dir: Path,
    output_dir: Path,
    agent_name: str,
    model: str | None,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: float,
    split: str | None,
    run_id: str | None,
    max_tasks: int | None,
) -> int:
    try:
        run_dir, summary = run_from_directory(
            tasks_dir=tasks_dir,
            output_dir=output_dir,
            agent_name=agent_name,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            only_split=split,
            run_id=run_id,
            max_tasks=max_tasks,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    print(f"Run completed for split '{summary.split}' with agent '{summary.agent_name}'")
    print(f"Tasks passed: {summary.passed_tasks}/{summary.total_tasks} ({summary.pass_rate:.2%})")
    print(f"Average score: {summary.average_score:.2f}")
    print(f"Average latency: {summary.average_latency_ms:.3f} ms")
    print(f"Average tool precision/recall: {summary.average_tool_precision:.2f}/{summary.average_tool_recall:.2f}")
    print(f"Artifacts: {run_dir}")
    return 0


def _cmd_leaderboard(results_dir: Path, output_dir: Path) -> int:
    try:
        path = generate_leaderboard(results_dir=results_dir, output_dir=output_dir)
    except ValueError as exc:
        print(str(exc))
        return 1

    print(f"Leaderboard generated at {path}")
    return 0


def _cmd_run_profile(
    profile_path: Path,
    timestamp: str | None,
    tasks_dir_override: Path | None,
    output_dir_override: Path | None,
    leaderboard_output_dir_override: Path | None,
    skip_leaderboard: bool,
) -> int:
    try:
        profile = load_profile(profile_path)
        result = execute_profile(
            profile,
            profile_path=profile_path,
            timestamp=timestamp,
            tasks_dir_override=tasks_dir_override,
            output_dir_override=output_dir_override,
            leaderboard_output_dir_override=leaderboard_output_dir_override,
            skip_leaderboard=skip_leaderboard,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    print(f"Profile completed: {result.profile_name}")
    for outcome in result.outcomes:
        summary = outcome.summary
        print(
            f"- {outcome.run_label}: {summary.passed_tasks}/{summary.total_tasks} "
            f"({summary.pass_rate:.2%}) -> {outcome.artifacts_dir}"
        )

    if result.leaderboard_dir:
        print(f"Leaderboard refreshed at {result.leaderboard_dir}")
    elif skip_leaderboard:
        print("Leaderboard generation skipped")

    return 0


def _cmd_agents() -> int:
    for agent_name in list_available_agents():
        print(agent_name)
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        raise SystemExit(_cmd_validate(Path(args.tasks_dir), _split_arg(args.split)))

    if args.command == "run":
        raise SystemExit(
            _cmd_run(
                tasks_dir=Path(args.tasks_dir),
                output_dir=Path(args.output_dir),
                agent_name=args.agent,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
                split=_split_arg(args.split),
                run_id=args.run_id,
                max_tasks=args.max_tasks,
            )
        )

    if args.command == "leaderboard":
        raise SystemExit(_cmd_leaderboard(Path(args.results_dir), Path(args.output_dir)))

    if args.command == "run-profile":
        raise SystemExit(
            _cmd_run_profile(
                profile_path=Path(args.profile),
                timestamp=args.timestamp,
                tasks_dir_override=Path(args.tasks_dir) if args.tasks_dir else None,
                output_dir_override=Path(args.output_dir) if args.output_dir else None,
                leaderboard_output_dir_override=Path(args.leaderboard_output_dir)
                if args.leaderboard_output_dir
                else None,
                skip_leaderboard=args.skip_leaderboard,
            )
        )

    if args.command == "agents":
        raise SystemExit(_cmd_agents())

    raise SystemExit(parser.format_help())


if __name__ == "__main__":
    main()
