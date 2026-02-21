"""CLI entrypoint for Open Agent Eval Lab."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-agent-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate task files")
    validate_parser.add_argument("--tasks-dir", required=True, help="Directory with JSON task files")

    run_parser = subparsers.add_parser("run", help="Run agent evaluation")
    run_parser.add_argument("--tasks-dir", required=True, help="Directory with JSON task files")
    run_parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(f"Not implemented yet: {args.command}")


if __name__ == "__main__":
    main()
