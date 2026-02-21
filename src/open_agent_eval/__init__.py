"""Open Agent Eval Lab package."""

from open_agent_eval.api import (
    build_leaderboard,
    load_benchmark_profile,
    run_benchmark,
    run_benchmark_profile,
    validate_dataset,
)

__all__ = [
    "__version__",
    "build_leaderboard",
    "load_benchmark_profile",
    "run_benchmark",
    "run_benchmark_profile",
    "validate_dataset",
]

__version__ = "0.4.0"
