import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval import (
    build_leaderboard,
    run_benchmark,
    validate_dataset,
)


class ApiTests(unittest.TestCase):
    def test_validate_dataset_and_run_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_dir = root / "tasks"
            output_dir = root / "results"
            leaderboard_dir = root / "leaderboard"
            (tasks_dir / "public" / "api_usage").mkdir(parents=True)

            payload = {
                "task_id": "task.api",
                "title": "Task",
                "category": "api_usage",
                "split": "public",
                "instructions": "Return Done",
                "expected_response": "Done",
                "grader": {"type": "exact"},
            }
            (tasks_dir / "public" / "api_usage" / "task.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            report = validate_dataset(tasks_dir, split="public")
            self.assertTrue(report.is_valid)
            self.assertEqual(1, len(report.tasks))

            run_dir, summary = run_benchmark(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                agent="echo-reference",
                split="public",
                run_id="api-smoke-run",
            )
            self.assertEqual("api-smoke-run", summary.run_id)
            self.assertTrue((run_dir / "summary.json").exists())

            generated = build_leaderboard(results_dir=output_dir, output_dir=leaderboard_dir)
            self.assertTrue((generated / "leaderboard.json").exists())


if __name__ == "__main__":
    unittest.main()
