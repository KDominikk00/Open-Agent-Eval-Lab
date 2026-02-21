import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.agents import EchoReferenceAgent
from open_agent_eval.runner import run_benchmark, run_from_directory
from open_agent_eval.schemas import TaskSpec


class RunnerTests(unittest.TestCase):
    def test_run_benchmark_with_reference_agent_passes_all_tasks(self) -> None:
        tasks = [
            TaskSpec(
                task_id="task.one",
                title="Task one",
                category="api_usage",
                instructions="Return hello",
                expected_response="hello",
                required_tools=["tool_a"],
                max_steps=4,
                tags=["smoke"],
            )
        ]

        results, summary = run_benchmark(tasks, EchoReferenceAgent(), run_id="run-test")

        self.assertEqual(1, len(results))
        self.assertTrue(results[0].success)
        self.assertEqual("run-test", summary.run_id)
        self.assertEqual(1, summary.passed_tasks)
        self.assertEqual(1.0, summary.pass_rate)

    def test_run_from_directory_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tasks_dir = tmp_path / "tasks"
            output_dir = tmp_path / "results"
            (tasks_dir / "public" / "api_usage").mkdir(parents=True)

            payload = {
                "task_id": "task.valid",
                "title": "Valid task",
                "category": "api_usage",
                "split": "public",
                "instructions": "Do the thing",
                "expected_response": "Done",
                "required_tools": ["http_docs"],
            }
            (tasks_dir / "public" / "api_usage" / "task.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            run_dir, summary = run_from_directory(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                agent_name="echo-reference",
                run_id="run-20260221-000001",
            )

            self.assertEqual("run-20260221-000001", summary.run_id)
            self.assertTrue((run_dir / "results.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())

            lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(1, len(lines))
            self.assertTrue(json.loads(lines[0])["success"])


if __name__ == "__main__":
    unittest.main()
