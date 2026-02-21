import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.agents import EchoReferenceAgent
from open_agent_eval.runner import run_benchmark, run_from_directory
from open_agent_eval.schemas import GraderSpec, TaskSpec


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
                grader=GraderSpec(type="exact"),
            )
        ]

        results, summary = run_benchmark(tasks, EchoReferenceAgent(), run_id="run-test", split_label="public")

        self.assertEqual(1, len(results))
        self.assertTrue(results[0].success)
        self.assertEqual("run-test", summary.run_id)
        self.assertEqual(1, summary.passed_tasks)
        self.assertEqual(1.0, summary.pass_rate)
        self.assertEqual(1.0, summary.average_score)

    def test_run_from_directory_writes_artifacts_and_filters_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tasks_dir = tmp_path / "tasks"
            output_dir = tmp_path / "results"
            (tasks_dir / "public" / "api_usage").mkdir(parents=True)
            (tasks_dir / "hidden" / "api_usage").mkdir(parents=True)

            public_payload = {
                "task_id": "task.public",
                "title": "Public task",
                "category": "api_usage",
                "split": "public",
                "instructions": "Do the thing",
                "expected_response": "Done",
                "required_tools": ["http_docs"],
            }
            hidden_payload = {
                "task_id": "task.hidden",
                "title": "Hidden task",
                "category": "api_usage",
                "split": "hidden",
                "instructions": "Do the thing",
                "expected_response": "Done",
                "required_tools": ["http_docs"],
            }

            (tasks_dir / "public" / "api_usage" / "public.json").write_text(
                json.dumps(public_payload), encoding="utf-8"
            )
            (tasks_dir / "hidden" / "api_usage" / "hidden.json").write_text(
                json.dumps(hidden_payload), encoding="utf-8"
            )

            run_dir, summary = run_from_directory(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                agent_name="echo-reference",
                run_id="run-20260221-000001",
                only_split="public",
            )

            self.assertEqual("run-20260221-000001", summary.run_id)
            self.assertEqual("public", summary.split)
            self.assertEqual(1, summary.total_tasks)
            self.assertTrue((run_dir / "results.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "categories.json").exists())
            self.assertTrue((run_dir / "config.json").exists())

            lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(1, len(lines))
            row = json.loads(lines[0])
            self.assertEqual("task.public", row["task_id"])
            self.assertTrue(row["success"])

            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual("public", config["split"])


if __name__ == "__main__":
    unittest.main()
