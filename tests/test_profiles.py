import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.profiles import execute_profile, load_profile


class ProfileTests(unittest.TestCase):
    def _write_task(self, path: Path, *, task_id: str, split: str) -> None:
        payload = {
            "task_id": task_id,
            "title": "Task",
            "category": "api_usage",
            "split": split,
            "instructions": "Return Done",
            "expected_response": "Done",
            "required_tools": ["http_docs"],
            "grader": {"type": "exact"},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_load_profile_and_execute_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_dir = root / "tasks"
            self._write_task(tasks_dir / "public" / "api_usage" / "task.json", task_id="public.task", split="public")

            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "local-test-profile",
                        "tasks_dir": "tasks",
                        "output_dir": "results",
                        "split": "public",
                        "max_tasks": 1,
                        "generate_leaderboard": True,
                        "leaderboard_output_dir": "leaderboard",
                        "runs": [
                            {
                                "agent": "echo-reference",
                                "run_id_prefix": "echo-public",
                            },
                            {
                                "agent": "tool-hint",
                                "run_id_prefix": "tool-public",
                                "temperature": 0.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            profile = load_profile(profile_path)
            result = execute_profile(profile, profile_path=profile_path, timestamp="20260221-120000")

            self.assertEqual("local-test-profile", result.profile_name)
            self.assertEqual(2, len(result.outcomes))
            self.assertTrue(result.leaderboard_dir)

            run_ids = [outcome.run_id for outcome in result.outcomes]
            self.assertIn("echo-public-20260221-120000", run_ids)
            self.assertIn("tool-public-20260221-120000", run_ids)

            for outcome in result.outcomes:
                self.assertTrue((Path(outcome.artifacts_dir) / "summary.json").exists())

            self.assertTrue((Path(result.leaderboard_dir) / "leaderboard.json").exists())

    def test_load_profile_requires_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "bad-profile",
                        "tasks_dir": "tasks",
                        "output_dir": "results",
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_profile(profile_path)

    def test_load_profile_rejects_non_positive_max_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "bad-profile",
                        "runs": [
                            {
                                "agent": "echo-reference",
                                "max_output_tokens": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_profile(profile_path)

    def test_load_profile_rejects_non_positive_max_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "bad-profile",
                        "max_tasks": 0,
                        "runs": [
                            {
                                "agent": "echo-reference",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_profile(profile_path)


if __name__ == "__main__":
    unittest.main()
