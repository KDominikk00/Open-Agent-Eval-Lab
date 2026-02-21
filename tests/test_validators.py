import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.validators import validate_tasks_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ValidatorTests(unittest.TestCase):
    def test_valid_nested_task_directory_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            _write_json(
                tasks_dir / "public" / "api_usage" / "ok.json",
                {
                    "task_id": "task.valid",
                    "title": "Valid task",
                    "category": "api_usage",
                    "split": "public",
                    "instructions": "Do the thing",
                    "expected_response": "Done",
                },
            )

            report = validate_tasks_dir(tasks_dir)
            self.assertTrue(report.is_valid)
            self.assertEqual(1, len(report.tasks))

    def test_file_outside_split_folder_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            _write_json(
                tasks_dir / "orphan.json",
                {
                    "task_id": "task.orphan",
                    "title": "Orphan task",
                    "category": "api_usage",
                    "instructions": "Do the thing",
                    "expected_response": "Done",
                },
            )

            report = validate_tasks_dir(tasks_dir)
            self.assertFalse(report.is_valid)
            self.assertIn("tasks/public", report.issues[0].message)

    def test_split_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            _write_json(
                tasks_dir / "public" / "api_usage" / "bad.json",
                {
                    "task_id": "task.bad",
                    "title": "Bad task",
                    "category": "api_usage",
                    "split": "hidden",
                    "instructions": "Do the thing",
                    "expected_response": "Done",
                },
            )

            report = validate_tasks_dir(tasks_dir)
            self.assertFalse(report.is_valid)
            self.assertIn("split mismatch", report.issues[0].message)

    def test_duplicate_task_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            payload = {
                "task_id": "task.same",
                "title": "Task",
                "category": "api_usage",
                "split": "public",
                "instructions": "Do the thing",
                "expected_response": "Done",
            }
            _write_json(tasks_dir / "public" / "api_usage" / "first.json", payload)
            _write_json(tasks_dir / "hidden" / "api_usage" / "second.json", {**payload, "split": "hidden"})

            report = validate_tasks_dir(tasks_dir)
            self.assertFalse(report.is_valid)
            self.assertEqual(1, len(report.issues))
            self.assertIn("duplicate", report.issues[0].message)

    def test_only_split_filters_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            _write_json(
                tasks_dir / "public" / "api_usage" / "public.json",
                {
                    "task_id": "task.public",
                    "title": "Task",
                    "category": "api_usage",
                    "split": "public",
                    "instructions": "Do",
                    "expected_response": "Done",
                },
            )
            _write_json(
                tasks_dir / "hidden" / "api_usage" / "hidden.json",
                {
                    "task_id": "task.hidden",
                    "title": "Task",
                    "category": "api_usage",
                    "split": "hidden",
                    "instructions": "Do",
                    "expected_response": "Done",
                },
            )

            report = validate_tasks_dir(tasks_dir, only_split="public")
            self.assertTrue(report.is_valid)
            self.assertEqual(1, len(report.tasks))
            self.assertEqual("task.public", report.tasks[0].task_id)


if __name__ == "__main__":
    unittest.main()
