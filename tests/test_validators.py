import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.validators import validate_tasks_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class ValidatorTests(unittest.TestCase):
    def test_valid_task_directory_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            _write_json(
                tasks_dir / "ok.json",
                {
                    "task_id": "task.valid",
                    "title": "Valid task",
                    "category": "api_usage",
                    "instructions": "Do the thing",
                    "expected_response": "Done",
                },
            )

            report = validate_tasks_dir(tasks_dir)
            self.assertTrue(report.is_valid)
            self.assertEqual(1, len(report.tasks))

    def test_missing_required_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            _write_json(
                tasks_dir / "missing.json",
                {
                    "task_id": "task.missing",
                    "title": "Missing instructions",
                    "category": "api_usage",
                    "expected_response": "Done",
                },
            )

            report = validate_tasks_dir(tasks_dir)
            self.assertFalse(report.is_valid)
            self.assertEqual(1, len(report.issues))
            self.assertIn("instructions", report.issues[0].message)

    def test_duplicate_task_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp)
            payload = {
                "task_id": "task.same",
                "title": "Task",
                "category": "api_usage",
                "instructions": "Do the thing",
                "expected_response": "Done",
            }
            _write_json(tasks_dir / "first.json", payload)
            _write_json(tasks_dir / "second.json", payload)

            report = validate_tasks_dir(tasks_dir)
            self.assertFalse(report.is_valid)
            self.assertEqual(1, len(report.issues))
            self.assertIn("duplicate", report.issues[0].message)


if __name__ == "__main__":
    unittest.main()
