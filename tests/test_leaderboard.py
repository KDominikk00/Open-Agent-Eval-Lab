import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.leaderboard import generate_leaderboard


class LeaderboardTests(unittest.TestCase):
    def test_generate_leaderboard_outputs_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            output_dir = root / "leaderboard"

            run_a = results_dir / "run-a"
            run_b = results_dir / "run-b"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)

            (run_a / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "agent_name": "echo-reference",
                        "split": "public",
                        "total_tasks": 8,
                        "pass_rate": 1.0,
                        "average_score": 1.0,
                        "average_latency_ms": 0.01,
                        "started_at_utc": "2026-02-21T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            (run_b / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-b",
                        "agent_name": "instruction-snippet",
                        "split": "public",
                        "total_tasks": 8,
                        "pass_rate": 0.25,
                        "average_score": 0.25,
                        "average_latency_ms": 0.02,
                        "started_at_utc": "2026-02-21T00:01:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            generate_leaderboard(results_dir=results_dir, output_dir=output_dir)

            self.assertTrue((output_dir / "leaderboard.json").exists())
            self.assertTrue((output_dir / "leaderboard.md").exists())
            self.assertTrue((output_dir / "leaderboard.html").exists())

            payload = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
            self.assertEqual(2, len(payload["runs"]))
            self.assertEqual("run-a", payload["runs"][0]["run_id"])


if __name__ == "__main__":
    unittest.main()
