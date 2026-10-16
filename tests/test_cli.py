import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_agent_eval import cli

TASKS = str(Path(__file__).resolve().parent.parent / "tasks")
PROFILE = Path(__file__).resolve().parent.parent / "profiles" / "local-smoke.json"


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with patch("sys.argv", ["open-agent-eval", *argv]), contextlib.redirect_stdout(out):
        try:
            cli.main()
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue()


class CliTests(unittest.TestCase):
    def test_agents_lists_offline_policies(self) -> None:
        code, out = run_cli("agents")
        self.assertEqual(0, code)
        for name in ("oracle", "keyword-retriever", "random"):
            self.assertIn(name, out.split())

    def test_validate_ok_and_failure(self) -> None:
        code, out = run_cli("validate", "--tasks-dir", TASKS, "--split", "all")
        self.assertEqual(0, code)
        self.assertIn("Validated", out)
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_cli("validate", "--tasks-dir", tmp)
        self.assertEqual(1, code)
        self.assertIn("[ERROR]", out)

    def test_run_trials_label_and_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results, site = f"{tmp}/results", f"{tmp}/site"
            code, out = run_cli(
                "run", "--tasks-dir", TASKS, "--output-dir", results, "--agent", "random",
                "--split", "public", "--trials", "2", "--label", "smoke", "--run-id", "r1",
            )
            self.assertEqual(0, code)
            self.assertIn("95% CI", out)
            summary = json.loads(Path(results, "r1", "summary.json").read_text())
            self.assertEqual((2, "smoke"), (summary["trials"], summary["label"]))

            code, out = run_cli("leaderboard", "--results-dir", results, "--output-dir", site)
            self.assertEqual(1, code)
            self.assertIn("--include-smoke", out)

            code, _ = run_cli("leaderboard", "--results-dir", results, "--output-dir", site, "--include-smoke")
            self.assertEqual(0, code)
            self.assertTrue(Path(site, "traces.html").exists())

    def test_run_reports_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_cli("run", "--tasks-dir", TASKS, "--output-dir", tmp, "--agent", "nope")
            self.assertEqual(1, code)
            self.assertIn("Unknown agent", out)
            code, out = run_cli("run", "--tasks-dir", TASKS, "--output-dir", tmp, "--trials", "0")
            self.assertEqual(1, code)

    def test_existing_run_id_fails_cleanly_before_running_the_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = ("run", "--tasks-dir", TASKS, "--output-dir", tmp, "--agent", "oracle", "--run-id", "dup")
            self.assertEqual(0, run_cli(*args)[0])
            with patch("open_agent_eval.runner.get_agent") as get_agent:
                code, out = run_cli(*args)
            get_agent.assert_not_called()
        self.assertEqual(1, code)
        self.assertIn("already exists", out)

    def test_run_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_cli(
                "run-profile", "--profile", str(PROFILE), "--tasks-dir", TASKS,
                "--output-dir", f"{tmp}/r", "--leaderboard-output-dir", f"{tmp}/l", "--timestamp", "20260101-000000",
            )
        self.assertEqual(0, code, out)
        self.assertIn("Profile completed", out)

    def test_no_command_prints_help(self) -> None:
        code, _ = run_cli()
        self.assertNotEqual(0, code)


if __name__ == "__main__":
    unittest.main()
