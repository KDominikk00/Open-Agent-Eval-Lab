import json
import tempfile
import unittest
from pathlib import Path

from open_agent_eval.leaderboard import generate_leaderboard
from open_agent_eval.runner import run_from_directory
from open_agent_eval.stats import bootstrap_mean_interval, wilson_interval

TASKS = Path(__file__).resolve().parent.parent / "tasks"


class IntervalTests(unittest.TestCase):
    def test_wilson_known_value(self) -> None:
        low, high = wilson_interval(0.5, 100)
        self.assertAlmostEqual(0.4038, low, places=3)
        self.assertAlmostEqual(0.5962, high, places=3)

    def test_wilson_stays_in_bounds_at_extremes(self) -> None:
        low, high = wilson_interval(1.0, 5)
        self.assertLess(low, 1.0)
        self.assertEqual(1.0, high)
        low, high = wilson_interval(0.0, 5)
        self.assertEqual(0.0, low)
        self.assertGreater(high, 0.0)

    def test_wilson_narrows_with_n(self) -> None:
        small = wilson_interval(0.6, 10)
        large = wilson_interval(0.6, 1000)
        self.assertGreater(small[1] - small[0], large[1] - large[0])

    def test_bootstrap_is_seeded_and_brackets_mean(self) -> None:
        values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        first = bootstrap_mean_interval(values)
        self.assertEqual(first, bootstrap_mean_interval(values))
        self.assertLess(first[0], 0.625)
        self.assertGreater(first[1], 0.625)

    def test_bootstrap_constant_values(self) -> None:
        self.assertEqual((1.0, 1.0), bootstrap_mean_interval([1.0, 1.0, 1.0]))


def _read_results(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]


class TrialAndReportTests(unittest.TestCase):
    def _run(self, out: Path, agent: str, **kwargs):
        return run_from_directory(
            tasks_dir=TASKS, output_dir=out, agent_name=agent, only_split="public", **kwargs
        )

    def test_trials_reseed_random_agent_and_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_a, summary = self._run(Path(tmp), "random", run_id="a", trials=3)
            run_b, _ = self._run(Path(tmp), "random", run_id="b", trials=3)
            self.assertEqual(3, summary.trials)
            self.assertEqual(summary.total_tasks * 3, summary.total_attempts)
            first, second = _read_results(run_a), _read_results(run_b)
            self.assertEqual([r["success"] for r in first], [r["success"] for r in second])
            self.assertEqual({0, 1, 2}, {r["trial"] for r in first})
            traces = {t: [json.dumps(r["trace"]) for r in first if r["trial"] == t] for t in (0, 1)}
            self.assertNotEqual(traces[0], traces[1])

    def test_summary_has_intervals_and_breakdowns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, summary = self._run(Path(tmp), "oracle", run_id="o")
            self.assertEqual(1.0, summary.pass_rate)
            self.assertEqual(1.0, summary.pass_rate_ci_high)
            self.assertLess(summary.pass_rate_ci_low, 1.0)
            breakdowns = json.loads((run_dir / "breakdowns.json").read_text())
            self.assertEqual({"category", "difficulty", "grader_type"}, set(breakdowns))
            self.assertIn("contains", {b["category"] for b in breakdowns["grader_type"]})

    def test_invalid_trials_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            self._run(Path(tmp), "oracle", trials=0)

    def test_leaderboard_hides_smoke_and_renders_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results, site = Path(tmp) / "r", Path(tmp) / "site"
            self._run(results, "oracle", run_id="oracle-1")
            self._run(results, "random", run_id="random-1", trials=2)
            self._run(results, "tool-hint", run_id="hint-smoke", label="smoke", max_tasks=2)

            generate_leaderboard(results, site)
            payload = json.loads((site / "leaderboard.json").read_text())
            self.assertEqual({"oracle", "random"}, {a["agent_name"] for a in payload["aggregates"]})
            self.assertEqual("oracle", payload["aggregates"][0]["agent_name"])
            html = (site / "leaderboard.html").read_text()
            self.assertIn("traces.html", html)
            self.assertIn("Pass rate by category", html)
            traces = (site / "traces.html").read_text()
            self.assertIn("search_logs", traces)

            generate_leaderboard(results, site, include_smoke=True)
            payload = json.loads((site / "leaderboard.json").read_text())
            self.assertIn("tool-hint", {a["agent_name"] for a in payload["aggregates"]})

    def test_trace_viewer_escapes_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "r" / "x"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({"run_id": "x", "agent_name": "<b>a</b>", "split": "public"}))
            (run / "results.jsonl").write_text(
                json.dumps({"task_id": "t", "success": False, "response": "<script>alert(1)</script>", "trace": []}) + "\n"
            )
            generate_leaderboard(Path(tmp) / "r", Path(tmp) / "site")
            for name in ("leaderboard.html", "traces.html"):
                text = (Path(tmp) / "site" / name).read_text()
                self.assertNotIn("<script>alert", text)
                self.assertNotIn("<b>a</b>", text)


class PublishedSnapshotTests(unittest.TestCase):
    def test_published_baselines_match_current_dataset(self) -> None:
        """Fails when tasks change but scripts/publish_baselines.sh was not re-run."""

        from open_agent_eval.validators import validate_tasks_dir

        published = TASKS.parent / "benchmarks" / "published"
        self.assertTrue(published.exists(), "run scripts/publish_baselines.sh")
        public_ids = sorted(t.task_id for t in validate_tasks_dir(TASKS, only_split="public").tasks)
        for run_dir in sorted(published.iterdir()):
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(public_ids, sorted(config["selected_task_ids"]), run_dir.name)


if __name__ == "__main__":
    unittest.main()
