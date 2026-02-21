import unittest

from open_agent_eval.graders import compute_tool_match, grade_response
from open_agent_eval.schemas import GraderSpec, TaskSpec


class GraderTests(unittest.TestCase):
    def test_exact_grader_matches_ignoring_case_and_spacing(self) -> None:
        task = TaskSpec(
            task_id="t1",
            title="Exact",
            category="c",
            instructions="i",
            expected_response="Hello   World",
            grader=GraderSpec(type="exact", ignore_case=True, collapse_whitespace=True),
        )

        result = grade_response(task, " hello world ")
        self.assertTrue(result.success)

    def test_contains_grader_requires_all_needles(self) -> None:
        task = TaskSpec(
            task_id="t2",
            title="Contains",
            category="c",
            instructions="i",
            expected_response="unused",
            grader=GraderSpec(type="contains", needles=["alpha", "beta"]),
        )

        ok = grade_response(task, "alpha and beta are present")
        bad = grade_response(task, "only alpha")
        self.assertTrue(ok.success)
        self.assertFalse(bad.success)

    def test_json_grader_supports_subset_mode(self) -> None:
        task = TaskSpec(
            task_id="t3",
            title="Json",
            category="c",
            instructions="i",
            expected_response='{"status":"ok"}',
            grader=GraderSpec(type="json", allow_extra_keys=True),
        )

        ok = grade_response(task, '{"status":"ok","count":3}')
        bad = grade_response(task, '{"status":"nope"}')
        self.assertTrue(ok.success)
        self.assertFalse(bad.success)

    def test_tool_precision_and_recall(self) -> None:
        precision, recall = compute_tool_match(["logs", "db"], ["logs", "metrics"])
        self.assertEqual(0.5, precision)
        self.assertEqual(0.5, recall)


if __name__ == "__main__":
    unittest.main()
