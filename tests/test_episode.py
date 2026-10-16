import os
import unittest
from pathlib import Path
from unittest.mock import patch

from open_agent_eval.agents import get_agent
from open_agent_eval.episode import FinalAnswer, ToolCall, run_episode
from open_agent_eval.graders import grade_task
from open_agent_eval.runner import run_benchmark
from open_agent_eval.schemas import TaskSchemaError, task_from_dict
from open_agent_eval.tools import Environment, ToolError, evaluate_expression

RECORD = {
    "task_id": "public.tool_use.timeout-cause",
    "title": "Find the timeout cause",
    "category": "tool_use",
    "instructions": "Why did checkout time out? Search the logs, then report the upstream service.",
    "expected_response": "payments-gateway",
    "max_steps": 4,
    "environment": {
        "logs": ["INFO checkout ok", "ERROR checkout timeout calling payments-gateway after 30s"],
        "files": {"notes.txt": "n/a"},
    },
    "tools": ["search_logs", "read_file", "write_file"],
    "reference_trajectory": [{"tool": "search_logs", "args": {"query": "timeout"}}],
    "grader": {
        "type": "contains",
        "needles": ["payments-gateway"],
        "required_calls": ["search_logs"],
        "forbidden_calls": ["write_file"],
    },
}


def make_task(**overrides):
    record = {**RECORD, **overrides}
    return task_from_dict(record, Path("t.json"), inferred_split="public")


class ToolTests(unittest.TestCase):
    def test_calculator_is_safe(self) -> None:
        self.assertEqual("14", evaluate_expression("2 + 3 * 4"))
        for bad in ("__import__('os')", "2 ** 9999", "1/0", "abc"):
            with self.assertRaises(ToolError):
                evaluate_expression(bad)

    def test_environment_is_isolated_and_restricted(self) -> None:
        env = Environment({"files": {"a.txt": "hi"}}, tools=["read_file", "write_file"])
        self.assertEqual("hi", env.call("read_file", {"path": "a.txt"}))
        env.call("write_file", {"path": "b.txt", "content": "x"})
        self.assertEqual("x", env.files["b.txt"])
        with self.assertRaises(ToolError):
            env.call("search_logs", {"query": "x"})
        with self.assertRaises(ToolError):
            env.call("read_file", {"path": "missing"})

    def test_db_query_filters_rows(self) -> None:
        env = Environment({"tables": {"users": [{"id": 1, "plan": "pro"}, {"id": 2, "plan": "free"}]}})
        self.assertEqual('[{"id": 2, "plan": "free"}]', env.call("db_query", {"table": "users", "where": {"plan": "free"}}))


class SchemaTests(unittest.TestCase):
    def test_rejects_unknown_tool_and_orphan_tools(self) -> None:
        with self.assertRaises(TaskSchemaError):
            make_task(tools=["rm_rf"])
        with self.assertRaises(TaskSchemaError):
            make_task(environment={}, tools=["read_file"], reference_trajectory=[])

    def test_rejects_trajectory_outside_enabled_tools(self) -> None:
        with self.assertRaises(TaskSchemaError):
            make_task(reference_trajectory=[{"tool": "db_query", "args": {"table": "x"}}])


class EpisodeTests(unittest.TestCase):
    def test_oracle_passes_and_records_trace(self) -> None:
        task = make_task()
        results, summary = run_benchmark([task], get_agent("oracle"), "r", "public")
        self.assertEqual(1.0, summary.pass_rate)
        self.assertEqual("search_logs", results[0].trace[0]["tool"])

    def test_forbidden_call_fails_even_with_right_answer(self) -> None:
        grade = grade_task(make_task(), "payments-gateway", ["search_logs", "write_file"], {})
        self.assertFalse(grade.success)
        self.assertIn("forbidden", grade.reason)

    def test_call_order_and_final_files(self) -> None:
        task = make_task(
            grader={
                "type": "contains",
                "needles": ["ok"],
                "call_order": ["read_file", "write_file"],
                "final_files": {"out.txt": "done"},
            }
        )
        self.assertTrue(grade_task(task, "ok", ["read_file", "write_file"], {"out.txt": "Done"}).success)
        self.assertFalse(grade_task(task, "ok", ["write_file", "read_file"], {"out.txt": "done"}).success)
        self.assertFalse(grade_task(task, "ok", ["read_file", "write_file"], {}).success)

    def test_step_budget_truncates_episode(self) -> None:
        class Looper:
            name = "looper"

            def decide(self, task, history):
                return ToolCall("list_files", {})

        result = run_episode(make_task(max_steps=3), Looper())
        self.assertTrue(result.truncated)
        self.assertEqual(3, result.steps)
        self.assertEqual("", result.response)

    def test_invalid_tool_call_becomes_error_observation(self) -> None:
        class Clumsy:
            name = "clumsy"

            def decide(self, task, history):
                return ToolCall("db_query", {}) if not history else FinalAnswer("x")

        result = run_episode(make_task(), Clumsy())
        self.assertTrue(result.trace[0].is_error)

    def test_policy_ordering_oracle_beats_random(self) -> None:
        task = make_task()
        rates = {}
        for name in ("oracle", "keyword-retriever", "random"):
            _, summary = run_benchmark([task], get_agent(name), "r", "public")
            rates[name] = summary.pass_rate
        self.assertGreaterEqual(rates["oracle"], rates["keyword-retriever"])
        self.assertGreaterEqual(rates["keyword-retriever"], rates["random"])

    def test_random_agent_is_deterministic(self) -> None:
        task = make_task()
        first = run_episode(task, get_agent("random"))
        second = run_episode(task, get_agent("random"))
        self.assertEqual(first.trace, second.trace)


class ApiToolLoopTests(unittest.TestCase):
    @patch("open_agent_eval.agents._post_json")
    def test_openai_tool_loop(self, mock_post) -> None:
        mock_post.side_effect = [
            {"choices": [{"message": {"tool_calls": [
                {"id": "c", "function": {"name": "search_logs", "arguments": '{"query": "timeout"}'}}]}}]},
            {"choices": [{"message": {"content": "payments-gateway"}}]},
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            output = get_agent("openai-chat").run(make_task())
        self.assertEqual(["search_logs"], output.tool_calls)
        self.assertEqual("payments-gateway", output.response)
        second_messages = mock_post.call_args_list[1].kwargs["payload"]["messages"]
        self.assertEqual("tool", second_messages[-1]["role"])
        self.assertIn("payments-gateway", second_messages[-1]["content"])

    @patch("open_agent_eval.agents._post_json")
    def test_anthropic_tool_loop(self, mock_post) -> None:
        mock_post.side_effect = [
            {"content": [{"type": "tool_use", "id": "t", "name": "search_logs", "input": {"query": "timeout"}}]},
            {"content": [{"type": "text", "text": "payments-gateway"}]},
        ]
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            output = get_agent("anthropic-chat").run(make_task())
        self.assertEqual(["search_logs"], output.tool_calls)
        self.assertEqual("payments-gateway", output.response)
        last = mock_post.call_args_list[1].kwargs["payload"]["messages"][-1]
        self.assertEqual("tool_result", last["content"][0]["type"])


if __name__ == "__main__":
    unittest.main()
