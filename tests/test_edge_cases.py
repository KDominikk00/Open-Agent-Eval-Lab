import io
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from open_agent_eval.agents import _post_json, get_agent
from open_agent_eval.schemas import TaskSchemaError, task_from_dict
from open_agent_eval.tools import Environment, ToolError, validate_environment_spec

BASE = {
    "task_id": "public.x.t",
    "title": "t",
    "category": "x",
    "instructions": "i",
    "expected_response": "e",
}


def parse(**extra):
    return task_from_dict({**BASE, **extra}, Path("t.json"), inferred_split="public")


class SchemaEdgeCases(unittest.TestCase):
    def test_invalid_fields_rejected(self) -> None:
        bad_records = [
            {"difficulty": "impossible"},
            {"split": "hidden"},
            {"max_steps": 0},
            {"tags": [1]},
            {"grader": {"type": "fuzzy"}},
            {"grader": {"type": "contains"}},
            {"grader": {"type": "exact", "needles": ["a"]}},
            {"grader": {"type": "exact", "allow_extra_keys": True}},
            {"grader": {"type": "exact", "required_calls": ["nope"]}},
            {"grader": {"type": "exact", "final_files": {"a": 1}}},
            {"grader": "exact"},
            {"environment": {"bogus": 1}},
            {"environment": {"files": {"a": 1}}},
            {"environment": {"logs": "x"}},
            {"environment": {"tables": {"t": ["row"]}}},
            {"environment": {"files": {}}, "reference_trajectory": [{"tool": "nope"}]},
        ]
        for extra in bad_records:
            with self.subTest(extra=extra), self.assertRaises(TaskSchemaError):
                parse(**extra)

    def test_root_must_be_object(self) -> None:
        with self.assertRaises(TaskSchemaError):
            task_from_dict([], Path("t.json"))  # type: ignore[arg-type]

    def test_valid_environment_accepted(self) -> None:
        self.assertIsNone(validate_environment_spec({"files": {"a": "b"}, "logs": ["x"], "tables": {"t": [{}]}}))
        self.assertIsNotNone(validate_environment_spec("x"))


class ToolEdgeCases(unittest.TestCase):
    def test_every_tool_reports_bad_input(self) -> None:
        env = Environment({"files": {"a": "b"}, "tables": {"t": [{"k": 1}]}, "logs": ["Hello"]})
        cases = [
            ("read_file", {}),
            ("write_file", {"path": "x"}),
            ("db_query", {"table": "missing"}),
            ("db_query", {"table": "t", "where": "k=1"}),
            ("calculator", {"expression": 3}),
            ("nope", {}),
        ]
        for name, args in cases:
            with self.subTest(tool=name), self.assertRaises(ToolError):
                env.call(name, args)
        with self.assertRaises(ToolError):
            env.call("read_file", [])  # type: ignore[arg-type]

    def test_happy_paths(self) -> None:
        env = Environment({"files": {"a": "b"}, "logs": ["Hello World"], "http": {"u": "body"}})
        self.assertEqual("a", env.call("list_files", {}))
        self.assertEqual("Hello World", env.call("search_logs", {"query": "hello"}))
        self.assertEqual("(no matching log lines)", env.call("search_logs", {"query": "zzz"}))
        self.assertEqual("body", env.call("http_get", {"url": "u"}))
        self.assertEqual("404 Not Found", env.call("http_get", {"url": "other"}))
        self.assertEqual("2.5", env.call("calculator", {"expression": "5 / 2"}))

    def test_environment_does_not_mutate_task_spec(self) -> None:
        spec = {"files": {"a": "b"}}
        Environment(spec).call("write_file", {"path": "a", "content": "changed"})
        self.assertEqual("b", spec["files"]["a"])


class HttpErrorPaths(unittest.TestCase):
    def _call(self):
        return _post_json(url="http://x", payload={}, headers={}, timeout_seconds=1)

    def test_success(self) -> None:
        resp = MagicMock()
        resp.read.return_value = b'{"ok": 1}'
        resp.__enter__.return_value = resp
        with patch("open_agent_eval.agents.request.urlopen", return_value=resp):
            self.assertEqual({"ok": 1}, self._call())

    def test_failures_become_value_errors(self) -> None:
        http_error = urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(b"slow down"))  # type: ignore[arg-type]
        raising = [http_error, urllib.error.URLError("dns")]
        for exc in raising:
            with patch("open_agent_eval.agents.request.urlopen", side_effect=exc), self.assertRaises(ValueError):
                self._call()
        for body in (b"not json", b"[1]"):
            resp = MagicMock()
            resp.read.return_value = body
            resp.__enter__.return_value = resp
            with patch("open_agent_eval.agents.request.urlopen", return_value=resp), self.assertRaises(ValueError):
                self._call()


class AgentFactoryErrors(unittest.TestCase):
    def test_unknown_and_bad_import_specs(self) -> None:
        for spec in ("nope", "no.such.module:Thing", "open_agent_eval.agents:Missing"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                get_agent(spec)

    def test_openai_malformed_responses(self) -> None:
        task = parse()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
            for response in ({}, {"choices": ["x"]}, {"choices": [{"message": "x"}]}):
                with patch("open_agent_eval.agents._post_json", return_value=response), self.assertRaises(ValueError):
                    get_agent("openai-chat").run(task)


if __name__ == "__main__":
    unittest.main()
