import os
import unittest
from unittest.mock import patch

from open_agent_eval.agents import get_agent, list_available_agents
from open_agent_eval.schemas import TaskSpec


class AgentTests(unittest.TestCase):
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="task.api",
            title="API task",
            category="api_usage",
            instructions="Return a valid curl command.",
            expected_response="curl https://example.com",
            required_tools=["http_docs"],
            max_steps=4,
        )

    def test_available_agents_include_api_backed_agents(self) -> None:
        names = list_available_agents()
        self.assertIn("openai-chat", names)
        self.assertIn("anthropic-chat", names)

    @patch("open_agent_eval.agents._post_json")
    def test_openai_agent_parses_message_and_tool_calls(self, mock_post_json) -> None:
        mock_post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "curl -H 'Accept: application/json' https://example.com/hello",
                        "tool_calls": [
                            {"function": {"name": "http_docs"}},
                        ],
                    }
                }
            ]
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            agent = get_agent("openai-chat", model="gpt-4o-mini")
            output = agent.run(self._task())

        self.assertIn("curl", output.response)
        self.assertEqual(["http_docs"], output.tool_calls)
        self.assertEqual(2, output.steps)

    @patch("open_agent_eval.agents._post_json")
    def test_anthropic_agent_parses_text_and_tool_use(self, mock_post_json) -> None:
        mock_post_json.return_value = {
            "content": [
                {"type": "text", "text": "Use parameterized queries to avoid SQL injection."},
                {"type": "tool_use", "name": "security_guide"},
            ]
        }

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            agent = get_agent("anthropic-chat", model="claude-3-5-haiku-latest")
            output = agent.run(self._task())

        self.assertIn("parameterized", output.response)
        self.assertEqual(["security_guide"], output.tool_calls)
        self.assertEqual(2, output.steps)

    @patch("open_agent_eval.agents._post_json")
    def test_openai_agent_requires_api_key(self, mock_post_json) -> None:
        mock_post_json.return_value = {}

        with patch.dict(os.environ, {}, clear=True):
            agent = get_agent("openai-chat", model="gpt-4o-mini")
            with self.assertRaises(ValueError):
                agent.run(self._task())


if __name__ == "__main__":
    unittest.main()
