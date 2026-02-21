import unittest

from open_agent_eval.agents import AgentOutput, get_agent
from open_agent_eval.schemas import TaskSpec


class AgentPluginTests(unittest.TestCase):
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="public.plugin.example",
            title="Plugin task",
            category="reasoning",
            instructions="Return output",
            expected_response="output",
            required_tools=["logs"],
            tags=["triage"],
            max_steps=4,
        )

    def test_custom_agent_class_import_path(self) -> None:
        agent = get_agent("examples.custom_agents.keyword_spotter:KeywordSpotterAgent")
        output = agent.run(self._task())

        self.assertIsInstance(output, AgentOutput)
        self.assertIn("reasoning", output.response)
        self.assertIn("logs", output.tool_calls)

    def test_custom_agent_factory_import_path_receives_kwargs(self) -> None:
        agent = get_agent(
            "examples.custom_agents.keyword_spotter:build_agent",
            model="gpt-test",
            temperature=0.25,
            max_output_tokens=222,
            timeout_seconds=12.0,
        )

        output = agent.run(self._task())
        self.assertIn("gpt-test", output.response)
        self.assertIn("t=0.25", output.response)

    def test_unknown_agent_without_import_path_fails(self) -> None:
        with self.assertRaises(ValueError):
            get_agent("does-not-exist")


if __name__ == "__main__":
    unittest.main()
