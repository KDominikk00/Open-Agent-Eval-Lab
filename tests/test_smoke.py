import unittest

from open_agent_eval.cli import build_parser


class CliSmokeTests(unittest.TestCase):
    def test_cli_has_subcommands(self) -> None:
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices
        self.assertIn("validate", choices)
        self.assertIn("run", choices)


if __name__ == "__main__":
    unittest.main()
