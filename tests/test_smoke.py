from open_agent_eval.cli import build_parser


def test_cli_has_subcommands() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert "validate" in choices
    assert "run" in choices
