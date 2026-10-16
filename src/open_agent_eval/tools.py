"""Deterministic, sandboxed tool simulators for agent episodes.

Every tool operates on an in-memory ``Environment`` built from a task's
``environment`` fixture, so runs never touch the real filesystem or network.
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any, Callable

ENVIRONMENT_KEYS = {"files", "logs", "tables", "http"}
MAX_SEARCH_RESULTS = 20


class ToolError(ValueError):
    """Raised when a tool call is invalid; surfaced to the agent as an observation."""


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "read_file": {
        "description": "Read the full text of a file from the sandbox filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "write_file": {
        "description": "Create or overwrite a file in the sandbox filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    "list_files": {
        "description": "List all file paths in the sandbox filesystem.",
        "parameters": {"type": "object", "properties": {}},
    },
    "search_logs": {
        "description": "Case-insensitive substring search over application log lines.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    "db_query": {
        "description": "Read-only lookup of rows in a table, optionally filtered by column equality.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "where": {"type": "object"},
            },
            "required": ["table"],
        },
    },
    "http_get": {
        "description": "Fetch a URL from the simulated network and return the response body.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "calculator": {
        "description": "Evaluate an arithmetic expression (+ - * / // % ** and parentheses).",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

KNOWN_TOOLS = frozenset(TOOL_SPECS)

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ToolError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("unsupported expression")


def evaluate_expression(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except SyntaxError as exc:
        raise ToolError("invalid expression syntax") from exc
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(round(value, 10) if isinstance(value, float) else value)


def validate_environment_spec(spec: Any) -> str | None:
    """Return an error message if ``spec`` is not a valid environment fixture."""

    if not isinstance(spec, dict):
        return "environment must be an object"
    unknown = set(spec) - ENVIRONMENT_KEYS
    if unknown:
        return f"environment has unknown key(s): {', '.join(sorted(unknown))}"

    for key in ("files", "http"):
        value = spec.get(key, {})
        if not isinstance(value, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
        ):
            return f"environment.{key} must map strings to strings"

    logs = spec.get("logs", [])
    if not isinstance(logs, list) or any(not isinstance(line, str) for line in logs):
        return "environment.logs must be a list of strings"

    tables = spec.get("tables", {})
    if not isinstance(tables, dict) or any(
        not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
        for rows in tables.values()
    ):
        return "environment.tables must map table names to lists of row objects"
    return None


class Environment:
    """Mutable sandbox state shared by all tools within one episode."""

    def __init__(self, spec: dict[str, Any], tools: list[str] | None = None) -> None:
        error = validate_environment_spec(spec)
        if error:
            raise ValueError(error)
        self.files: dict[str, str] = dict(spec.get("files", {}))
        self.logs: list[str] = list(spec.get("logs", []))
        self.tables: dict[str, list[dict[str, Any]]] = {
            name: [dict(row) for row in rows] for name, rows in spec.get("tables", {}).items()
        }
        self.http: dict[str, str] = dict(spec.get("http", {}))
        self.enabled = list(tools) if tools else sorted(KNOWN_TOOLS)

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Execute one tool and return its textual observation."""

        if name not in KNOWN_TOOLS:
            raise ToolError(f"unknown tool '{name}'")
        if name not in self.enabled:
            raise ToolError(f"tool '{name}' is not available for this task")
        if not isinstance(args, dict):
            raise ToolError("tool arguments must be an object")
        handler = getattr(self, f"_tool_{name}")
        return handler(args)

    @staticmethod
    def _string_arg(args: dict[str, Any], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str):
            raise ToolError(f"missing or non-string argument '{key}'")
        return value

    def _tool_read_file(self, args: dict[str, Any]) -> str:
        path = self._string_arg(args, "path")
        if path not in self.files:
            raise ToolError(f"file not found: {path}")
        return self.files[path]

    def _tool_write_file(self, args: dict[str, Any]) -> str:
        path = self._string_arg(args, "path")
        self.files[path] = self._string_arg(args, "content")
        return f"wrote {len(self.files[path])} characters to {path}"

    def _tool_list_files(self, args: dict[str, Any]) -> str:
        return "\n".join(sorted(self.files)) or "(no files)"

    def _tool_search_logs(self, args: dict[str, Any]) -> str:
        query = self._string_arg(args, "query").casefold()
        matches = [line for line in self.logs if query in line.casefold()]
        if not matches:
            return "(no matching log lines)"
        return "\n".join(matches[:MAX_SEARCH_RESULTS])

    def _tool_db_query(self, args: dict[str, Any]) -> str:
        table = self._string_arg(args, "table")
        if table not in self.tables:
            raise ToolError(f"unknown table: {table}")
        where = args.get("where") or {}
        if not isinstance(where, dict):
            raise ToolError("'where' must be an object")
        rows = [row for row in self.tables[table] if all(row.get(k) == v for k, v in where.items())]
        return json.dumps(rows, sort_keys=True)

    def _tool_http_get(self, args: dict[str, Any]) -> str:
        url = self._string_arg(args, "url")
        if url not in self.http:
            return "404 Not Found"
        return self.http[url]

    def _tool_calculator(self, args: dict[str, Any]) -> str:
        return evaluate_expression(self._string_arg(args, "expression"))
