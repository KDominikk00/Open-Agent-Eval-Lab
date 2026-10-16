"""Baseline and API-backed agent implementations."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib import error, request

from open_agent_eval.episode import (
    Action,
    EpisodeResult,
    FinalAnswer,
    ToolCall,
    TraceStep,
    run_episode,
)
from open_agent_eval.schemas import TaskSpec
from open_agent_eval.tools import KNOWN_TOOLS, TOOL_SPECS


@dataclass(frozen=True)
class AgentOutput:
    """Structured output returned by any benchmarked agent."""

    response: str
    tool_calls: list[str]
    steps: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    final_files: dict[str, str] = field(default_factory=dict)


def output_from_episode(result: EpisodeResult) -> AgentOutput:
    """Convert an episode result into the flat ``AgentOutput`` consumed by the runner."""

    return AgentOutput(
        response=result.response,
        tool_calls=[step.tool for step in result.trace],
        steps=result.steps,
        trace=result.trace_as_dicts(),
        final_files=result.final_files,
    )


class Agent(Protocol):
    """Protocol implemented by all agent backends."""

    @property
    def name(self) -> str:
        """Display name used in reports."""

    def run(self, task: TaskSpec) -> AgentOutput:
        """Execute one task and return the agent output."""


@dataclass(frozen=True)
class EchoReferenceAgent:
    """Upper-bound baseline that returns the expected reference answer."""

    name: str = "echo-reference"

    def run(self, task: TaskSpec) -> AgentOutput:
        steps = min(task.max_steps, max(1, len(task.required_tools) + 1))
        return AgentOutput(
            response=task.expected_response,
            tool_calls=task.required_tools,
            steps=steps,
        )


@dataclass(frozen=True)
class InstructionSnippetAgent:
    """Weak baseline that copies a snippet from instructions."""

    name: str = "instruction-snippet"

    def run(self, task: TaskSpec) -> AgentOutput:
        snippet = task.instructions.split(".", maxsplit=1)[0].strip()
        if not snippet:
            snippet = task.instructions.strip()
        return AgentOutput(response=snippet, tool_calls=[], steps=1)


@dataclass(frozen=True)
class ToolHintAgent:
    """Baseline that prioritizes naming the expected tools in its response."""

    name: str = "tool-hint"

    def run(self, task: TaskSpec) -> AgentOutput:
        if task.required_tools:
            response = f"Use tools: {', '.join(task.required_tools)}"
            tool_calls = list(task.required_tools)
            steps = min(task.max_steps, len(task.required_tools) + 1)
        else:
            response = "No external tools required"
            tool_calls = []
            steps = 1

        return AgentOutput(response=response, tool_calls=tool_calls, steps=steps)


@dataclass(frozen=True)
class ImportedAgentAdapter:
    """Adapter for user-provided agent objects loaded from import paths."""

    name: str
    _delegate: Any

    def run(self, task: TaskSpec) -> AgentOutput:
        run_method = getattr(self._delegate, "run", None)
        if not callable(run_method):
            raise ValueError("Imported agent is missing a callable run(task) method")

        raw_output = run_method(task)
        return _coerce_agent_output(raw_output)


def _task_prompt(task: TaskSpec) -> str:
    required_tools = ", ".join(task.required_tools) if task.required_tools else "none"
    return (
        "Solve the following benchmark task.\n"
        f"Task ID: {task.task_id}\n"
        f"Category: {task.category}\n"
        f"Required tools: {required_tools}\n"
        f"Instructions: {task.instructions}\n"
        "Return only the final answer without extra preamble."
    )


def _episode_tool_names(task: TaskSpec) -> list[str]:
    return sorted(task.tools) if task.tools else sorted(KNOWN_TOOLS)


def _episode_prompt(task: TaskSpec) -> str:
    return (
        "Solve the following task using the provided tools when helpful. "
        "When you are done, reply with only the final answer and no tool call.\n"
        f"Task ID: {task.task_id}\nInstructions: {task.instructions}"
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _post_json(
    *, url: str, payload: dict[str, Any], headers: dict[str, str], timeout_seconds: float
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"upstream API error ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise ValueError(f"network error while calling upstream API: {exc.reason}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response from upstream API: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("upstream API response must be a JSON object")

    return parsed


def _resolve_api_key(*, explicit_key: str | None, env_name: str) -> str:
    if explicit_key:
        return explicit_key

    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    raise ValueError(f"Missing API key. Set {env_name} or provide an explicit key.")


def _extract_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text.strip())
        return "\n".join(part for part in parts if part).strip()

    return ""


def _extract_tool_calls_openai(message: dict[str, Any]) -> list[str]:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    names: list[str] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


@dataclass(frozen=True)
class OpenAIChatAgent:
    """Agent backed by OpenAI Chat Completions API."""

    name: str = "openai-chat"
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_seconds: float = 45.0
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        api_key = _resolve_api_key(explicit_key=self.api_key, env_name="OPENAI_API_KEY")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _model(self) -> str:
        return self.model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    @staticmethod
    def _first_message(response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI response missing choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("OpenAI response choice is malformed")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI response missing message payload")
        return message

    def run(self, task: TaskSpec) -> AgentOutput:
        if task.environment:
            return output_from_episode(run_episode(task, self))

        payload: dict[str, Any] = {
            "model": self._model(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are an evaluation agent. Return concise, correct final answers.",
                },
                {"role": "user", "content": _task_prompt(task)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        response = _post_json(
            url=self.endpoint,
            payload=payload,
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )

        message = self._first_message(response)
        response_text = _extract_openai_content(message.get("content"))
        tool_calls = _extract_tool_calls_openai(message)
        steps = min(task.max_steps, max(1, len(tool_calls) + 1))

        return AgentOutput(response=response_text, tool_calls=tool_calls, steps=steps)

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        """One turn of a native tool-calling conversation, rebuilt from the trace."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an evaluation agent. Use tools, then answer concisely."},
            {"role": "user", "content": _episode_prompt(task)},
        ]
        for index, step in enumerate(history):
            call_id = f"call_{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": step.tool, "arguments": json.dumps(step.args)},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": step.observation})

        payload: dict[str, Any] = {
            "model": self._model(),
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": TOOL_SPECS[name]["description"],
                        "parameters": TOOL_SPECS[name]["parameters"],
                    },
                }
                for name in _episode_tool_names(task)
            ],
            "parallel_tool_calls": False,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        response = _post_json(
            url=self.endpoint,
            payload=payload,
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )
        message = self._first_message(response)

        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls and isinstance(raw_calls[0], dict):
            function = raw_calls[0].get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                return ToolCall(function["name"], _parse_arguments(function.get("arguments")))
        return FinalAnswer(_extract_openai_content(message.get("content")))


def _extract_anthropic_content(content: Any) -> tuple[str, list[str]]:
    if not isinstance(content, list):
        return "", []

    texts: list[str] = []
    tool_calls: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        elif item_type == "tool_use":
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                tool_calls.append(name.strip())

    return "\n".join(texts).strip(), tool_calls


@dataclass(frozen=True)
class AnthropicChatAgent:
    """Agent backed by Anthropic Messages API."""

    name: str = "anthropic-chat"
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_seconds: float = 45.0
    api_key: str | None = None
    endpoint: str = "https://api.anthropic.com/v1/messages"
    anthropic_version: str = "2023-06-01"

    def _headers(self) -> dict[str, str]:
        api_key = _resolve_api_key(explicit_key=self.api_key, env_name="ANTHROPIC_API_KEY")
        return {
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

    def _model(self) -> str:
        return self.model or os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"

    def run(self, task: TaskSpec) -> AgentOutput:
        if task.environment:
            return output_from_episode(run_episode(task, self))

        payload: dict[str, Any] = {
            "model": self._model(),
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": _task_prompt(task)}],
        }

        response = _post_json(
            url=self.endpoint,
            payload=payload,
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )

        response_text, tool_calls = _extract_anthropic_content(response.get("content"))
        if not response_text and not tool_calls:
            raise ValueError("Anthropic response missing content blocks")

        steps = min(task.max_steps, max(1, len(tool_calls) + 1))
        return AgentOutput(response=response_text, tool_calls=tool_calls, steps=steps)

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        """One turn of a native tool-use conversation, rebuilt from the trace."""

        messages: list[dict[str, Any]] = [{"role": "user", "content": _episode_prompt(task)}]
        for index, step in enumerate(history):
            tool_id = f"toolu_{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": tool_id, "name": step.tool, "input": step.args}],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": step.observation,
                            "is_error": step.is_error,
                        }
                    ],
                }
            )

        payload: dict[str, Any] = {
            "model": self._model(),
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "tools": [
                {
                    "name": name,
                    "description": TOOL_SPECS[name]["description"],
                    "input_schema": TOOL_SPECS[name]["parameters"],
                }
                for name in _episode_tool_names(task)
            ],
        }
        response = _post_json(
            url=self.endpoint,
            payload=payload,
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )

        content = response.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("Anthropic response missing content blocks")
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("name"), str):
                return ToolCall(block["name"], _parse_arguments(block.get("input")))
        text, _ = _extract_anthropic_content(content)
        return FinalAnswer(text)


def _registry() -> dict[str, Agent]:
    return {name: factory() for name, factory in _builtin_agent_factories().items()}


def _builtin_agent_factories() -> dict[str, Callable[..., Agent]]:
    from open_agent_eval.policies import KeywordRetrieverAgent, OracleAgent, RandomAgent

    return {
        "oracle": lambda **_: OracleAgent(),
        "keyword-retriever": lambda **_: KeywordRetrieverAgent(),
        "random": lambda **_: RandomAgent(),
        "echo-reference": lambda **_: EchoReferenceAgent(),
        "instruction-snippet": lambda **_: InstructionSnippetAgent(),
        "tool-hint": lambda **_: ToolHintAgent(),
        "openai-chat": lambda **kwargs: OpenAIChatAgent(
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_output_tokens", 512),
            timeout_seconds=kwargs.get("timeout_seconds", 45.0),
        ),
        "anthropic-chat": lambda **kwargs: AnthropicChatAgent(
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_output_tokens", 512),
            timeout_seconds=kwargs.get("timeout_seconds", 45.0),
        ),
    }


def list_available_agents() -> list[str]:
    """List built-in agents."""

    return sorted(_builtin_agent_factories())


def _coerce_agent_output(raw_output: Any) -> AgentOutput:
    if isinstance(raw_output, AgentOutput):
        return raw_output

    if isinstance(raw_output, str):
        return AgentOutput(response=raw_output, tool_calls=[], steps=1)

    if isinstance(raw_output, dict):
        response = raw_output.get("response", "")
        if not isinstance(response, str):
            response = str(response)

        raw_tool_calls = raw_output.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise ValueError("Imported agent output field 'tool_calls' must be a list")
        tool_calls: list[str] = []
        for item in raw_tool_calls:
            if not isinstance(item, str):
                raise ValueError("Imported agent output field 'tool_calls' must contain strings")
            item = item.strip()
            if item:
                tool_calls.append(item)

        steps = raw_output.get("steps", 1)
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("Imported agent output field 'steps' must be a positive integer")

        return AgentOutput(response=response, tool_calls=tool_calls, steps=steps)

    raise ValueError(
        "Imported agent returned an unsupported output type. "
        "Expected AgentOutput, str, or dict."
    )


def _split_import_spec(spec: str) -> tuple[str, str]:
    module_name, separator, symbol_name = spec.partition(":")
    if not separator or not module_name.strip() or not symbol_name.strip():
        raise ValueError(
            "Custom agent spec must be in the format 'module.path:SymbolName'."
        )
    return module_name.strip(), symbol_name.strip()


def _resolve_symbol(spec: str) -> Any:
    module_name, symbol_name = _split_import_spec(spec)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Failed to import module '{module_name}': {exc}") from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' does not define symbol '{symbol_name}'") from exc


def _call_with_supported_kwargs(callable_obj: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj()

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return callable_obj(**kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(**accepted_kwargs)


def _instantiate_custom_agent(spec: str, init_kwargs: dict[str, Any]) -> Agent:
    symbol = _resolve_symbol(spec)

    if not callable(symbol):
        raise ValueError(f"Imported symbol '{spec}' is not callable")

    try:
        candidate = _call_with_supported_kwargs(symbol, init_kwargs)
    except TypeError as exc:
        raise ValueError(f"Failed to initialize custom agent '{spec}': {exc}") from exc

    if not hasattr(candidate, "run") or not callable(candidate.run):
        raise ValueError(
            f"Custom agent '{spec}' must resolve to an object that implements run(task)"
        )

    raw_name = getattr(candidate, "name", spec)
    resolved_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else spec

    return ImportedAgentAdapter(name=resolved_name, _delegate=candidate)


def get_agent(
    agent_name: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    timeout_seconds: float = 45.0,
) -> Agent:
    """Return a built-in or user-provided agent."""

    init_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
    }

    factories = _builtin_agent_factories()
    factory = factories.get(agent_name)
    if factory is not None:
        return factory(**init_kwargs)

    if ":" in agent_name:
        return _instantiate_custom_agent(agent_name, init_kwargs)

    options = ", ".join(sorted(factories))
    raise ValueError(
        f"Unknown agent '{agent_name}'. Available built-ins: {options}. "
        "For custom agents, use 'module.path:SymbolName'."
    )
