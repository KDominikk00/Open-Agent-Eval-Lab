"""Baseline and API-backed agent implementations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol
from urllib import error, request

from open_agent_eval.schemas import TaskSpec


@dataclass(frozen=True)
class AgentOutput:
    """Structured output returned by any benchmarked agent."""

    response: str
    tool_calls: list[str]
    steps: int


class Agent(Protocol):
    """Protocol implemented by all agent backends."""

    name: str

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

    def run(self, task: TaskSpec) -> AgentOutput:
        model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = _resolve_api_key(explicit_key=self.api_key, env_name="OPENAI_API_KEY")

        payload: dict[str, Any] = {
            "model": model,
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
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds=self.timeout_seconds,
        )

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI response missing choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("OpenAI response choice is malformed")

        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI response missing message payload")

        response_text = _extract_openai_content(message.get("content"))
        tool_calls = _extract_tool_calls_openai(message)
        steps = min(task.max_steps, max(1, len(tool_calls) + 1))

        return AgentOutput(response=response_text, tool_calls=tool_calls, steps=steps)


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

    def run(self, task: TaskSpec) -> AgentOutput:
        model = self.model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        api_key = _resolve_api_key(explicit_key=self.api_key, env_name="ANTHROPIC_API_KEY")

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": _task_prompt(task)}],
        }

        response = _post_json(
            url=self.endpoint,
            payload=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.anthropic_version,
                "Content-Type": "application/json",
            },
            timeout_seconds=self.timeout_seconds,
        )

        response_text, tool_calls = _extract_anthropic_content(response.get("content"))
        if not response_text and not tool_calls:
            raise ValueError("Anthropic response missing content blocks")

        steps = min(task.max_steps, max(1, len(tool_calls) + 1))
        return AgentOutput(response=response_text, tool_calls=tool_calls, steps=steps)


def _registry() -> dict[str, Agent]:
    return {
        "echo-reference": EchoReferenceAgent(),
        "instruction-snippet": InstructionSnippetAgent(),
        "tool-hint": ToolHintAgent(),
        "openai-chat": OpenAIChatAgent(),
        "anthropic-chat": AnthropicChatAgent(),
    }


def list_available_agents() -> list[str]:
    """List all available baseline agents."""

    return sorted(_registry())


def get_agent(
    agent_name: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    timeout_seconds: float = 45.0,
) -> Agent:
    """Return a baseline agent by name."""

    if agent_name == "openai-chat":
        return OpenAIChatAgent(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )

    if agent_name == "anthropic-chat":
        return AnthropicChatAgent(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )

    registry = _registry()

    try:
        return registry[agent_name]
    except KeyError as exc:
        options = ", ".join(sorted(registry))
        raise ValueError(f"Unknown agent '{agent_name}'. Available agents: {options}") from exc
