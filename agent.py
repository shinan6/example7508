"""Small synchronous tool-calling agent, using only the Python standard library."""

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Agent:
    """A model callable receives (messages, tool_schemas) and returns a message."""

    def __init__(self, model, tools=(), max_turns=8):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.model = model
        self.tools = {}
        self.max_turns = max_turns
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"Duplicate tool: {tool.name}")
            self.tools[tool.name] = tool

    def run(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        schemas = [tool.schema() for tool in self.tools.values()]
        for turn in range(self.max_turns):
            message = self.model(messages, schemas)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                text = message.get("content") or message.get("refusal")
                if not isinstance(text, str) or not text:
                    raise RuntimeError("Model returned no text or tool calls")
                return text
            if turn == self.max_turns - 1:
                break
            for call in calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": self._execute(call),
                })
        raise RuntimeError(f"Agent exceeded {self.max_turns} model turns")

    def _execute(self, call):
        try:
            function = call["function"]
            name = function["name"]
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            result = self.tools[name].handler(**arguments)
            return json.dumps({"result": result}, allow_nan=False)
        except Exception as error:
            # Let the model correct bad arguments or respond to a tool failure.
            return json.dumps({"error": f"{type(error).__name__}: {error}"})


@dataclass
class OpenAIModel:
    """Minimal non-streaming Chat Completions adapter."""

    model: str
    api_key: str

    def __post_init__(self):
        if not self.model or not self.api_key:
            raise ValueError("A model name and API key are required")

    def __call__(self, messages, tools):
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                completion = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Model API returned HTTP {error.code}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError("Model API connection failed or timed out") from error
        choice = completion["choices"][0]
        if choice["finish_reason"] not in ("stop", "tool_calls"):
            raise RuntimeError(f"Model stopped early: {choice['finish_reason']}")
        return choice["message"]
