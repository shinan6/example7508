"""Run a scripted offline demo, or use a live model with --live."""

import argparse
import json
import os

from agent import Agent, OpenAIModel, Tool


def add(a, b):
    if type(a) not in (int, float) or type(b) not in (int, float):
        raise ValueError("a and b must be numbers")
    return a + b


ADD = Tool(
    name="add",
    description="Add two numbers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    handler=add,
)


def offline_model(messages, tools):
    """Scripted model stand-in; demonstrates only the fixed 2 + 3 example."""
    if messages[-1]["role"] == "tool":
        result = json.loads(messages[-1]["content"])["result"]
        return {"role": "assistant", "content": f"2 + 3 = {result}"}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "demo_add",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
        }],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="What is 2 + 3? Use add.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    args = parser.parse_args()
    if not args.live and args.prompt != parser.get_default("prompt"):
        parser.error("Custom prompts require --live; the offline demo is scripted")
    if args.live and (not args.model or not os.environ.get("OPENAI_API_KEY")):
        parser.error("--live requires OPENAI_API_KEY and --model or OPENAI_MODEL")
    model = (
        OpenAIModel(args.model, os.environ["OPENAI_API_KEY"])
        if args.live else offline_model
    )
    try:
        print(Agent(model, [ADD]).run(args.prompt))
    except RuntimeError as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
