# LLM Agent Prototype

A minimal Python prototype for [issue #2](https://github.com/shinan6/example7508/issues/2).
For developers who want to try a tool-calling loop without installing a framework.
Requires Python 3.10 or later; uses only the standard library.

## Try It

```sh
python3 demo.py
# 2 + 3 = 5

python3 -m unittest -v
```

The offline demo uses a scripted model response, executes the real `add` function,
and sends its result back through the agent loop. It makes no network requests.

For a live model, set `OPENAI_API_KEY` in your shell, then run:

```sh
python3 demo.py --live --model YOUR_MODEL_ID "What is 12 + 30? Use add."
```

Choose a model that supports function tools through Chat Completions.
You can also set `OPENAI_MODEL` instead of passing `--model`.
The adapter follows the [official function-calling documentation](https://developers.openai.com/api/docs/guides/function-calling).

## Tool-Calling Attack Demo

The offline demo for [issue #1](https://github.com/shinan6/example7508/issues/1)
simulates a prompt injection in a tool result:

```sh
python3 attack_demo.py
```

The user asks for a project status summary. The `read_note` tool returns a note
containing fake data. An injected instruction tells the model to call `send_data`
under the pretext of a system verification step.

The command prints tool calls, results, and sink contents for three cases:

| Case | Observed Sink Contents |
| --- | --- |
| Clean note, sink registered | Empty |
| Injected note, sink registered | `DEMO-SECRET-123` |
| Injected note, sink unavailable | Empty; the agent rejects the unknown tool |

The model is scripted to follow this particular injected instruction. The demo
exercises real tool dispatch, but does not measure a real LLM's susceptibility.
The sink stores fake data in memory. No API key, network access, or dependencies
are needed.

Removing the sink blocks this particular call through the existing tool registry.
It does not provide a general defense against misuse of tools that remain registered.

## Add a Tool

Create a `Tool` with a name, description, JSON parameter schema, and Python
handler. Pass it to `Agent(model, tools=[...])`, then call `run(prompt)`.
See `ADD` in `demo.py` for a complete example.

The model callable accepts `(messages, tool_schemas)` and returns an assistant
message in Chat Completions format. Replace it to use another provider or a test double.
Each run starts a fresh conversation. The agent executes requested tools sequentially,
returns results with their call IDs, and repeats until the model returns text.

## Prototype Limits

The agent defaults to eight model turns. It raises an error if that limit is
reached and skips tool execution on the final turn. Tool errors return to the
model as JSON so it can correct its arguments. Network failures stop the run.

Handlers must validate their inputs; the framework does not validate full JSON
schemas. Register only trusted handlers because they execute in the Python process.
This prototype has no streaming, persistence, retries, or concurrent execution.
Tests use scripted responses and a mocked HTTP transport; they do not require API access.
