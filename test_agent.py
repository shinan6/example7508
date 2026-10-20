import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from agent import Agent, OpenAIModel
from demo import ADD, offline_model


def tool_call(name="add", arguments='{"a": 2, "b": 3}', call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class AgentTests(unittest.TestCase):
    def test_offline_demo(self):
        self.assertEqual(Agent(offline_model, [ADD]).run("Add 2 and 3"), "2 + 3 = 5")

    def test_multiple_calls_preserve_ids_and_results(self):
        calls = [tool_call(), tool_call(arguments='{"a": 8, "b": 1}', call_id="call_2")]

        def model(messages, tools):
            self.assertEqual(tools[0]["function"]["name"], "add")
            if len(messages) == 1:
                return {"role": "assistant", "tool_calls": calls}
            self.assertEqual(messages[1]["tool_calls"], calls)
            self.assertEqual([m["tool_call_id"] for m in messages[2:]], ["call_1", "call_2"])
            self.assertEqual(
                [json.loads(m["content"]) for m in messages[2:]],
                [{"result": 5}, {"result": 9}],
            )
            return {"role": "assistant", "content": "5 and 9"}

        self.assertEqual(Agent(model, [ADD]).run("Add twice"), "5 and 9")

    def test_tool_errors_are_returned_to_model(self):
        for call in (
            tool_call(name="missing"), tool_call(arguments="{"),
            tool_call(arguments="[]"), tool_call(arguments='{"a": 2}'),
            tool_call(arguments='{"a": "two", "b": 3}'),
        ):
            with self.subTest(call=call):
                def model(messages, tools):
                    if len(messages) == 1:
                        return {"role": "assistant", "tool_calls": [call]}
                    self.assertIn("error", json.loads(messages[-1]["content"]))
                    return {"role": "assistant", "content": "Handled failure"}
                self.assertEqual(Agent(model, [ADD]).run("Try"), "Handled failure")

    def test_turn_limit_stops_before_executing_last_turn(self):
        model = Mock(return_value={"role": "assistant", "tool_calls": [tool_call()]})
        with patch.object(ADD, "handler", return_value=5) as handler:
            with self.assertRaisesRegex(RuntimeError, "exceeded 2"):
                Agent(model, [ADD], max_turns=2).run("Keep going")
            self.assertEqual(model.call_count, 2)
            handler.assert_called_once_with(a=2, b=3)

    def test_plain_answer_and_empty_answer(self):
        self.assertEqual(Agent(lambda *_: {"content": "Hello"}).run("Hi"), "Hello")
        with self.assertRaisesRegex(RuntimeError, "no text"):
            Agent(lambda *_: {"content": None}).run("Hi")

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            Agent(offline_model, [ADD, ADD])
        with self.assertRaises(ValueError):
            Agent(offline_model, max_turns=0)
        with self.assertRaises(ValueError):
            OpenAIModel("test-model", "")

    @patch("agent.urlopen")
    def test_http_adapter(self, urlopen):
        message = {"role": "assistant", "content": "5"}
        response = {"choices": [{"message": message, "finish_reason": "stop"}]}
        urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps(response).encode())
        model = OpenAIModel("test-model", "test-key")
        messages = [{"role": "user", "content": "Add"}]
        self.assertEqual(model(messages, [ADD.schema()]), message)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {
            "model": "test-model", "messages": messages, "tools": [ADD.schema()],
        })
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 30)

    @patch("agent.urlopen")
    def test_http_error(self, urlopen):
        urlopen.side_effect = HTTPError("https://api.openai.com", 401, "Unauthorized", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            OpenAIModel("test-model", "test-key")([], [])


if __name__ == "__main__":
    unittest.main()
