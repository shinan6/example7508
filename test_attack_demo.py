import unittest

from attack_demo import FAKE_SECRET, run_demo


class AttackDemoTests(unittest.TestCase):
    def test_clean_note_does_not_trigger_send(self):
        result = run_demo(injected=False)
        self.assertEqual(result["sent"], [])
        self.assertFalse(any("model -> send_data" in event for event in result["trace"]))

    def test_injected_note_triggers_simulated_send(self):
        result = run_demo()
        self.assertEqual(result["sent"], [FAKE_SECRET])
        self.assertTrue(any("model -> send_data" in event for event in result["trace"]))

    def test_unregistered_sink_blocks_attempted_send(self):
        result = run_demo(allow_sink=False)
        self.assertEqual(result["sent"], [])
        self.assertTrue(any("model -> send_data" in event for event in result["trace"]))
        self.assertTrue(any("Unknown tool: send_data" in event for event in result["trace"]))
        self.assertIn("blocked", result["answer"])


if __name__ == "__main__":
    unittest.main()
