"""Tests for the voice input module."""
import inspect
import unittest
from gemma_agent import voice_input


class TestVoiceInput(unittest.TestCase):

    def test_listen_to_microphone_signature(self):
        sig = inspect.signature(voice_input.listen_to_microphone)
        self.assertIn("duration", sig.parameters)
        self.assertIn("silence_timeout", sig.parameters)


if __name__ == "__main__":
    unittest.main()
