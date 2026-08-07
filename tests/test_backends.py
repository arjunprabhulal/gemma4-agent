"""Tests for the local Ollama backend."""
import unittest
from gemma_agent.backends import LocalGemmaBackend


class TestLocalGemmaBackend(unittest.TestCase):

    def test_backend_initialization(self):
        local_b = LocalGemmaBackend(model_name="gemma4:26b")
        self.assertEqual(local_b.model_name, "gemma4:26b")


if __name__ == "__main__":
    unittest.main()
