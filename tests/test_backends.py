"""Tests for the local Ollama backend."""
import json
import unittest
from unittest.mock import Mock, patch
from gemma_agent.backends import LocalGemmaBackend


class TestLocalGemmaBackend(unittest.TestCase):

    def test_backend_initialization(self):
        local_b = LocalGemmaBackend(model_name="gemma4:26b")
        self.assertEqual(local_b.model_name, "gemma4:26b")

    def test_streaming_generate_response(self):
        """Streaming path: NDJSON chunks assemble content, tool calls, metrics."""
        backend = LocalGemmaBackend(model_name="stub")
        lines = [
            json.dumps({"message": {"content": "Hel"}}).encode(),
            json.dumps({"message": {"content": "lo"}}).encode(),
            json.dumps({"message": {"content": "", "tool_calls": [
                {"function": {"name": "bash_run", "arguments": {"command": "ls"}}}]}}).encode(),
            json.dumps({"done": True, "prompt_eval_count": 10, "eval_count": 5,
                        "total_duration": 2_000_000_000}).encode(),
        ]
        resp = Mock(status_code=200)
        resp.iter_lines = lambda: iter(lines)
        deltas = []
        with patch("gemma_agent.backends.requests.post", return_value=resp) as post:
            content, tool_calls, metrics = backend.generate_response(
                [{"role": "user", "content": "hi"}], on_token=deltas.append)

        self.assertEqual(content, "Hello")
        self.assertEqual(deltas, ["Hel", "lo"])
        self.assertEqual(tool_calls, [{"name": "bash_run", "arguments": {"command": "ls"}}])
        self.assertEqual(metrics["prompt_tokens"], 10)
        self.assertEqual(metrics["completion_tokens"], 5)
        self.assertEqual(metrics["duration_sec"], 2.0)
        self.assertTrue(post.call_args.kwargs["json"]["stream"])

    def test_non_streaming_path_unchanged(self):
        backend = LocalGemmaBackend(model_name="stub")
        resp = Mock(status_code=200)
        resp.json = lambda: {"message": {"content": "plain"}, "prompt_eval_count": 3, "eval_count": 2}
        with patch("gemma_agent.backends.requests.post", return_value=resp) as post:
            content, tool_calls, metrics = backend.generate_response(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(content, "plain")
        self.assertIsNone(tool_calls)
        self.assertFalse(post.call_args.kwargs["json"]["stream"])


if __name__ == "__main__":
    unittest.main()
