"""Tests for the local Ollama backend."""
import json
import os
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

    def test_native_thinking_wrapped_into_think_tags(self):
        """Ollama's native `thinking` field is normalized into <think> tags so
        the existing extraction pipeline handles it uniformly."""
        backend = LocalGemmaBackend(model_name="stub")
        resp = Mock(status_code=200)
        resp.json = lambda: {"message": {"content": "Four", "thinking": "2 plus 2 is 4"},
                             "prompt_eval_count": 3, "eval_count": 2}
        with patch("gemma_agent.backends.requests.post", return_value=resp) as post:
            content, _, _ = backend.generate_response(
                [{"role": "user", "content": "2+2?"}], think=True)
        self.assertEqual(content, "<think>2 plus 2 is 4</think>Four")
        self.assertTrue(post.call_args.kwargs["json"]["think"])

    def test_check_connection_paths(self):
        backend = LocalGemmaBackend(model_name="stub")
        ok_resp = Mock(status_code=200)
        ok_resp.json = lambda: {"models": [{"name": "gemma4:26b"}, {"model": "noname"}]}
        with patch("gemma_agent.backends.requests.get", return_value=ok_resp):
            ok, msg = backend.check_connection()
        self.assertTrue(ok)
        self.assertIn("gemma4:26b", msg)

        bad_resp = Mock(status_code=500)
        with patch("gemma_agent.backends.requests.get", return_value=bad_resp):
            ok, msg = backend.check_connection()
        self.assertFalse(ok)

        with patch("gemma_agent.backends.requests.get", side_effect=OSError("refused")):
            ok, msg = backend.check_connection()
        self.assertFalse(ok)
        self.assertIn("refused", msg)

        # 200 with malformed body: connected, honest note — not a fake refusal
        weird = Mock(status_code=200)
        weird.json = Mock(side_effect=ValueError("bad json"))
        with patch("gemma_agent.backends.requests.get", return_value=weird):
            ok, msg = backend.check_connection()
        self.assertTrue(ok)
        self.assertIn("unexpected", msg)

    def test_generate_response_error_paths(self):
        backend = LocalGemmaBackend(model_name="stub")
        err_resp = Mock(status_code=500, text="boom")
        with patch("gemma_agent.backends.requests.post", return_value=err_resp):
            content, tc, metrics = backend.generate_response([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 500", content)
        self.assertIsNone(tc)

        with patch("gemma_agent.backends.requests.post", side_effect=OSError("down")):
            content, tc, _ = backend.generate_response([{"role": "user", "content": "hi"}])
        self.assertIn("Error communicating", content)

        # Streaming HTTP error surface
        err_resp2 = Mock(status_code=502, text="bad gateway")
        with patch("gemma_agent.backends.requests.post", return_value=err_resp2):
            content, _, _ = backend.generate_response(
                [{"role": "user", "content": "hi"}], on_token=lambda d: None)
        self.assertIn("HTTP 502", content)

    def test_image_paths_with_spaces_detected_when_quoted(self):
        import tempfile
        from gemma_agent.backends import _extract_image_paths
        d = tempfile.mkdtemp()
        path = os.path.join(d, "my photo file.png")
        with open(path, "wb") as f:
            f.write(b"png")
        try:
            text, found = _extract_image_paths(f'describe "{path}" please')
            self.assertEqual(found, [path])
            self.assertEqual(text, f'describe "{path}" please')  # text untouched
            # Unquoted spaced path still undetectable (documented); no crash
            _, found2 = _extract_image_paths(f'describe {path} please')
            self.assertNotIn(path, found2)
        finally:
            os.remove(path)

    def test_streaming_thinking_deltas(self):
        backend = LocalGemmaBackend(model_name="stub")
        lines = [
            json.dumps({"message": {"thinking": "hmm "}}).encode(),
            json.dumps({"message": {"thinking": "ok"}}).encode(),
            json.dumps({"message": {"content": "Done"}}).encode(),
            json.dumps({"done": True, "prompt_eval_count": 1, "eval_count": 1}).encode(),
        ]
        resp = Mock(status_code=200)
        resp.iter_lines = lambda: iter(lines)
        deltas = []
        with patch("gemma_agent.backends.requests.post", return_value=resp):
            content, _, _ = backend.generate_response(
                [{"role": "user", "content": "hi"}], on_token=deltas.append, think=True)
        self.assertEqual(content, "<think>hmm ok</think>Done")
        self.assertEqual(deltas, ["hmm ", "ok", "Done"])

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
