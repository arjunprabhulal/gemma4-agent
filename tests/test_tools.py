"""Tests for the ToolRegistry and its built-in tools.

Note: web_search and fetch_skill hit the live network (DuckDuckGo,
GitHub) — they skip when the network is unavailable rather than passing
vacuously. Skill fetching uses a temp cache dir so tests never write to the
user's real ~/.gemma.
"""
import os
import tempfile
import unittest
from gemma_agent.tools import ToolRegistry
from gemma_agent.skills import SkillManager


class TestToolRegistry(unittest.TestCase):

    def test_tool_registry_all_10_tools(self):
        tools = ToolRegistry()
        schemas = tools.get_schemas()
        self.assertEqual(len(schemas), 11)
        tool_names = [t["function"]["name"] for t in schemas]

        expected_tools = [
            "bash_run", "read_file", "write_file", "list_directory",
            "python_eval", "web_fetch", "web_search", "fetch_skill",
            "take_screenshot", "ripgrep_search", "analyze_audio"
        ]
        for t in expected_tools:
            self.assertIn(t, tool_names)

    def test_analyze_audio_payload_and_errors(self):
        """Native-audio tool: correct OpenAI-format payload, clean error paths."""
        import struct
        import tempfile
        import wave as wave_mod
        from unittest.mock import Mock, patch
        tools = ToolRegistry()

        self.assertIn("does not exist", tools.execute("analyze_audio", {"filepath": "/nope/missing.wav"}))
        self.assertIn("supports .wav and .mp3", tools.execute("analyze_audio", {"filepath": "/tmp/notes.txt"}))

        path = tempfile.mktemp(suffix=".wav")
        with wave_mod.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<h", 0) * 1600)
        try:
            resp = Mock(status_code=200)
            resp.json = lambda: {"choices": [{"message": {"content": "a quiet tone"}}]}
            with patch("gemma_agent.tools.requests.post", return_value=resp) as post:
                out = tools.execute("analyze_audio", {"filepath": path, "question": "what is it?"})
            self.assertEqual(out, "a quiet tone")
            # The verified working endpoint and payload shape
            self.assertIn("/v1/chat/completions", post.call_args[0][0])
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "gemma4:12b")
            audio_part = payload["messages"][0]["content"][0]
            self.assertEqual(audio_part["type"], "input_audio")
            self.assertEqual(audio_part["input_audio"]["format"], "wav")
            self.assertGreater(len(audio_part["input_audio"]["data"]), 100)

            # Missing model → actionable hint
            resp404 = Mock(status_code=404, text="model not found")
            with patch("gemma_agent.tools.requests.post", return_value=resp404):
                out = tools.execute("analyze_audio", {"filepath": path})
            self.assertIn("ollama pull", out)
        finally:
            os.remove(path)

    def test_confirmation_gate_blocks_and_allows(self):
        """Denied approvals cancel system tools with a model-readable message;
        approvals run; read-only tools never prompt."""
        calls = []
        tools = ToolRegistry()
        tools.confirm_callback = lambda name, args: calls.append(name) or False
        res = tools.execute("bash_run", {"command": "echo hi"})
        self.assertIn("Cancelled by user", res)
        self.assertEqual(calls, ["bash_run"])

        tools.confirm_callback = lambda name, args: True
        res = tools.execute("bash_run", {"command": "echo approved"})
        self.assertIn("approved", res)

        # Read-only tools bypass the gate entirely
        tools.confirm_callback = lambda name, args: (_ for _ in ()).throw(AssertionError("must not prompt"))
        res = tools.execute("list_directory", {"directory_path": "."})
        self.assertNotIn("Cancelled", res)

    def test_bash_output_is_truncated(self):
        """A single command cannot blow out the model context."""
        tools = ToolRegistry()
        res = tools.execute("bash_run", {"command": "python3 -c \"print('x' * 50000)\""})
        self.assertLess(len(res), 12000)
        self.assertIn("truncated", res)

    def test_read_file_is_truncated(self):
        import tempfile
        tools = ToolRegistry()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("y" * 50000)
            path = f.name
        try:
            res = tools.execute("read_file", {"filepath": path})
            self.assertLess(len(res), 15000)
            self.assertIn("truncated", res)
        finally:
            os.remove(path)

    def test_ripgrep_fallback_output_is_capped(self):
        """The Python fallback (used when rg isn't a PATH binary) must obey the
        same cap as the rg path — an attacker demonstrated a 3MB return."""
        import tempfile
        from unittest.mock import patch
        tools = ToolRegistry()
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "huge.txt"), "w") as f:
            f.write("needle " + "x" * 3_000_000 + "\n")          # one 3MB line
            for _ in range(30):
                f.write("needle " + "y" * 5000 + "\n")            # many long lines
        with patch("gemma_agent.tools.subprocess.run", side_effect=FileNotFoundError):
            res = tools.execute("ripgrep_search", {"query": "needle", "path": d})
        self.assertLess(len(res), 4000)                            # bounded (was 3MB)
        for line in res.splitlines():
            self.assertLess(len(line), 350)                        # per-line cap
        self.assertIn("truncated", res)                            # total cap notice

    def test_list_directory_output_size_capped(self):
        """300 long-named entries must not exceed the byte cap."""
        import tempfile
        tools = ToolRegistry()
        d = tempfile.mkdtemp()
        for i in range(300):
            open(os.path.join(d, f"{i:03d}_" + "n" * 200 + ".txt"), "w").close()
        res = tools.execute("list_directory", {"directory_path": d})
        self.assertLess(len(res), 9000)
        self.assertIn("truncated", res)

    def test_execute_coerces_string_arguments(self):
        """Models often emit arguments as a JSON string; it must still run."""
        tools = ToolRegistry()
        res = tools.execute("list_directory", '{"directory_path": "."}')
        self.assertNotIn("Tool Execution Error", res)
        self.assertNotIn("must be a JSON object", res)

        res = tools.execute("list_directory", "not json at all")
        self.assertIn("must be a JSON object", res)

    def test_web_fetch_blocks_private_addresses(self):
        tools = ToolRegistry()
        res = tools.execute("web_fetch", {"url": "http://localhost:11434/api/tags"})
        self.assertIn("URL blocked", res)
        res = tools.execute("web_fetch", {"url": "file:///etc/passwd"})
        self.assertIn("URL blocked", res)

    def test_take_screenshot_tool(self):
        tools = ToolRegistry()
        target_path = "/tmp/test_gemma_screenshot.png"
        if os.path.exists(target_path):
            os.remove(target_path)
        try:
            res = tools.execute("take_screenshot", {"filename": target_path})
            if "Screenshot captured" in res:
                self.assertTrue(os.path.getsize(target_path) > 0)
            else:
                self.assertIn("Screenshot error", res)  # headless/no-permission envs
        finally:
            if os.path.exists(target_path):
                os.remove(target_path)

    def test_ripgrep_search_tool(self):
        tools = ToolRegistry()
        res = tools.execute("ripgrep_search", {"query": "ToolRegistry", "path": "."})
        self.assertIn("ToolRegistry", res)

    def test_web_search_tool(self):
        tools = ToolRegistry()
        res = tools.execute("web_search", {"query": "Python programming language"})
        if "Result 1:" not in res:
            self.skipTest(f"web search unavailable: {res[:80]}")
        self.assertTrue(len(res) > 20)

    def test_fetch_skill_tool(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SkillManager(cache_dir=os.path.join(td, "skills"))
            tools = ToolRegistry(skill_manager=sm)
            res = tools.execute("fetch_skill", {"skill_name": "gke-basics"})
            if "Could not query" in res or "Error fetching" in res:
                self.skipTest(f"GitHub unavailable: {res[:80]}")
            # The returned skill must actually be the requested one
            self.assertIn("gke-basics", res)


if __name__ == "__main__":
    unittest.main()
