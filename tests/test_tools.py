"""Tests for the ToolRegistry and its built-in tools.

Note: web_search and fetch_google_skill hit the live network (DuckDuckGo,
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
        self.assertEqual(len(schemas), 10)
        tool_names = [t["function"]["name"] for t in schemas]

        expected_tools = [
            "bash_run", "read_file", "write_file", "list_directory",
            "python_eval", "web_fetch", "web_search", "fetch_google_skill",
            "take_screenshot", "ripgrep_search"
        ]
        for t in expected_tools:
            self.assertIn(t, tool_names)

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

    def test_fetch_google_skill_tool(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SkillManager(cache_dir=os.path.join(td, "skills"))
            tools = ToolRegistry(skill_manager=sm)
            res = tools.execute("fetch_google_skill", {"skill_name": "gke-basics"})
            if "Could not query" in res or "Error fetching" in res:
                self.skipTest(f"GitHub unavailable: {res[:80]}")
            # The returned skill must actually be the requested one
            self.assertIn("gke-basics", res)


if __name__ == "__main__":
    unittest.main()
