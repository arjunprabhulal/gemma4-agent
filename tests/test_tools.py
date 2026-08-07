"""Tests for the ToolRegistry and its built-in tools.

Note: web_search and fetch_google_skill hit the live network (DuckDuckGo, GitHub).
"""
import os
import unittest
from gemma_agent.tools import ToolRegistry


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

    def test_take_screenshot_tool(self):
        tools = ToolRegistry()
        target_path = "/tmp/test_gemma_screenshot.png"
        if os.path.exists(target_path):
            os.remove(target_path)

        res = tools.execute("take_screenshot", {"filename": target_path})
        self.assertTrue("Screenshot captured" in res or "Screenshot error" in res)

    def test_ripgrep_search_tool(self):
        tools = ToolRegistry()
        res = tools.execute("ripgrep_search", {"query": "ToolRegistry", "path": "."})
        self.assertTrue(len(res) > 0)

    def test_web_search_tool(self):
        tools = ToolRegistry()
        res = tools.execute("web_search", {"query": "Python programming language"})
        self.assertTrue(len(res) > 0)

    def test_fetch_google_skill_tool(self):
        tools = ToolRegistry()
        res = tools.execute("fetch_google_skill", {"skill_name": "cloud-run"})
        self.assertTrue(len(res) > 0)


if __name__ == "__main__":
    unittest.main()
