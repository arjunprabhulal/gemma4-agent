"""Tests for the experimental MCP server-config registry.

Uses a temporary config path so tests never touch the user's real
~/.gemma/mcp_servers.json.
"""
import os
import tempfile
import unittest
from gemma_agent.mcp import MCPManager


class TestMCPManager(unittest.TestCase):

    def test_register_list_and_remove(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "mcp_servers.json")
            mgr = MCPManager(config_path=cfg)

            msg = mgr.connect_server("test_db", "npx -y @modelcontextprotocol/server-postgres")
            self.assertIn("registered", msg)
            self.assertIn("test_db", mgr.list_servers())

            # Config persists to the isolated path and reloads
            reloaded = MCPManager(config_path=cfg)
            self.assertIn("test_db", reloaded.list_servers())

            self.assertIn("removed", reloaded.disconnect_server("test_db"))
            self.assertNotIn("test_db", reloaded.list_servers())
            self.assertIn("not found", reloaded.disconnect_server("test_db"))


if __name__ == "__main__":
    unittest.main()
