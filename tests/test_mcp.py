"""Tests for the experimental MCP server-config registry."""
import unittest
from gemma_agent.mcp import MCPManager


class TestMCPManager(unittest.TestCase):

    def test_register_and_list(self):
        mcp_mgr = MCPManager()
        msg = mcp_mgr.connect_server("test_db", "npx -y @modelcontextprotocol/server-postgres")
        self.assertIn("registered", msg)
        self.assertIn("test_db", mcp_mgr.list_servers())


if __name__ == "__main__":
    unittest.main()
