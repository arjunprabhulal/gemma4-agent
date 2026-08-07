import os
import json
from typing import Dict, Any, Optional

class MCPManager:
    """
    Experimental MCP server-configuration registry.

    Stores and lists MCP server entries in ~/.gemma/mcp_servers.json. Note: this
    does NOT yet spawn server processes or speak the MCP protocol — entries are
    configuration only, pending a full MCP client implementation.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.connected_servers: Dict[str, Dict[str, Any]] = {}
        self.config_path = config_path or os.path.expanduser("~/.gemma/mcp_servers.json")
        self.load_mcp_config()

    def load_mcp_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Validate shape: a hand-edited config in another format (e.g.
                # the standard {"mcpServers": ...}) must not break every /mcp use.
                if isinstance(data, dict):
                    self.connected_servers = {
                        str(k): v for k, v in data.items()
                        if isinstance(v, dict) and "command" in v
                    }
                else:
                    self.connected_servers = {}
            except Exception:
                self.connected_servers = {}

    def save_mcp_config(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.connected_servers, f, indent=2)
        except Exception:
            pass

    def connect_server(self, name: str, command: str) -> str:
        self.connected_servers[name.lower()] = {
            "name": name,
            "command": command,
            "status": "registered"
        }
        self.save_mcp_config()
        return f"MCP Server '{name}' registered (config saved; experimental — server is not spawned yet). Command: {command}"

    def disconnect_server(self, name: str) -> str:
        name_lower = name.lower()
        if name_lower in self.connected_servers:
            del self.connected_servers[name_lower]
            self.save_mcp_config()
            return f"MCP Server '{name}' removed from the registry."
        return f"MCP Server '{name}' not found."

    def list_servers(self) -> str:
        if not self.connected_servers:
            return "No MCP servers registered yet. Use `/mcp connect <name> <command>` to register a server config (experimental — servers are not spawned yet)."
        out = "### 🔌 Registered MCP Servers (Experimental)\n\n"
        for sname, sinfo in self.connected_servers.items():
            out += f"- **{sinfo.get('name', sname)}**: `{sinfo.get('command', '?')}` (Status: {sinfo.get('status', 'registered')})\n"
        return out
