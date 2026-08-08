"""End-to-end smoke test: drives the real REPL with piped input.

Static review and unit tests exercise code paths; this test exercises the
product — the actual CLI process, its banner, slash-command handling, and
clean exit. No Ollama model calls are made (only slash commands), so it
runs offline. It never writes to the user's real ~/.gemma config
(read-only /mcp listing only).
"""
import os
import subprocess
import sys
import unittest


class TestCLIEndToEnd(unittest.TestCase):

    def _run_repl(self, commands, timeout=60):
        env = {**os.environ, "COLUMNS": "200"}  # keep rich output unwrapped for stable assertions
        return subprocess.run(
            [sys.executable, "-m", "gemma_agent.cli"],
            input="\n".join(commands) + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def test_slash_command_session(self):
        proc = self._run_repl([
            "/help",
            "/model",                    # bare /model must show usage, not "Unknown command"
            "/model gemma4:12b",
            "/tools",
            "/mcp",
            "/definitely-not-a-command",
            "/exit",
        ])
        out = proc.stdout

        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr[-2000:]}")
        self.assertIn("GEMMA 4 AGENTIC CLI", out)          # banner rendered
        self.assertIn("/model <tag>", out)                 # /help table rendered
        self.assertIn("Current model:", out)               # bare /model regression guard
        self.assertNotIn("Unknown command '/model'", out)
        self.assertIn("Switched local Gemma model tag to 'gemma4:12b'", out)
        self.assertIn("bash_run", out)                     # /tools listing
        self.assertIn("MCP", out)                          # /mcp panel
        self.assertIn("Unknown command", out)              # bad command handled gracefully
        self.assertIn("Goodbye", out)                      # clean /exit

    def test_slash_command_vs_file_path_detection(self):
        """Absolute file paths start with '/' but are prompts, not commands."""
        from gemma_agent.cli import _is_slash_command
        # Real commands
        self.assertTrue(_is_slash_command("/help"))
        self.assertTrue(_is_slash_command("/model gemma4:12b"))
        self.assertTrue(_is_slash_command("/skills clear"))
        self.assertTrue(_is_slash_command("/definitely-a-typo"))
        # File paths — must reach the model as prompts (the exact user-reported case)
        self.assertFalse(_is_slash_command("/Users/me/Desktop/diagram.png convert this into code"))
        self.assertFalse(_is_slash_command("/tmp/x.png describe"))
        self.assertFalse(_is_slash_command("/etc"))          # bare existing path
        self.assertFalse(_is_slash_command("plain question"))

    def test_exec_mode_flag_parses(self):
        # --help must work and exit 0 without a model or network
        proc = subprocess.run(
            [sys.executable, "-m", "gemma_agent.cli", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--model", proc.stdout)
        self.assertIn("-e EXEC", proc.stdout)


if __name__ == "__main__":
    unittest.main()
