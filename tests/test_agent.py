"""Tests for the GemmaAgent orchestrator loop."""
import unittest
from gemma_agent.backends import LocalGemmaBackend
from gemma_agent.agent import GemmaAgent
from gemma_agent.tools import ToolRegistry


class TestGemmaAgent(unittest.TestCase):

    def test_agent_run_turn(self):
        class MockBackend(LocalGemmaBackend):
            def generate_response(self, messages, tools_schema=None):
                return "Mocked Gemma response.", None, {"duration_sec": 0.1, "backend_label": "Mock Backend"}

        backend = MockBackend(model_name="gemma4:26b")
        agent = GemmaAgent(backend=backend, tool_registry=ToolRegistry())
        self.assertTrue(hasattr(agent, "backend"))
        self.assertEqual(agent.backend.model_name, "gemma4:26b")

        response = agent.run_turn("Hello Gemma!")
        self.assertEqual(response, "Mocked Gemma response.")
        self.assertTrue(len(agent.history) >= 2)

    def test_tool_execution_loop_with_dedup(self):
        """A tool call executes once; an identical repeat is deduplicated; dedup notes never leak."""
        class ScriptedBackend(LocalGemmaBackend):
            def __init__(self):
                super().__init__(model_name="stub")
                self.step = 0

            def generate_response(self, messages, tools_schema=None):
                self.step += 1
                if self.step <= 2:  # same call twice — second must be deduped
                    return "", [{"name": "python_eval", "arguments": {"code": "print('hi')"}}], {"duration_sec": 0}
                return "All done.", None, {"duration_sec": 0}

        agent = GemmaAgent(backend=ScriptedBackend(), tool_registry=ToolRegistry())
        response = agent.run_turn("run it")

        self.assertEqual(response, "All done.")
        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("hi", tool_msgs[0]["content"])
        leaked_notes = [m for m in agent.history
                        if m.get("role") == "system" and str(m.get("content", "")).startswith("System Note:")]
        self.assertEqual(leaked_notes, [])

    def test_failed_tool_call_can_be_retried(self):
        """Self-healing: an errored call is exempt from dedup so the model can re-run it."""
        class RetryBackend(LocalGemmaBackend):
            def __init__(self):
                super().__init__(model_name="stub")
                self.step = 0

            def generate_response(self, messages, tools_schema=None):
                self.step += 1
                if self.step <= 2:  # identical failing call requested twice
                    return "", [{"name": "python_eval", "arguments": {"code": "import sys; sys.exit(1)"}}], {"duration_sec": 0}
                return "Gave up gracefully.", None, {"duration_sec": 0}

        agent = GemmaAgent(backend=RetryBackend(), tool_registry=ToolRegistry())
        agent.run_turn("run failing thing")

        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 2)

    def test_forced_grounding_for_post_cutoff_topics(self):
        """Post-cutoff topics trigger an automatic web_search whose results are
        injected before the model call; results never persist; /ground off disables."""
        from unittest.mock import patch
        seen_messages = []

        class CaptureBackend(LocalGemmaBackend):
            def generate_response(self, messages, tools_schema=None):
                seen_messages.append([dict(m) for m in messages])
                return "ok", None, {"duration_sec": 0}

        agent = GemmaAgent(backend=CaptureBackend(model_name="stub"), tool_registry=ToolRegistry())

        with patch.object(agent.tools, "execute", return_value="Result 1: ADK is real") as pexec:
            agent.run_turn("write a google ADK agent using the Maps MCP server")
        pexec.assert_called_once()
        self.assertEqual(pexec.call_args[0][0], "web_search")
        notes = [m for m in seen_messages[-1] if str(m.get("content", "")).startswith("System Note:")]
        self.assertEqual(len(notes), 1)
        self.assertIn("Result 1: ADK is real", notes[0]["content"])
        # Injected results stripped from history after the turn
        leaked = [m for m in agent.history if str(m.get("content", "")).startswith("System Note:")]
        self.assertEqual(leaked, [])

        # Non-trigger topics: no search, no note
        with patch.object(agent.tools, "execute") as pexec:
            agent.run_turn("what is a python list comprehension")
        pexec.assert_not_called()

        # /ground off disables entirely
        agent.grounding_enabled = False
        with patch.object(agent.tools, "execute") as pexec:
            agent.run_turn("build another adk thing")
        pexec.assert_not_called()

    def test_thinking_toggle_rebuilds_system_prompt_in_place(self):
        agent = GemmaAgent(backend=LocalGemmaBackend(model_name="stub"), tool_registry=ToolRegistry())
        self.assertTrue(agent.thinking_enabled)
        self.assertIn("THINKING & REASONING MODE", agent.history[0]["content"])
        self.assertIn("January 2025", agent.history[0]["content"])  # knowledge cutoff stated

        agent.history.append({"role": "user", "content": "hi"})
        agent.set_thinking(False)
        self.assertIn("THINKING MODE IS OFF", agent.history[0]["content"])
        self.assertNotIn("THINKING & REASONING MODE", agent.history[0]["content"])
        self.assertEqual(len(agent.history), 2)  # conversation preserved

        agent.set_thinking(True)
        self.assertIn("THINKING & REASONING MODE", agent.history[0]["content"])

    def test_string_tool_arguments_are_coerced(self):
        """Arguments arriving as a JSON string (OpenAI wire format) must not crash the loop."""
        class StrArgsBackend(LocalGemmaBackend):
            def __init__(self):
                super().__init__(model_name="stub")
                self.step = 0

            def generate_response(self, messages, tools_schema=None):
                self.step += 1
                if self.step == 1:
                    return "", [{"name": "python_eval", "arguments": "{\"code\": \"print('coerced-ok')\"}"}], {"duration_sec": 0}
                return "Done.", None, {"duration_sec": 0}

        agent = GemmaAgent(backend=StrArgsBackend(), tool_registry=ToolRegistry())
        resp = agent.run_turn("go")
        self.assertEqual(resp, "Done.")
        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("coerced-ok", tool_msgs[0]["content"])

    def test_iteration_cap_reached(self):
        class EndlessBackend(LocalGemmaBackend):
            def __init__(self):
                super().__init__(model_name="stub")
                self.step = 0

            def generate_response(self, messages, tools_schema=None):
                self.step += 1
                return "", [{"name": "python_eval", "arguments": {"code": f"print({self.step})"}}], {"duration_sec": 0}

        agent = GemmaAgent(backend=EndlessBackend(), tool_registry=ToolRegistry())
        response = agent.run_turn("loop forever", max_iterations=3)
        self.assertIn("maximum tool execution limit (3 iterations)", response)

    def test_parse_json_tool_calls(self):
        agent = GemmaAgent(backend=LocalGemmaBackend(model_name="stub"), tool_registry=ToolRegistry())

        fenced = 'Let me check.\n```json\n{"tool": "read_file", "arguments": {"filepath": "x.txt"}}\n```'
        self.assertEqual(
            agent._parse_json_tool_calls(fenced),
            [{"name": "read_file", "arguments": {"filepath": "x.txt"}}],
        )
        bare = '{"tool": "list_directory", "arguments": {}}'
        self.assertEqual(
            agent._parse_json_tool_calls(bare),
            [{"name": "list_directory", "arguments": {}}],
        )
        self.assertIsNone(agent._parse_json_tool_calls("no tools mentioned here"))
        self.assertIsNone(agent._parse_json_tool_calls('```json\n{"not_a_tool": 1}\n```'))
        self.assertIsNone(agent._parse_json_tool_calls(""))

    def test_extract_thinking_text(self):
        agent = GemmaAgent(backend=LocalGemmaBackend(model_name="stub"), tool_registry=ToolRegistry())
        think, content = agent._extract_thinking_text("<think>Reasoning step 1</think>Final answer.")
        self.assertEqual(think, "Reasoning step 1")
        self.assertEqual(content, "Final answer.")
        think, content = agent._extract_thinking_text("Plain answer.")
        self.assertIsNone(think)
        self.assertEqual(content, "Plain answer.")


if __name__ == "__main__":
    unittest.main()
