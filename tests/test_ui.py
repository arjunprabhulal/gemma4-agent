"""Smoke tests for every UI rendering function — hostile inputs must never
raise (markup injection was an empirically confirmed crash class)."""
import unittest
from gemma_agent import ui

HOSTILE = "[/bad]closing [red]open ``` `code` [link=x]y[/link] <script>alert(1)</script> \\[esc]"


class TestUIRendering(unittest.TestCase):

    def test_all_print_functions_survive_hostile_input(self):
        ui.print_banner("BACKEND [/x]", "model[/foo]tag")
        ui.print_user_prompt(HOSTILE)
        ui.print_agent_thought(HOSTILE)
        ui.print_thinking_panel(HOSTILE)          # HTML-ish → Text fallback path
        ui.print_thinking_panel("plain *markdown* thinking")
        ui.print_tool_call("tool[/x]", {"cmd[/y]": HOSTILE, "n": 42, "long": "x" * 100})
        ui.print_tool_call("write_file", {"filepath": "a[/b].py", "content": "line\n" * 20})
        ui.print_tool_result(HOSTILE, is_error=True)
        ui.print_tool_result("ok " * 500, is_error=False)   # truncation path
        ui.print_markdown(HOSTILE, title="T[/t]", speak=False)
        ui.print_info(HOSTILE)
        ui.print_success(HOSTILE)
        ui.print_error(HOSTILE)
        ui.print_turn_metrics(1.23, 100, 50, backend_label="Local ([/foo]x)")
        ui.print_turn_metrics(0.5)                # zero-token branch
        ui.print_help()

    def test_streaming_live_updates(self):
        with ui.streaming_live() as update:
            update("partial")
            update("partial with [/hostile] markup " + "x" * 3000)  # tail bound

    def test_speak_text_disabled_is_noop(self):
        prev = ui.voice_enabled
        ui.voice_enabled = False
        try:
            ui.speak_text("should not spawn say")  # returns before subprocess
        finally:
            ui.voice_enabled = prev

    def test_spoken_replies_muted_by_default(self):
        """TTS is opt-in: a fresh session never speaks until /voice unmute."""
        self.assertTrue(ui.speech_muted)

    def test_speak_text_muted_never_spawns_say(self):
        """/voice mute: listening stays on, but no `say` process may start."""
        from unittest.mock import patch
        prev_v, prev_m = ui.voice_enabled, ui.speech_muted
        ui.voice_enabled, ui.speech_muted = True, True
        try:
            with patch("gemma_agent.ui.subprocess.Popen") as popen:
                ui.speak_text("silent please")
            popen.assert_not_called()
        finally:
            ui.voice_enabled, ui.speech_muted = prev_v, prev_m
