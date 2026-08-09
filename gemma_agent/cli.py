#!/usr/bin/env python3
import argparse
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from gemma_agent.backends import LocalGemmaBackend
from gemma_agent.agent import GemmaAgent
from gemma_agent.tools import ToolRegistry
from gemma_agent import ui


# Single source of truth for every REPL command and alias.
KNOWN_COMMANDS = frozenset({
    "/exit", "/quit",
    "/help", "/clear", "/history",
    "/voice", "/mic", "/talk",
    "/stop", "/pause",
    "/model", "/backend",
    "/skills", "/mcp", "/tools",
    "/think", "/ground", "/confirm",
})


def _is_slash_command(user_input: str) -> bool:
    """True only for genuine /commands, decided against KNOWN_COMMANDS.

    Absolute file paths also begin with '/' (e.g. /Users/me/shot.png) and must
    reach the model as normal prompts; for unknown '/' tokens, path-like input
    falls through while command-like typos still get the 'Unknown command' help.
    """
    if not user_input.startswith("/"):
        return False
    first_token = user_input.split()[0]
    if first_token.lower() in KNOWN_COMMANDS:
        return True
    if "/" in first_token[1:] or os.path.exists(first_token):
        return False  # a filesystem path, not a command
    return True  # unknown single '/word' — likely a typo, show command help


def main():
    parser = argparse.ArgumentParser(
        description="gemma4-agent - Autonomous Local AI Agent powered by Google DeepMind Gemma 4 models."
    )
    parser.add_argument(
        "-m", "--model",
        default="gemma4:26b",
        help="Local Gemma model tag (default: 'gemma4:26b')."
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Base URL for local Ollama engine (default: http://localhost:11434)"
    )
    parser.add_argument(
        "-v", "--voice",
        action="store_true",
        help="Enable 2-Way Voice Mode (Microphone Input + Spoken Speaker Output)."
    )
    parser.add_argument(
        "--mic-duration",
        type=int,
        default=7,
        help="Seconds to wait for speech to begin each listen cycle (default: 7). Recording itself stops after 1.5s of silence or 30s max."
    )
    parser.add_argument(
        "-e", "--exec",
        type=str,
        help="Run a single user query non-interactively and exit."
    )
    parser.add_argument(
        "--setup-voice",
        action="store_true",
        help="Download and cache the local voice transcription model (~74MB, one-time), then exit."
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Skip approval prompts for bash_run/python_eval/write_file (prompts are on by default in the REPL; -e mode never prompts)."
    )

    args = parser.parse_args()

    # One-time setup: fetch the local voice model, like `ollama pull` for speech
    if args.setup_voice:
        from gemma_agent.voice_input import ensure_voice_model
        ensure_voice_model()
        return

    model_name = args.model or "gemma4:26b"
    backend = LocalGemmaBackend(model_name=model_name, host=args.host)
    tools = ToolRegistry(ollama_host=args.host)
    agent = GemmaAgent(backend=backend, tool_registry=tools)

    def _confirm_tool(tool_name: str, tool_args: dict) -> bool:
        preview = str(tool_args.get("command") or tool_args.get("code")
                      or tool_args.get("filepath") or tool_args)[:200]
        from rich.markup import escape as _esc
        answer = ui.console.input(
            f"[warning]⚠  Approve [bold]{_esc(tool_name)}[/bold]: {_esc(preview)!s} — run it? \\[y/N] [/warning]"
        )
        return answer.strip().lower() in ("y", "yes")

    # Approval gate for system-modifying tools: on by default in the
    # interactive REPL; off in one-shot -e mode and with --yolo.
    if not args.exec and not args.yolo:
        tools.confirm_callback = _confirm_tool

    voice_mode = args.voice
    ui.voice_enabled = args.voice

    # If single execution query requested
    if args.exec:
        ui.print_info(f"Running local query with model '{model_name}'...")
        response = agent.run_turn(args.exec)
        ui.print_markdown(response, title="Agent Response")
        return

    # Interactive REPL session
    ui.print_banner(backend_name="100% LOCAL GEMMA (OLLAMA)", model_name=model_name)

    if voice_mode:
        ui.print_success("🎙️🔊 2-Way Voice Assistant Mode is ENABLED (Mic Input + Spoken Speaker Output).")
        from gemma_agent.voice_input import ensure_voice_model
        ensure_voice_model()

    # Check connection to local backend
    connected, msg = backend.check_connection()
    if connected:
        ui.print_success(msg)
    else:
        ui.print_error(msg)
        ui.print_info("Make sure Ollama is running locally (`ollama serve`).")

    mic_duration = args.mic_duration
    voice_engine = "whisper"

    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            if voice_mode:
                from gemma_agent.voice_input import listen_to_microphone
                spoken = listen_to_microphone(duration=mic_duration, engine=voice_engine, ollama_host=args.host)
                if spoken:
                    user_input = spoken
                    ui.print_success(f"🗣️  Recognized Spoken Instruction: \"{user_input}\"")
                else:
                    user_input = session.prompt(HTML("\n<ansiyellow>⚡</ansiyellow> <b>gemma4-agent</b> (Press Enter to mic, or type query) > ")).strip()
            else:
                user_input = session.prompt(HTML("\n<ansiyellow>⚡</ansiyellow> <b>gemma4-agent</b> > ")).strip()

            if not user_input:
                continue

            # Handle slash commands (file paths starting with '/' fall through
            # to the agent as normal prompts)
            if _is_slash_command(user_input):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd in ["/exit", "/quit"]:
                    ui.print_info("Exiting Gemma 4 CLI Agent. Goodbye! 👋")
                    break
                elif cmd == "/help":
                    ui.print_help()
                    continue
                elif cmd == "/clear":
                    agent.clear_history()
                    continue
                elif cmd in ["/voice", "/mic", "/talk"]:
                    was_on = voice_mode
                    options_given = False
                    for part in cmd_parts[1:]:
                        p = part.lower()
                        if p.isdigit():
                            mic_duration = max(1, int(p))  # 0 would cancel the mic instantly
                            options_given = True
                        elif p in ("gemma", "whisper"):
                            voice_engine = p
                            options_given = True
                        elif p in ("mute", "unmute"):
                            ui.speech_muted = p == "mute"
                            options_given = True
                        elif p == "engine":
                            continue  # allow '/voice engine gemma' phrasing
                        else:
                            ui.print_error(f"Unknown /voice option '{part}' — usage: /voice [seconds] [gemma|whisper] [mute|unmute]")
                    # Passing options while already listening reconfigures; bare /voice toggles
                    voice_mode = True if (options_given and was_on) else not was_on
                    ui.voice_enabled = voice_mode
                    if voice_mode:
                        speaker = "🔇 (muted)" if ui.speech_muted else "🔊"
                        ui.print_success(f"Voice Assistant Mode is now ENABLED 🎙️{speaker} ({mic_duration}s speech-wait · {voice_engine} ears)")
                        from gemma_agent.voice_input import ensure_voice_model
                        if voice_engine == "whisper":
                            ensure_voice_model()
                    else:
                        ui.print_success("Voice Assistant Mode is now DISABLED 🔇")
                    continue
                elif cmd in ["/stop", "/pause"]:
                    voice_mode = False
                    ui.voice_enabled = False
                    ui.print_success("Voice Assistant Mode is now DISABLED 🔇")
                    continue
                elif cmd == "/confirm":
                    if len(cmd_parts) > 1 and cmd_parts[1].lower() in ["on", "off"]:
                        enable = cmd_parts[1].lower() == "on"
                    else:
                        enable = tools.confirm_callback is None
                    tools.confirm_callback = _confirm_tool if enable else None
                    if enable:
                        ui.print_success("Tool approval ENABLED 🛡 — bash_run/python_eval/write_file ask before running")
                    else:
                        ui.print_success("Tool approval DISABLED — system tools run without asking")
                    continue
                elif cmd == "/ground":
                    if len(cmd_parts) > 1 and cmd_parts[1].lower() in ["on", "off"]:
                        agent.grounding_enabled = cmd_parts[1].lower() == "on"
                    else:
                        agent.grounding_enabled = not agent.grounding_enabled
                    if agent.grounding_enabled:
                        ui.print_success("Grounding ENABLED 🔎 — post-cutoff topics get live web context automatically")
                    else:
                        ui.print_success("Grounding DISABLED — the model answers from training data only")
                    continue
                elif cmd == "/think":
                    if len(cmd_parts) > 1 and cmd_parts[1].lower() in ["on", "off"]:
                        enabled = cmd_parts[1].lower() == "on"
                    else:
                        enabled = not agent.thinking_enabled
                    agent.set_thinking(enabled)
                    if enabled:
                        ui.print_success("Thinking mode ENABLED 🧠 — reasoning panels before complex tasks")
                    else:
                        ui.print_success("Thinking mode DISABLED ⚡ — direct answers, faster responses")
                    continue
                elif cmd in ["/model", "/backend"]:
                    if len(cmd_parts) > 1:
                        new_model = cmd_parts[1]
                        agent.backend.model_name = new_model
                        ui.print_success(f"Switched local Gemma model tag to '{new_model}'")
                    else:
                        ui.print_info(f"Current model: '{agent.backend.model_name}' — usage: /model <tag> (e.g. /model gemma4:12b)")
                    continue
                elif cmd == "/skills":
                    if len(cmd_parts) > 1 and cmd_parts[1].lower() == "clear":
                        removed = agent.skill_manager.clear_cache()
                        ui.print_success(f"Skill cache cleared ({removed} file(s) removed). Skills will re-fetch fresh on demand.")
                        continue
                    skills_text = "### ☁️ Active Agent Skills\n\n"
                    if agent.skill_manager.skills:
                        for s_id, s_info in agent.skill_manager.skills.items():
                            skills_text += f"- **{s_id}** — {s_info['description']}\n"
                    else:
                        skills_text += "_No skills cached yet — ask the agent about a Google Cloud topic (or any repo with SKILL.md files) and it will fetch the skill on demand._\n"
                    ui.print_markdown(skills_text, title="Agent Skills", speak=False)
                    continue
                elif cmd == "/mcp":
                    from gemma_agent.mcp import MCPManager
                    mcp_mgr = MCPManager()
                    sub = cmd_parts[1].lower() if len(cmd_parts) > 1 else ""
                    if sub == "connect":
                        if len(cmd_parts) > 2:
                            srv_name = cmd_parts[2]
                            srv_cmd = " ".join(cmd_parts[3:]) if len(cmd_parts) > 3 else "npx -y @modelcontextprotocol/server-postgres"
                            ui.print_success(mcp_mgr.connect_server(srv_name, srv_cmd))
                        else:
                            ui.print_error("Usage: /mcp connect <name> [command]")
                    elif sub in ["disconnect", "remove"]:
                        if len(cmd_parts) > 2:
                            ui.print_success(mcp_mgr.disconnect_server(cmd_parts[2]))
                        else:
                            ui.print_error("Usage: /mcp disconnect <name>")
                    else:
                        ui.print_markdown(mcp_mgr.list_servers(), title="Model Context Protocol (MCP)", speak=False)
                    continue
                elif cmd == "/tools":
                    tool_list = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tools.tools.values())
                    ui.print_markdown(tool_list, title="Registered Tools", speak=False)
                    continue
                elif cmd == "/history":
                    ui.print_info(f"History contains {len(agent.history)} messages (200-char preview per message).")
                    for m in agent.history:
                        role = m.get("role")
                        content = m.get("content", "")
                        if role != "system":
                            preview = content[:200] + ("..." if len(content) > 200 else "")
                            ui.print_markdown(f"**{role.upper()}**: {preview}", speak=False)
                    continue
                else:
                    ui.print_error(f"Unknown command '{cmd}'. Type /help for assistance.")
                    continue

            # Pause voice mode if user says "stop"
            if voice_mode and user_input.lower() in ["stop", "stop voice", "pause"]:
                voice_mode = False
                ui.voice_enabled = False
                ui.print_info("Voice mode paused. Switched back to normal typing prompt.")
                continue

            # Execute turn (Ctrl+C cancels the turn, not the session)
            ui.print_user_prompt(user_input)
            try:
                response = agent.run_turn(user_input)
            except KeyboardInterrupt:
                ui.print_info("Turn cancelled.")
                continue
            ui.print_markdown(response, title="Gemma Agent")

        except (KeyboardInterrupt, EOFError):
            ui.print_info("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            ui.print_error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
