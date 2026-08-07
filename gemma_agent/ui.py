from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from rich.syntax import Syntax
from rich import box
from rich.theme import Theme

import re
import subprocess

from rich.markup import escape

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "agent": "bold magenta",
    "tool": "bold cyan",
    "user": "bold green",
})

console = Console(theme=custom_theme)
voice_enabled = False


_speech_proc: Optional[subprocess.Popen] = None


def speak_text(text: str):
    """Speak text out loud asynchronously using macOS native 'say' command."""
    global _speech_proc
    if not voice_enabled or not text:
        return
    try:
        # Strip markdown syntax and code blocks for spoken speech
        clean_text = re.sub(r"```.*?```", "code block omitted", text, flags=re.DOTALL)
        clean_text = re.sub(r"[`#*_\-\[\]]", "", clean_text)
        clean_text = clean_text.strip()[:500]  # Speak top 500 chars
        if clean_text:
            wait_for_speech_to_finish()
            _speech_proc = subprocess.Popen(["say", "-r", "210", clean_text])
    except Exception:
        pass


def wait_for_speech_to_finish(timeout: Optional[float] = None):
    """Block until in-progress spoken output ends, so the microphone never records our own TTS voice."""
    global _speech_proc
    proc = _speech_proc
    if proc is not None:
        try:
            proc.wait(timeout=timeout)
        except Exception:
            pass
        _speech_proc = None


def print_banner(backend_name: str, model_name: str):
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    
    header_text = Text("✨ 💎 GOOGLE GEMMA 4 AGENTIC CLI", style="bold cyan")
    
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="bold yellow", justify="right")
    info_table.add_column(style="bold white", justify="left")
    info_table.add_row("Backend:", f"[bold cyan]{backend_name}[/bold cyan]")
    info_table.add_row("Model:", f"[bold green]{model_name}[/bold green]")
    info_table.add_row("Commands:", "[dim]/voice, /stop, /model, /tools, /skills, /clear, /help, /exit[/dim]")

    panel_content = Table.grid(expand=True)
    panel_content.add_column()
    panel_content.add_row(header_text)
    panel_content.add_row("")
    panel_content.add_row(info_table)

    console.print()
    console.print(Panel(
        panel_content,
        box=box.ROUNDED,
        border_style="bold cyan",
        subtitle="[dim]Type your query, talk via mic, or type /help[/dim]",
        subtitle_align="right",
        padding=(1, 2)
    ))


def print_user_prompt(prompt: str):
    console.print(f"\n[user]💬 You:[/user] {escape(prompt)}")


def print_agent_thought(thought: str):
    console.print(Panel(
        Text(thought.strip(), style="italic dim yellow"),
        title="[bold yellow]🧠 Internal Reasoning[/bold yellow]",
        title_align="left",
        box=box.ROUNDED,
        border_style="yellow",
        padding=(0, 1)
    ))


def print_thinking_panel(thinking_text: str):
    console.print(Panel(
        Markdown(thinking_text.strip()),
        title="[bold cyan]🧠 Deep Reasoning & Planning (Gemma 4 Think)[/bold cyan]",
        title_align="left",
        box=box.ROUNDED,
        border_style="bold cyan",
        padding=(0, 1)
    ))


def print_tool_call(tool_name: str, args: Dict[str, Any]):
    formatted_args = []
    for k, v in args.items():
        key = escape(str(k))
        if isinstance(v, str) and ("\n" in v or len(v) > 60):
            formatted_args.append(f"[bold yellow]{key}[/bold yellow]=[dim]<{len(v)} chars>[/dim]")
        else:
            formatted_args.append(f"[bold yellow]{key}[/bold yellow]={escape(repr(v))}")

    args_str = ", ".join(formatted_args)

    console.print(f"  [bold yellow]⚡[/bold yellow] [tool]Tool Execution:[/tool] [bold cyan]{escape(str(tool_name))}[/bold cyan]({args_str})")
    
    if tool_name == "write_file" and "content" in args and "filepath" in args:
        code_preview = args["content"]
        if len(code_preview.splitlines()) > 15:
            lines = code_preview.splitlines()
            code_preview = "\n".join(lines[:12]) + f"\n... [{len(lines)-12} more lines]"
        
        ext = args["filepath"].split(".")[-1] if "." in args["filepath"] else "text"
        syntax = Syntax(code_preview, ext, theme="monokai", line_numbers=True)
        console.print(Panel(
            syntax,
            title=f"[dim]📁 Creating File: {escape(str(args['filepath']))}[/dim]",
            title_align="left",
            box=box.ROUNDED,
            border_style="dim blue",
            padding=(0, 1)
        ))


def print_tool_result(result: str, is_error: bool = False):
    border_color = "red" if is_error else "dim green"
    label = "[bold red]❌ Tool Output Error[/bold red]" if is_error else "[bold green]✅ Tool Output Success[/bold green]"
    
    max_len = 800
    display_res = result.strip()
    if len(display_res) > max_len:
        display_res = display_res[:max_len] + f"\n... [truncated {len(display_res) - max_len} chars]"
    
    console.print(Panel(
        escape(display_res),
        title=label,
        title_align="left",
        box=box.ROUNDED,
        border_style=border_color,
        padding=(0, 1)
    ))


def print_markdown(content: str, title: Optional[str] = None):
    md = Markdown(content.strip())
    panel_title = f"[bold cyan]✨ 💎 Google Gemma 4[/bold cyan] [bold magenta]{title or 'Agent'}[/bold magenta]"
    console.print(Panel(
        md,
        title=panel_title,
        title_align="left",
        box=box.ROUNDED,
        border_style="bold magenta",
        padding=(1, 2)
    ))
    
    speak_text(content)


def print_info(msg: str):
    console.print(f"[info]ℹ️ {escape(msg)}[/info]")


def print_success(msg: str):
    console.print(f"[success]✨ {escape(msg)}[/success]")


def print_error(msg: str):
    console.print(f"[error]❌ {escape(msg)}[/error]")


def print_turn_metrics(
    duration_sec: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    backend_label: str = "Local Engine"
):
    total_tokens = prompt_tokens + completion_tokens

    metrics_table = Table.grid(padding=(0, 2))
    metrics_table.add_column(style="dim white")
    metrics_table.add_column(style="dim white")
    metrics_table.add_column(style="dim white")

    token_info = f"🪙 Tokens: [bold yellow]{prompt_tokens}[/bold yellow] prompt + [bold yellow]{completion_tokens}[/bold yellow] completion = [bold cyan]{total_tokens}[/bold cyan] total" if total_tokens > 0 else "🪙 Tokens: [dim]N/A[/dim]"

    metrics_table.add_row(
        f"⏱️  Time: [bold green]{duration_sec:.2f}s[/bold green]",
        token_info,
        f"⚙️ Backend: [bold blue]{backend_label}[/bold blue]"
    )
    
    console.print(Panel(
        metrics_table,
        title="[dim]📊 Performance & Telemetry[/dim]",
        title_align="right",
        box=box.ROUNDED,
        border_style="dim cyan",
        padding=(0, 1)
    ))


def print_help():
    table = Table(title="✨ Gemma 4 Agent CLI Commands", box=box.ROUNDED, border_style="cyan")
    table.add_column("Command", style="bold yellow")
    table.add_column("Description", style="white")
    table.add_row("/help", "Show this help message")
    table.add_row("/clear", "Clear session conversation history")
    table.add_row("/tools", "List all active agent tools")
    table.add_row("/skills", "List all active Google Cloud Skills (google/skills)")
    table.add_row("/mcp", "Register and list MCP server configs (experimental)")
    table.add_row("/history", "Display conversation transcript")
    table.add_row("/voice", "Toggle 2-Way Voice Mode (Mic Input + Speaker Output)")
    table.add_row("/stop", "Stop/Disable Voice Assistant mode and return to typing")
    table.add_row("/model <tag>", "Switch active local Gemma model tag on the fly (alias: /backend)")
    table.add_row("/exit or /quit", "Exit the agent CLI")
    console.print(table)
