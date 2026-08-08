"""
Gemma Agent Orchestrator Module.

Manages conversation state, tool call execution loops, reasoning block extraction,
and telemetry tracking for the Google Gemma 4 CLI Agent.
"""

import re
import json
import inspect
from typing import List, Dict, Any, Optional, Tuple
from rich.markup import escape
from gemma_agent.backends import BaseBackend
from gemma_agent.tools import ToolRegistry
from gemma_agent.skills import SkillManager
from gemma_agent import ui

SYSTEM_PROMPT_TEMPLATE = """You are Gemma CLI Agent, an autonomous terminal assistant powered by Google Gemma 4 (knowledge cutoff: January 2025).
You answer questions, write code, run commands, and analyze files directly in the user's terminal.

{tool_descriptions}

RULES:
1. Use tools to act (run commands, read/write files, execute code). After a tool result arrives, do not repeat the identical call — use the result and answer.
2. Answer from your own knowledge for things you know well. Use web_search or fetch_skill for information newer than your cutoff or when unsure of exact APIs — never invent imports, class names, or flags, and never write mock code unless asked.
3. Be concise: short questions deserve short answers.
4. Tool calls may also be written as a JSON block — escape quotes and newlines when arguments contain code or multiline text:
```json
{{"tool": "tool_name", "arguments": {{"arg_name": "arg_value"}}}}
```

{thinking_section}
"""

THINKING_SECTION = """THINKING & REASONING MODE:
1. For multi-step tasks, coding, analysis, or anything involving tools, wrap your step-by-step reasoning inside `<think>...</think>` tags before acting — explain your plan and why you chose specific tools.
2. For greetings, simple factual questions, and casual conversation, answer directly WITHOUT thinking tags — a short question deserves a fast answer."""

NO_THINKING_SECTION = """THINKING MODE IS OFF:
Do NOT use <think> tags or write out reasoning steps. Answer directly and concisely."""

# Products/SDKs released or heavily changed after the model's January 2025
# cutoff. A request mentioning these gets a per-turn grounding nudge — the
# model cannot reliably know what it doesn't know, so the trigger is ours.
GROUNDING_TRIGGERS = (
    "adk", "agent development kit", "agents-cli", "agent starter pack",
    "a2a", "agent2agent", "agent engine", "rag engine",
    "gemma 4", "gemma4", "gemini 3", "gemini agents api",
    "mcp server", "mcp tool", "model context protocol",
    "skills.sh", "agent skills",
)

GROUNDING_NOTE = (
    "System Note: This request involves a product or SDK released or changed after your "
    "January 2025 knowledge cutoff — your training data about it is incomplete or absent. "
    "NEVER claim such a product does not exist, and NEVER guess its APIs. Live web results "
    "for this request are provided below: base your answer on them, and call fetch_skill "
    "or web_search yourself if you need more detail."
)


def _normalize_args(raw: Any) -> Dict[str, Any]:
    """Coerce tool-call arguments to a dict.

    Local models frequently emit `arguments` as a JSON-encoded STRING (the
    OpenAI wire format) instead of an object; passing that through crashes
    the loop downstream.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


# Anchored patterns for genuine tool failures. Matching anywhere in the text
# false-positives on file contents/logs that merely mention errors.
_ERR_PREFIX = re.compile(r"^(Error\b|Tool Execution Error|Web search|Screenshot error|Search error)")
_ERR_EXIT = re.compile(r"^Exit code:\s*-?[1-9]")


class GemmaAgent:
    """
    Autonomous Agent orchestrator for Google Gemma 4 models.

    Handles message history, backend model communication, function tool invocation,
    self-correction loops, and terminal UI rendering.
    """

    def __init__(self, backend: BaseBackend, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize GemmaAgent instance.

        Args:
            backend (BaseBackend): Active model backend (LocalGemmaBackend).
            tool_registry (Optional[ToolRegistry]): Tool registry for tool execution.
        """
        self.backend = backend
        self.skill_manager = SkillManager()
        self.tools = tool_registry or ToolRegistry(skill_manager=self.skill_manager)
        if not self.tools.skill_manager:
            self.tools.skill_manager = self.skill_manager
        self.thinking_enabled = True
        self.grounding_enabled = True
        self.history: List[Dict[str, Any]] = []
        self._init_system_prompt()

    def _build_system_prompt(self) -> str:
        tool_desc = self.tools.get_system_prompt_tool_descriptions()
        skills_desc = self.skill_manager.get_all_skills_system_prompt()
        thinking = THINKING_SECTION if self.thinking_enabled else NO_THINKING_SECTION
        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_desc, thinking_section=thinking
        ) + skills_desc

    def _init_system_prompt(self) -> None:
        """Construct system prompt including tool descriptions and registered skills."""
        self.history = [{"role": "system", "content": self._build_system_prompt()}]

    def set_thinking(self, enabled: bool) -> None:
        """Toggle reasoning mode, rebuilding the system prompt in place so the
        current conversation is preserved."""
        self.thinking_enabled = enabled
        if self.history and self.history[0].get("role") == "system":
            self.history[0]["content"] = self._build_system_prompt()
        else:
            self.history.insert(0, {"role": "system", "content": self._build_system_prompt()})

    def clear_history(self) -> None:
        """Reset conversation history back to initial system prompt."""
        self._init_system_prompt()
        ui.print_success("Conversation history cleared.")

    def run_turn(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Execute a single conversational turn, supporting multi-step tool calls.

        Args:
            user_input (str): User instruction or question.
            max_iterations (int): Safety cap on consecutive tool executions.

        Returns:
            str: Final response string from the assistant.
        """
        self.history.append({"role": "user", "content": user_input})

        # Forced grounding — even injected nudges proved unreliable under
        # sampling: the model would skip searching or confidently deny that
        # post-cutoff products exist. So the agent grounds FOR it, visibly.
        lowered = user_input.lower()
        if self.grounding_enabled and any(t in lowered for t in GROUNDING_TRIGGERS):
            self._auto_ground(user_input)

        tools_schema = self.tools.get_schemas()
        executed_signatures = set()
        
        total_turn_duration = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_backend_label = "Local Engine"

        # Stream tokens live when the backend supports it (test stubs may not)
        supports_stream = "on_token" in inspect.signature(self.backend.generate_response).parameters

        for _ in range(max_iterations):
            if supports_stream:
                accumulated: List[str] = []
                with ui.streaming_live() as update_live:
                    def _on_token(delta: str) -> None:
                        accumulated.append(delta)
                        update_live("".join(accumulated))
                    content, tool_calls, metrics = self.backend.generate_response(
                        self.history, tools_schema=tools_schema, on_token=_on_token
                    )
            else:
                with ui.console.status(f"[bold cyan]🧠 Generating response ({escape(self.backend.model_name)})...[/bold cyan]", spinner="dots"):
                    content, tool_calls, metrics = self.backend.generate_response(self.history, tools_schema=tools_schema)
            
            # Aggregate metrics
            total_turn_duration += metrics.get("duration_sec", 0.0)
            total_prompt_tokens += metrics.get("prompt_tokens", 0)
            total_completion_tokens += metrics.get("completion_tokens", 0)
            last_backend_label = metrics.get("backend_label", last_backend_label)

            # Extract <think>...</think> reasoning blocks if present
            thinking_text, content = self._extract_thinking_text(content)
            if thinking_text:
                ui.print_thinking_panel(thinking_text)

            # Check for ReAct text fallback tool calls if native tool_calls is empty.
            # Some models wrap the tool block inside their <think> section — check there too.
            if not tool_calls:
                tool_calls = self._parse_json_tool_calls(content)
            if not tool_calls and thinking_text:
                tool_calls = self._parse_json_tool_calls(thinking_text)

            # If no tool calls requested, we have the final assistant message
            if not tool_calls:
                self._strip_dedup_notes()
                self.history.append({"role": "assistant", "content": content})
                ui.print_turn_metrics(
                    duration_sec=total_turn_duration,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    backend_label=last_backend_label
                )
                return content

            # Deduplicate tool calls in the same turn
            unique_tool_calls = []
            for tc in tool_calls:
                t_name = tc.get("name") or tc.get("tool")
                t_args = _normalize_args(tc.get("arguments") or tc.get("args") or {})
                tc["arguments"] = t_args
                sig = f"{t_name}:{json.dumps(t_args, sort_keys=True)}"
                if sig not in executed_signatures:
                    executed_signatures.add(sig)
                    unique_tool_calls.append(tc)
            
            if not unique_tool_calls:
                # All requested tool calls were already executed! Force final summary response.
                self.history.append({
                    "role": "system",
                    "content": "System Note: Requested tool call was already executed in a previous step. Please synthesize your final response for the user."
                })
                continue

            # If assistant returned reasoning or text alongside tool call
            if content and content.strip():
                ui.print_agent_thought(content)
                self.history.append({"role": "assistant", "content": content, "tool_calls": unique_tool_calls})
            else:
                self.history.append({"role": "assistant", "content": "", "tool_calls": unique_tool_calls})

            # Execute tool calls
            for tool_call in unique_tool_calls:
                tool_name = tool_call.get("name") or tool_call.get("tool")
                args = _normalize_args(tool_call.get("arguments") or tool_call.get("args") or {})

                ui.print_tool_call(tool_name, args)
                result = self.tools.execute(tool_name, args)

                is_err = bool(_ERR_PREFIX.match(result)) or bool(_ERR_EXIT.match(result))
                ui.print_tool_result(result, is_error=is_err)

                if is_err:
                    # A failed call must stay retryable, otherwise the model cannot
                    # re-run the same command after fixing the underlying problem.
                    executed_signatures.discard(f"{tool_name}:{json.dumps(args, sort_keys=True)}")
                    ui.print_info("🔄 Self-Healing Loop Active: Error detected. Gemma 4 will auto-correct and re-run...")

                # Append tool execution result back to conversation history
                self.history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": result
                })

        self._strip_dedup_notes()
        limit_msg = f"Reached maximum tool execution limit ({max_iterations} iterations)."
        # Close the turn in history so the next turn's model knows it was cut off,
        # and report metrics just like a normal turn.
        self.history.append({"role": "assistant", "content": limit_msg})
        ui.print_turn_metrics(
            duration_sec=total_turn_duration,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            backend_label=last_backend_label
        )
        return limit_msg

    def _auto_ground(self, user_input: str) -> None:
        """Deterministically fetch live web context for post-cutoff topics and
        inject it before the model's first response. Transparent (rendered like
        any tool call) and toggleable via /ground."""
        query = " ".join(user_input.split())[:120]
        ui.print_info("🔎 Post-cutoff topic detected — grounding with live web results before answering (/ground off to disable)...")
        ui.print_tool_call("web_search", {"query": query})
        result = self.tools.execute("web_search", {"query": query})
        ui.print_tool_result(result, is_error=result.startswith("Web search"))
        self.history.append({
            "role": "system",
            "content": f"{GROUNDING_NOTE}\n\nLive web results:\n{result[:2500]}"
        })

    def _strip_dedup_notes(self) -> None:
        """Remove mid-turn dedup steering notes so they never leak into later turns."""
        self.history = [
            m for m in self.history
            if not (m.get("role") == "system" and str(m.get("content", "")).startswith("System Note:"))
        ]

    def _parse_json_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse structured JSON tool call blocks formatted by model output.

        Args:
            text (str): Raw text generated by model.

        Returns:
            Optional[List[Dict[str, Any]]]: Parsed list of tool call dictionary objects or None.
        """
        if not text:
            return None
        
        # Match fenced JSON blocks: ```json {...} ```, ```JSON, untagged ``` {...} ```,
        # and top-level arrays of calls.
        pattern = r"```(?:json)?\s*([\[{].*?[\]}])\s*```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        tool_calls = []
        for raw_json in matches:
            try:
                data = json.loads(raw_json)
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and "tool" in item:
                    tool_calls.append({
                        "name": item["tool"],
                        "arguments": _normalize_args(item.get("arguments", {}))
                    })

        if not tool_calls:
            # Fallback inline JSON match
            try:
                if '"tool":' in text:
                    data = json.loads(text.strip())
                    if isinstance(data, dict) and "tool" in data:
                        tool_calls.append({
                            "name": data["tool"],
                            "arguments": _normalize_args(data.get("arguments", {}))
                        })
            except Exception:
                pass

        return tool_calls if tool_calls else None

    def _extract_thinking_text(self, text: str) -> Tuple[Optional[str], str]:
        """
        Extract <think>...</think> reasoning blocks from text.

        Args:
            text (str): Output text containing optional thinking tags.

        Returns:
            Tuple[Optional[str], str]: (Extracted thinking text, Cleaned output text).
        """
        if not text:
            return None, ""
        
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            thinking_content = "\n\n".join(m.strip() for m in matches if m.strip())
            clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
            return thinking_content, clean_text

        # Truncated generation: an opening <think> with no close tag would
        # otherwise leak raw reasoning into the final answer and history.
        if "<think>" in text:
            pre, _, rest = text.partition("<think>")
            return (rest.strip() or None), pre.strip()

        return None, text


