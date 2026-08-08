"""
Gemma Agent Orchestrator Module.

Manages conversation state, tool call execution loops, reasoning block extraction,
and telemetry tracking for the Google Gemma 4 CLI Agent.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from rich.markup import escape
from gemma_agent.backends import BaseBackend
from gemma_agent.tools import ToolRegistry
from gemma_agent.skills import SkillManager
from gemma_agent import ui

SYSTEM_PROMPT_TEMPLATE = """You are Gemma CLI Agent, an autonomous AI assistant powered by Google Gemma 4 models.
You assist users directly in their terminal environment by answering questions, writing code, executing commands, and analyzing files.
You are equipped with Agent Skills: official Google Cloud skills (google/skills — Cloud Run, GKE, BigQuery, AlloyDB, Spanner, and more) plus community skills from any GitHub repo using the SKILL.md convention.

{tool_descriptions}

THINKING & REASONING MODE:
1. Before answering or executing tools, wrap your step-by-step reasoning inside `<think>...</think>` tags.
2. Explain your plan, why you chose specific tools, and what actions you are taking.

CRITICAL RULES FOR TOOL EXECUTION:
1. When you need to perform an action (read files, run bash commands, write code, run python snippets), call a tool!
2. Once a tool has been executed and you receive its result, DO NOT call the exact same tool with the exact same parameters again.
3. After the tool succeeds, summarize the result and provide your final response to the user.
4. You can call tools using native function calling OR by outputting a clean JSON block:
```json
{{
  "tool": "tool_name",
  "arguments": {{
    "arg_name": "arg_value"
  }}
}}
```
"""


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
        self.history: List[Dict[str, Any]] = []
        self._init_system_prompt()

    def _init_system_prompt(self) -> None:
        """Construct system prompt including tool descriptions and registered skills."""
        tool_desc = self.tools.get_system_prompt_tool_descriptions()
        skills_desc = self.skill_manager.get_all_skills_system_prompt()
        sys_msg = SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_desc) + skills_desc
        self.history = [{"role": "system", "content": sys_msg}]

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
        
        tools_schema = self.tools.get_schemas()
        executed_signatures = set()
        
        total_turn_duration = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_backend_label = "Local Engine"

        for _ in range(max_iterations):
            # Request response from backend with animated spinner
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


