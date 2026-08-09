"""
Gemma Agent Tool Registry Module.

Provides registry and execution engine for native agent capabilities
(bash execution, file I/O, python eval, web search/fetch, screenshots, ripgrep).
"""

import os
import base64
import subprocess
import requests
import json
import socket
import ipaddress
import urllib.parse
from typing import Dict, Any, Callable, List, Optional


def _validate_public_url(url: str):
    """SSRF guard: allow only http(s) URLs that resolve to public addresses.

    Blocks loopback (e.g. the Ollama control API), link-local (cloud metadata
    services), RFC1918/private, and reserved ranges. Returns the first
    validated IP so http connections can be PINNED to it — requests would
    otherwise re-resolve DNS independently, and a low-TTL rebinding between
    the two lookups bypasses the check (demonstrated in review).

    Returns:
        (ok: bool, reason: str, pinned_ip: Optional[str])
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "only http/https URLs are allowed", None
        host = parsed.hostname
        if not host:
            return False, "missing host", None
        first_ip = None
        for info in socket.getaddrinfo(host, None):
            ip_str = str(info[4][0]).split("%")[0]
            ip = ipaddress.ip_address(ip_str)
            if not ip.is_global:
                return False, f"'{host}' resolves to a non-public address ({ip})", None
            if first_ip is None:
                first_ip = ip_str
        return True, "", first_ip
    except Exception as e:
        return False, str(e), None


# Tools that can modify the system or execute arbitrary code — these require
# user approval when a confirm callback is installed.
CONFIRM_TOOLS = frozenset({"bash_run", "python_eval", "write_file"})


def _truncate(text: str, limit: int, what: str) -> str:
    """Cap tool output so a single command can't blow out the model's context."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{what} truncated: {len(text) - limit} of {len(text)} chars omitted]"


class ToolRegistry:
    """Registry and execution engine for Gemma CLI Agent tools."""

    def __init__(self, skill_manager: Optional[Any] = None,
                 confirm_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
                 ollama_host: str = "http://localhost:11434",
                 audio_model: str = "gemma4:12b"):
        """Initialize ToolRegistry and register default tools.

        Args:
            confirm_callback: When set, called as (tool_name, args) before any
                CONFIRM_TOOLS execution; returning False cancels the call.
            ollama_host: Local Ollama base URL (used by analyze_audio).
            audio_model: Audio-capable Gemma 4 variant for analyze_audio —
                must be E2B/E4B/12B; the 26B/31B cannot take audio input.
        """
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.skill_manager = skill_manager
        self.confirm_callback = confirm_callback
        self.ollama_host = ollama_host.rstrip("/")
        self.audio_model = audio_model
        self.register_defaults()

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable) -> None:
        """
        Register a new tool capability in the registry.

        Args:
            name (str): Unique tool identifier.
            description (str): Human-readable tool description for model system prompt.
            parameters (Dict[str, Any]): JSON Schema dictionary defining tool parameters.
            func (Callable): Callable function implementing tool logic.
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func
        }

    def execute(self, name: str, kwargs: Dict[str, Any]) -> str:
        """
        Execute tool function by name with provided arguments.

        Args:
            name (str): Name of tool to execute.
            kwargs (Dict[str, Any]): Keyword arguments to pass to function.

        Returns:
            str: Output or error message resulting from tool execution.
        """
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."
        # Models sometimes emit arguments as a JSON-encoded string; coerce it
        # so the call still runs instead of dying on ** unpacking.
        if isinstance(kwargs, str):
            try:
                parsed = json.loads(kwargs)
                kwargs = parsed if isinstance(parsed, dict) else None
            except Exception:
                kwargs = None
        if not isinstance(kwargs, dict):
            return f"Error: Tool '{name}' arguments must be a JSON object."
        if name in CONFIRM_TOOLS and self.confirm_callback is not None:
            try:
                approved = self.confirm_callback(name, kwargs)
            except Exception:
                approved = False
            if not approved:
                return (f"Cancelled by user: the {name} call was not approved. "
                        "Ask the user how to proceed or try a different approach.")
        try:
            return self.tools[name]["func"](**kwargs)
        except Exception as e:
            return f"Tool Execution Error ({name}): {str(e)}"

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Export OpenAI-compatible function schema definitions for model tools parameter.

        Returns:
            List[Dict[str, Any]]: Schema definitions list.
        """
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return schemas

    def register_defaults(self):
        # 1. Bash Tool
        self.register(
            name="bash_run",
            description="Execute shell commands on the local machine (macOS/Linux). Use to list files, run scripts, check status, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact shell command line string to execute."}
                },
                "required": ["command"]
            },
            func=self._bash_run
        )

        # 2. Read File Tool
        self.register(
            name="read_file",
            description="Read the text content of a file at the specified file path.",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file."}
                },
                "required": ["filepath"]
            },
            func=self._read_file
        )

        # 3. Write File Tool
        self.register(
            name="write_file",
            description="Write text content to a file at the specified file path.",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the target file."},
                    "content": {"type": "string", "description": "Content to write into the file."}
                },
                "required": ["filepath", "content"]
            },
            func=self._write_file
        )

        # 4. List Directory Tool
        self.register(
            name="list_directory",
            description="List files and subdirectories inside a given directory path.",
            parameters={
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "description": "Path to the directory. Defaults to current directory '.'"}
                },
                "required": []
            },
            func=self._list_directory
        )

        # 5. Python Eval Tool
        self.register(
            name="python_eval",
            description="Execute a Python script string and return standard output and standard error.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code snippet to execute."}
                },
                "required": ["code"]
            },
            func=self._python_eval
        )

        # 6. Web Fetch Tool
        self.register(
            name="web_fetch",
            description="Fetch plain text content from a URL via HTTP GET.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch content from."}
                },
                "required": ["url"]
            },
            func=self._web_fetch
        )

        # 7. Web Search Tool
        self.register(
            name="web_search",
            description="Perform a live web search. Use ONLY for information likely newer than your training data (current news, recent releases, live facts, prices) — not for general knowledge or programming concepts you already know.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query."}
                },
                "required": ["query"]
            },
            func=self._web_search
        )

        # 8. Fetch Agent Skill Tool
        self.register(
            name="fetch_skill",
            description=(
                "Fetch an Agent Skill (SKILL.md instructions) into context from a GitHub repo. "
                "Defaults to Google's official google/skills (e.g. gke, cloud-run, bigquery, alloydb, spanner); "
                "pass source='owner/repo' for community skill repos such as vercel-labs/skills."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name or topic of the skill (e.g., 'gke-basics', 'cloud-run', 'find-skills')."},
                    "source": {"type": "string", "description": "Optional GitHub 'owner/repo' to fetch from (default: 'google/skills')."}
                },
                "required": ["skill_name"]
            },
            func=self._fetch_skill
        )

        # 9. Take Screenshot Vision Tool
        self.register(
            name="take_screenshot",
            description="Capture a live desktop screenshot to a file (cross-platform via mss, with macOS screencapture fallback) for visual inspection and UI analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Optional custom filename to save screenshot (default: /tmp/gemma_screenshot.png)."}
                },
                "required": []
            },
            func=self._take_screenshot
        )

        # 10. Analyze Audio Tool (native Gemma 4 hearing on the 12B)
        self.register(
            name="analyze_audio",
            description=(
                "Analyze a local audio file (.wav or .mp3) using Gemma 4's native audio "
                "understanding: transcribe speech, describe sounds, judge tone. "
                "Runs fully locally on an audio-capable Gemma 4 variant."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the local audio file (.wav or .mp3)."},
                    "question": {"type": "string", "description": "Optional question about the audio (default: transcribe and describe it)."}
                },
                "required": ["filepath"]
            },
            func=self._analyze_audio
        )

        # 11. Ripgrep Fast Code Search Tool
        self.register(
            name="ripgrep_search",
            description="Perform instant high-speed regex/keyword code searches across large codebases.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or text pattern to search for."},
                    "path": {"type": "string", "description": "Directory or file path to search. Defaults to '.'"}
                },
                "required": ["query"]
            },
            func=self._ripgrep_search
        )



    def get_system_prompt_tool_descriptions(self) -> str:
        # Names and descriptions only — full JSON schemas are already sent
        # natively in the API payload; duplicating them here cost ~1K prompt
        # tokens on every single turn.
        desc = "AVAILABLE AGENT TOOLS:\n"
        for name, tool in self.tools.items():
            params = ", ".join(tool["parameters"].get("properties", {}).keys())
            desc += f"- {name}({params}): {tool['description']}\n"
        return desc

    # Implementation helper functions
    def _bash_run(self, command: str) -> str:
        try:
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd()
            )
            out = _truncate(res.stdout.strip(), 6000, "stdout")
            err = _truncate(res.stderr.strip(), 3000, "stderr")
            ret = f"Exit code: {res.returncode}\n"
            if out:
                ret += f"STDOUT:\n{out}\n"
            if err:
                ret += f"STDERR:\n{err}\n"
            return ret.strip()
        except subprocess.TimeoutExpired:
            return "Error: Command execution timed out (60s limit)."
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _read_file(self, filepath: str) -> str:
        try:
            if not os.path.exists(filepath):
                return f"Error: File '{filepath}' does not exist."
            with open(filepath, "r", encoding="utf-8") as f:
                return _truncate(f.read(), 12000, "file content (use ripgrep_search or bash_run to target sections)")
        except Exception as e:
            return f"Error reading file '{filepath}': {str(e)}"

    def _write_file(self, filepath: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to '{filepath}'."
        except Exception as e:
            return f"Error writing file '{filepath}': {str(e)}"

    def _list_directory(self, directory_path: str = ".") -> str:
        try:
            if not os.path.exists(directory_path):
                return f"Error: Path '{directory_path}' does not exist."
            items = sorted(os.listdir(directory_path))
            res = []
            for item in items[:300]:
                full_p = os.path.join(directory_path, item)
                kind = "DIR " if os.path.isdir(full_p) else "FILE"
                size = os.path.getsize(full_p) if os.path.isfile(full_p) else 0
                res.append(f"[{kind}] {item} ({size} bytes)")
            if len(items) > 300:
                res.append(f"... [{len(items) - 300} more entries omitted]")
            if not res:
                return "(Empty directory)"
            return _truncate("\n".join(res), 8000, "directory listing")
        except Exception as e:
            return f"Error listing directory '{directory_path}': {str(e)}"

    def _python_eval(self, code: str) -> str:
        try:
            res = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            out = _truncate(res.stdout.strip(), 6000, "stdout")
            err = _truncate(res.stderr.strip(), 3000, "stderr")
            ret = f"Exit code: {res.returncode}\n"
            if out:
                ret += f"STDOUT:\n{out}\n"
            if err:
                ret += f"STDERR:\n{err}\n"
            return ret.strip()
        except subprocess.TimeoutExpired:
            return "Error: Python execution timed out (30s limit)."
        except Exception as e:
            return f"Error evaluating Python code: {str(e)}"

    def _web_fetch(self, url: str) -> str:
        ok, why, pinned_ip = _validate_public_url(url)
        if not ok:
            return f"Error: URL blocked ({why})."
        try:
            resp = None
            for _ in range(4):  # follow up to 3 redirects, re-validating every hop
                parsed = urllib.parse.urlparse(url)
                headers = {"User-Agent": "GemmaCLI/1.0"}
                fetch_url = url
                # For plain http, connect to the ALREADY-VALIDATED IP (with a Host
                # header) so a DNS rebinding between validation and fetch cannot
                # redirect us to an internal address. For https the certificate
                # check already binds the connection to the real hostname.
                if parsed.scheme == "http" and pinned_ip:
                    netloc = pinned_ip if ":" not in pinned_ip else f"[{pinned_ip}]"
                    if parsed.port:
                        netloc += f":{parsed.port}"
                    fetch_url = urllib.parse.urlunparse(parsed._replace(netloc=netloc))
                    headers["Host"] = parsed.hostname
                resp = requests.get(fetch_url, headers=headers, timeout=15, allow_redirects=False)
                if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
                    url = urllib.parse.urljoin(url, resp.headers["location"])
                    ok, why, pinned_ip = _validate_public_url(url)
                    if not ok:
                        return f"Error: redirect blocked ({why})."
                    continue
                break
            if resp.status_code in (301, 302, 303, 307, 308):
                return "Error: too many redirects (limit: 3)."
            resp.raise_for_status()
            text = resp.text
            # Simple strip HTML if present
            if "<html" in text.lower():
                import re
                text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
            return text[:4000] # Return top 4000 characters
        except Exception as e:
            return f"Error fetching URL '{url}': {str(e)}"

    def _web_search(self, query: str) -> str:
        try:
            import re
            url = "https://html.duckduckgo.com/html/"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            resp = requests.post(url, data={"q": query}, headers=headers, timeout=12)
            if resp.status_code != 200:
                return f"Web search HTTP Error {resp.status_code}"
            
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            results = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:5]]
            
            if results:
                return "\n\n".join(f"Result {i+1}: {res}" for i, res in enumerate(results))
            return f"No search results found for query '{query}'."
        except Exception as e:
            return f"Web search error: {str(e)}"

    def _fetch_skill(self, skill_name: str, source: str = "google/skills") -> str:
        if self.skill_manager:
            return self.skill_manager.search_and_fetch_github_skill(skill_name, source=source)
        from gemma_agent.skills import SkillManager
        sm = SkillManager()
        return sm.search_and_fetch_github_skill(skill_name, source=source)

    def _take_screenshot(self, filename: Optional[str] = None) -> str:
        target_path = filename or "/tmp/gemma_screenshot.png"
        try:
            # Primary: Pure python mss screenshot engine
            import mss
            with mss.MSS() as sct:
                sct.shot(output=target_path)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return f"Screenshot captured successfully and saved to: {target_path}. You can analyze it visually now!"
        except Exception:
            pass

        # Fallback: macOS screencapture CLI
        try:
            res = subprocess.run(["screencapture", "-x", target_path], capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return f"Screenshot captured successfully and saved to: {target_path}. You can analyze it visually now!"
            return f"Screenshot error (exit code {res.returncode}): {res.stderr.strip() or 'could not capture display'}"
        except Exception as e:
            return f"Screenshot error: {str(e)}"

    def _analyze_audio(self, filepath: str, question: str = "Transcribe any speech and describe this audio.") -> str:
        """Native Gemma 4 audio understanding via Ollama's OpenAI-compatible
        endpoint — the only path that delivers audio to the model (the native
        /api/chat silently drops audio fields; verified empirically)."""
        path = os.path.expanduser(filepath)
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext not in ("wav", "mp3"):
            return f"Error: analyze_audio supports .wav and .mp3 files (got '{ext or 'no extension'}')."
        if not os.path.isfile(path):
            return f"Error: File '{filepath}' does not exist."
        try:
            size = os.path.getsize(path)
            if size > 10 * 1024 * 1024:
                return f"Error: Audio file too large ({size // (1024 * 1024)}MB; 10MB limit)."
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            payload = {
                "model": self.audio_model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": b64, "format": ext}},
                        {"type": "text", "text": question},
                    ],
                }],
                # Thinking off + capped: with thinking on, the answer lands in
                # `reasoning` and content comes back empty once the cap hits.
                "reasoning_effort": "none",
                "max_tokens": 700,
            }
            resp = requests.post(f"{self.ollama_host}/v1/chat/completions", json=payload, timeout=300)
            if resp.status_code == 404:
                return (f"Error: audio model '{self.audio_model}' is not available locally. "
                        f"Run: ollama pull {self.audio_model}")
            if resp.status_code != 200:
                return f"Error: audio analysis failed (HTTP {resp.status_code}): {resp.text[:200]}"
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return "Error: the audio model returned an empty response."
            return _truncate(content, 4000, "audio analysis")
        except Exception as e:
            return f"Error analyzing audio '{filepath}': {str(e)}"

    def _ripgrep_search(self, query: str, path: str = ".") -> str:
        search_path = os.path.expanduser(path)
        try:
            res = subprocess.run(["rg", "--line-number", "--max-columns", "200", "--max-count", "20", query, search_path], capture_output=True, text=True, timeout=15)
            out = res.stdout.strip()
            if out:
                return out[:3000]
        except Exception:
            pass

        # Python fallback search if rg is not installed or returns nothing
        try:
            matches = []
            for root, dirs, files in os.walk(search_path):
                if ".git" in root or "node_modules" in root or ".venv" in root or "__pycache__" in root:
                    continue
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as file_obj:
                            for idx, line in enumerate(file_obj, 1):
                                if query.lower() in line.lower():
                                    matches.append(f"{fp}:{idx}:{line.strip()[:200]}")
                                    if len(matches) >= 20:
                                        break
                    except Exception:
                        pass
                    if len(matches) >= 20:
                        break
                if len(matches) >= 20:
                    break
            if not matches:
                return f"No matches found for pattern '{query}' in {search_path}."
            return _truncate("\n".join(matches), 3000, "search results")
        except Exception as e:
            return f"Search error: {str(e)}"
