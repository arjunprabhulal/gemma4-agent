"""
Gemma Agent Backends Module.

Provides abstract and concrete model backends for communicating with
local Ollama instances (LocalGemmaBackend).
"""

import os
import json
import time
import base64
import mimetypes
import requests
from typing import List, Dict, Any, Tuple, Optional


class BaseBackend:
    """Abstract base class defining backend interface for Gemma models."""

    def __init__(self, model_name: str):
        """
        Initialize base backend.

        Args:
            model_name (str): Identifier name or tag of the model.
        """
        self.model_name = model_name

    def check_connection(self) -> Tuple[bool, str]:
        """
        Check health/connectivity of backend service.

        Returns:
            Tuple[bool, str]: (Is connected, Status message).
        """
        raise NotImplementedError

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Generate completion response from backend model.

        Args:
            messages (List[Dict[str, Any]]): Conversation message history.
            tools_schema (Optional[List[Dict[str, Any]]]): Available function schemas.

        Returns:
            Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
                (Response content text, Requested tool calls list, Execution metrics dict).
        """
        raise NotImplementedError


def _extract_image_paths(content: str) -> Tuple[str, List[str]]:
    """
    Detect readable local image file paths mentioned in the text.

    The text is returned UNCHANGED — removing mentioned filenames breaks
    file-operation requests ("delete logo.png" must keep its target). Only
    readable regular files are attached, so directories or unreadable files
    named *.png stay visible in the prompt instead of vanishing silently.

    Returns:
        Tuple[str, List[str]]: (Original text, list of attachable image paths).
    """
    image_paths = []
    for t in content.split():
        clean_t = os.path.expanduser(t.strip("\"'"))
        if (clean_t.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))
                and os.path.isfile(clean_t) and os.access(clean_t, os.R_OK)):
            image_paths.append(clean_t)
    return content, image_paths


class LocalGemmaBackend(BaseBackend):
    """100% Local & Private Gemma Backend (via Ollama) with Multimodal Vision support."""

    def __init__(self, model_name: str = "gemma4:26b", host: str = "http://localhost:11434"):
        """
        Initialize LocalGemmaBackend.

        Args:
            model_name (str): Local Ollama model tag (default: 'gemma4:26b').
            host (str): Base URL of local Ollama service (default: 'http://localhost:11434').
        """
        super().__init__(model_name)
        self.host = host.rstrip('/')

    def check_connection(self) -> Tuple[bool, str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            if r.status_code == 200:
                try:
                    models = [m.get('name') or m.get('model', 'unknown') for m in r.json().get('models', [])]
                except Exception:
                    return True, f"Connected to Local Ollama ({self.host}), but the model list response was unexpected."
                return True, f"Connected to Local Ollama ({self.host}). Available models: {', '.join(models) if models else 'None'}"
            return False, f"Local server returned HTTP status {r.status_code}"
        except Exception as e:
            return False, f"Could not connect to local Gemma engine at {self.host}: {str(e)}"

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        url = f"{self.host}/api/chat"
        
        # Attach images for the most recent USER message wherever it sits in
        # history — inside a tool loop the user message is no longer last, but
        # the model still needs the image to synthesize its final answer.
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user_idx = i

        ollama_msgs = []
        for idx, m in enumerate(messages):
            role = m["role"]
            content_text = m.get("content") or ""

            _, img_paths = _extract_image_paths(content_text) if idx == last_user_idx else (content_text, [])

            msg_obj: Dict[str, Any] = {"role": role, "content": content_text}
            if role == "tool" and m.get("name"):
                msg_obj["tool_name"] = m["name"]
            
            # Replay tool calls in Ollama's expected schema
            if "tool_calls" in m and m["tool_calls"]:
                formatted_tc = []
                for tc in m["tool_calls"]:
                    t_name = tc.get("name") or tc.get("tool")
                    t_args = tc.get("arguments") or tc.get("args") or {}
                    formatted_tc.append({
                        "function": {
                            "name": t_name,
                            "arguments": t_args
                        }
                    })
                msg_obj["tool_calls"] = formatted_tc
                
            # If image paths found in latest message, encode into base64 for Ollama vision
            if img_paths:
                b64_images = []
                for ipath in img_paths:
                    try:
                        with open(ipath, "rb") as img_file:
                            b64_images.append(base64.b64encode(img_file.read()).decode("utf-8"))
                    except Exception:
                        pass
                if b64_images:
                    msg_obj["images"] = b64_images
                    
            ollama_msgs.append(msg_obj)

        payload = {
            "model": self.model_name,
            "messages": ollama_msgs,
            "stream": False,
        }
        if tools_schema:
            payload["tools"] = tools_schema

        start_time = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=300)
            elapsed = time.time() - start_time

            if resp.status_code != 200:
                return f"Local Engine Error (HTTP {resp.status_code}): {resp.text}", None, {"duration_sec": elapsed, "backend_label": "Local Ollama"}
            
            data = resp.json()
            message = data.get("message", {})
            content = message.get("content", "")
            
            tool_calls = None
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = []
                for tc in message["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_calls.append({
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments", {})
                    })

            # Extract Ollama token metrics
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            total_duration_ns = data.get("total_duration", 0)
            duration_sec = (total_duration_ns / 1e9) if total_duration_ns > 0 else elapsed

            metrics = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_sec": duration_sec,
                "backend_label": f"Local ({self.model_name})"
            }

            return content, tool_calls, metrics
        except Exception as e:
            elapsed = time.time() - start_time
            return f"Error communicating with local Gemma engine: {str(e)}", None, {"duration_sec": elapsed, "backend_label": "Local Ollama"}
