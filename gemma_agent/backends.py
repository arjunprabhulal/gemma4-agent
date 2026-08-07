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
    Scan content text for image file paths and extract valid local images for vision input.

    Args:
        content (str): Raw input prompt content text.

    Returns:
        Tuple[str, List[str]]: (Cleaned text without image paths, List of absolute image paths).
    """
    image_paths = []
    tokens = content.split()
    clean_tokens = []
    
    for t in tokens:
        clean_t = t.strip("\"'")
        if os.path.exists(os.path.expanduser(clean_t)) and clean_t.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
            image_paths.append(os.path.expanduser(clean_t))
        else:
            clean_tokens.append(t)
            
    return " ".join(clean_tokens), image_paths


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
                models = [m['name'] for m in r.json().get('models', [])]
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
        
        ollama_msgs = []
        for idx, m in enumerate(messages):
            role = m["role"]
            content_text = m.get("content") or ""
            
            # Only scan image paths for the newest user message to prevent re-encoding historic images
            is_latest_user = (role == "user" and idx == len(messages) - 1)
            clean_text, img_paths = _extract_image_paths(content_text) if is_latest_user else (content_text, [])
            
            msg_obj: Dict[str, Any] = {"role": role, "content": clean_text or content_text}
            
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
