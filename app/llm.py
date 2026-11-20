"""Optional LLM helper for polishing drafted administrative medical notes.

The core n-gram next-word predictor works 100% locally without any API keys.
This module provides an optional hook for users who configure Ollama, Groq, or Gemini in .env.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


def get_llm_provider() -> str:
    """Return the configured optional LLM provider name."""
    return os.getenv("LLM_PROVIDER", "none").strip().lower()


def is_llm_configured() -> bool:
    """Return True if an optional LLM provider has valid configuration."""
    provider = get_llm_provider()
    if provider == "ollama":
        return bool(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY", "").strip())
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY", "").strip())
    return False


def polish_note(note_text: str, timeout: float = 10.0) -> Optional[str]:
    """Optionally polish grammar/clarity of an administrative note using Ollama if enabled.

    Returns ``None`` if no LLM provider is configured or if the optional endpoint is unreachable.
    """
    provider = get_llm_provider()
    if provider != "ollama":
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    payload = {
        "model": model,
        "prompt": (
            "Rewrite the following administrative medical note for clarity and consistency. "
            "Do not invent clinical facts or patient identifiers. Return only the revised note:\n\n"
            f"{note_text}"
        ),
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response = data.get("response", "").strip()
            return response or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
