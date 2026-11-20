"""Local JSON storage for user feedback, custom clinic phrases, and saved draft notes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_STORAGE_FILENAME = ".medical_notes_data.json"


def _empty_state() -> Dict[str, Any]:
    return {
        "feedback": {},
        "custom_phrases": [],
        "drafts": [],
    }


class NotesStorage:
    """Persists user feedback on suggestions, custom phrases, and drafted notes."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is not None:
            self.path = Path(path)
        else:
            env_path = os.getenv("NOTES_STORAGE_PATH", DEFAULT_STORAGE_FILENAME)
            self.path = Path(env_path)
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return _empty_state()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            state = _empty_state()
            if isinstance(raw, dict):
                state["feedback"] = raw.get("feedback", {})
                state["custom_phrases"] = raw.get("custom_phrases", [])
                state["drafts"] = raw.get("drafts", [])
            return state
        except (json.JSONDecodeError, OSError):
            return _empty_state()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_key(context_tokens: List[str]) -> str:
        # Use up to the last 2 context tokens as the feedback lookup key
        trimmed = [t.strip().lower() for t in context_tokens if t.strip()][-2:]
        return " ".join(trimmed) if trimmed else "<s>"

    def record_feedback(self, context_tokens: List[str], word: str, accepted: bool) -> Dict[str, int]:
        """Record positive or negative feedback for a suggested word in a given context."""
        key = self._normalize_key(context_tokens)
        token = word.strip().lower()
        ctx_bucket = self._state["feedback"].setdefault(key, {})
        entry = ctx_bucket.setdefault(token, {"accepted": 0, "rejected": 0})
        if accepted:
            entry["accepted"] += 1
        else:
            entry["rejected"] += 1
        self._save()
        return dict(entry)

    def get_feedback_boosts(self, context_tokens: List[str]) -> Dict[str, float]:
        """Return multiplicative probability weights derived from user feedback.

        Accepted suggestions receive a boost (> 1.0); rejected ones are penalized (< 1.0).
        """
        boosts: Dict[str, float] = {}
        # Check exact 2-word context first, then 1-word backoff context
        keys = []
        norm_full = self._normalize_key(context_tokens)
        keys.append(norm_full)
        if len(context_tokens) >= 1:
            norm_last = self._normalize_key(context_tokens[-1:])
            if norm_last != norm_full:
                keys.append(norm_last)

        for weight_scale, key in zip((1.0, 0.5), keys):
            ctx_bucket = self._state["feedback"].get(key, {})
            for word, counts in ctx_bucket.items():
                accepted = int(counts.get("accepted", 0))
                rejected = int(counts.get("rejected", 0))
                net = (accepted - rejected) * weight_scale
                if net > 0:
                    factor = 1.0 + min(net * 0.65, 4.0)
                elif net < 0:
                    factor = max(0.1, 1.0 / (1.0 + abs(net) * 1.25))
                else:
                    factor = 1.0
                # Exact context overrides backoff
                if word not in boosts:
                    boosts[word] = factor
        return boosts

    def add_custom_phrase(self, phrase: str) -> bool:
        """Store a custom clinic phrase to enrich future model suggestions."""
        cleaned = phrase.strip()
        if not cleaned:
            return False
        if cleaned not in self._state["custom_phrases"]:
            self._state["custom_phrases"].append(cleaned)
            self._save()
            return True
        return False

    def get_custom_phrases(self) -> List[str]:
        """Return all saved custom clinic phrases."""
        return list(self._state["custom_phrases"])

    def save_draft(self, title: str, content: str) -> Dict[str, str]:
        """Persist a drafted administrative medical note."""
        draft = {
            "title": title.strip() or "Untitled Administrative Note",
            "content": content.strip(),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._state["drafts"].append(draft)
        self._save()
        return draft

    def list_drafts(self) -> List[Dict[str, str]]:
        """Return all saved draft notes."""
        return list(self._state["drafts"])

    def feedback_summary(self) -> Dict[str, int]:
        """Return aggregate counts of recorded feedback events."""
        accepted = 0
        rejected = 0
        contexts = len(self._state["feedback"])
        for bucket in self._state["feedback"].values():
            for counts in bucket.values():
                accepted += int(counts.get("accepted", 0))
                rejected += int(counts.get("rejected", 0))
        return {
            "contexts": contexts,
            "accepted": accepted,
            "rejected": rejected,
            "custom_phrases": len(self._state["custom_phrases"]),
            "drafts": len(self._state["drafts"]),
        }
