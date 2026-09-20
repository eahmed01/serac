"""Session management - persistent conversation storage and recall.

The problem: context window management compacts conversation history into summaries,
but the full conversation is lost. This module solves this by:

1. Persisting the full conversation to disk after each turn
2. Storing compacted summaries separately for the LLM context window
3. Enabling search/recall of the full conversation history post-compaction

This allows the agent to:
- Resume sessions from any point in time
- Search past conversations for specific details
- Maintain a searchable knowledge base of all interactions
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
        raise ValueError("Invalid session ID")


@dataclass
class SessionMessage:
    """A single message in a conversation session."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = 0.0
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", 0.0),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata"),
        )


@dataclass
class SessionMetadata:
    """Metadata for a session."""
    session_id: str
    title: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    compaction_points: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    total_tokens: int = 0
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "compaction_points": self.compaction_points,
            "tags": self.tags,
            "model": self.model,
            "provider": self.provider,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMetadata":
        return cls(
            session_id=data["session_id"],
            title=data.get("title", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            compaction_points=data.get("compaction_points", []),
            tags=data.get("tags", []),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
        )


class SessionStore:
    """Persistent session storage with search capabilities.

    Stores full conversation history to disk, enabling:
    - Resume from any point in time
    - Search/recall past conversations
    - Track compaction points for context window management

    Args:
        store_path: Directory to store sessions (default: ~/.serac/sessions)
    """

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path or os.path.expanduser("~/.serac/sessions"))
        self.store_path.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        _validate_session_id(session_id)
        return self.store_path / f"{session_id}.json"

    def create_session(
        self,
        session_id: Optional[str] = None,
        title: str = "",
        model: str = "",
        provider: str = "",
    ) -> SessionMetadata:
        """Create a new session."""
        import uuid

        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        _validate_session_id(session_id)

        now = time.time()
        metadata = SessionMetadata(
            session_id=session_id,
            title=title or f"Session {now}",
            created_at=now,
            updated_at=now,
            model=model,
            provider=provider,
        )

        self._save_session(session_id, [], metadata)
        logger.info("Created session %s", session_id)
        return metadata

    def append_message(self, session_id: str, message: SessionMessage) -> None:
        """Append a message to a session."""
        messages, metadata = self._load_session(session_id)
        messages.append(message)
        metadata.updated_at = time.time()
        self._save_session(session_id, messages, metadata)

    def get_messages(
        self,
        session_id: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[SessionMessage]:
        """Get messages from a session, optionally with pagination."""
        messages, _ = self._load_session(session_id)
        if start is not None and end is not None:
            return messages[start:end]
        elif start is not None:
            return messages[start:]
        elif end is not None:
            return messages[:end]
        return messages

    def search_messages(
        self,
        session_id: str,
        query: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Search messages in a session for a query string."""
        messages, _ = self._load_session(session_id)
        results = []
        for i, msg in enumerate(messages):
            if query.lower() in msg.content.lower():
                results.append({
                    "index": i,
                    "message": msg,
                    "timestamp": msg.timestamp,
                })
                if len(results) >= max_results:
                    break
        return results

    def record_compaction(
        self,
        session_id: str,
        summary: str,
        messages_before: int,
        messages_after: int,
        method: str = "summary",
    ) -> None:
        """Record a compaction point for context window management."""
        messages, metadata = self._load_session(session_id)
        metadata.compaction_points.append({
            "timestamp": time.time(),
            "summary": summary,
            "messages_before": messages_before,
            "messages_after": messages_after,
            "method": method,
        })
        metadata.updated_at = time.time()
        self._save_session(session_id, messages, metadata)

    def get_compaction_points(self, session_id: str) -> list[dict]:
        """Get compaction points for a session."""
        _, metadata = self._load_session(session_id)
        return metadata.compaction_points

    def list_sessions(self, limit: int = 20) -> list[SessionMetadata]:
        """List all sessions, sorted by updated_at (newest first)."""
        sessions = []
        for path in self.store_path.glob("*.json"):
            try:
                _, metadata = self._load_session(path.stem)
                sessions.append(metadata)
            except Exception:
                logger.exception("Failed to load session %s", path.stem)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    def _load_session(self, session_id: str) -> tuple[list[SessionMessage], SessionMetadata]:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        with open(path, "r") as f:
            data = json.load(f)
        messages = [SessionMessage.from_dict(m) for m in data.get("messages", [])]
        metadata = SessionMetadata.from_dict(data.get("metadata", {}))
        return messages, metadata

    def _save_session(
        self,
        session_id: str,
        messages: list[SessionMessage],
        metadata: SessionMetadata,
    ) -> None:
        """Save a session to disk."""
        data = {
            "messages": [m.to_dict() for m in messages],
            "metadata": metadata.to_dict(),
        }
        path = self._session_path(session_id)
        # Atomic write
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp_path), str(path))