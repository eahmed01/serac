"""Tests for session management - persistent conversation storage and recall."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from agent_framework.session import SessionMessage, SessionStore


def test_invalid_session_ids_are_confined(tmp_path):
    store = SessionStore(str(tmp_path))
    for session_id in ("../escape", "nested/id", "session.json", "x" * 65):
        with pytest.raises(ValueError, match="Invalid session ID"):
            store.create_session(session_id)
        with pytest.raises(ValueError, match="Invalid session ID"):
            store.get_messages(session_id)


def test_create_session():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        meta = store.create_session(
            session_id="test123",
            title="Test Session",
            model="qwen",
            provider="vllm",
        )
        assert meta.session_id == "test123"
        assert meta.title == "Test Session"
        assert meta.model == "qwen"


def test_append_and_get_messages():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        store.create_session("test123")

        store.append_message("test123", SessionMessage(
            role="user",
            content="Hello",
            timestamp=time.time(),
        ))
        store.append_message("test123", SessionMessage(
            role="assistant",
            content="Hi there!",
            timestamp=time.time(),
        ))

        messages = store.get_messages("test123")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"


def test_search_messages():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        store.create_session("test123")

        store.append_message("test123", SessionMessage(
            role="user",
            content="What is the capital of France?",
            timestamp=time.time(),
        ))
        store.append_message("test123", SessionMessage(
            role="assistant",
            content="The capital of France is Paris.",
            timestamp=time.time(),
        ))

        results = store.search_messages("test123", "Paris")
        assert len(results) >= 1
        assert "Paris" in results[0]["message"].content


def test_compaction_points():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        store.create_session("test123")

        store.record_compaction(
            session_id="test123",
            summary="User asked about France, assistant answered Paris.",
            messages_before=10,
            messages_after=3,
            method="summary",
        )

        points = store.get_compaction_points("test123")
        assert len(points) == 1
        assert points[0]["messages_before"] == 10
        assert points[0]["messages_after"] == 3


def test_list_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        store.create_session("test1", title="First")
        time.sleep(0.01)
        store.create_session("test2", title="Second")

        sessions = store.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].title == "Second"  # Newest first


def test_pagination():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        store.create_session("test123")

        for i in range(10):
            store.append_message("test123", SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                timestamp=time.time(),
            ))

        # Get first 3
        messages = store.get_messages("test123", start=0, end=3)
        assert len(messages) == 3

        # Get last 2
        messages = store.get_messages("test123", start=-2)
        assert len(messages) == 2