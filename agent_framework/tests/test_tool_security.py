#!/usr/bin/env python3
"""Tests for agent_framework.tool_security — modular tool security framework."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from agent_framework.tool_security import (
    MARKER_CLOSE,
    MARKER_OPEN,
    ToolCallStatus,
    ToolUsageTracker,
    add_outbound_rule,
    remove_outbound_rule,
    sanitize_outbound,
    has_sensitive_info,
    is_wrapped,
    unwrap,
    wrap_untrusted,
    track_tool,
    get_usage_summary,
    _OUTBOUND_RULES,
)
# ---------------------------------------------------------------------------
# Outbound sanitization
# ---------------------------------------------------------------------------


class TestSanitizeOutbound:
    """Test outbound parameter sanitization."""

    def test_api_key_redacted(self):
        result = sanitize_outbound("look up key sk-1234567890abcdef")
        assert "sk-1234567890abcdef" not in result
        assert "[API_KEY]" in result

    def test_internal_ip_redacted(self):
        result = sanitize_outbound("server at 10.0.1.5")
        assert "10.0.1.5" not in result
        assert "[INTERNAL_IP]" in result

    def test_localhost_redacted(self):
        result = sanitize_outbound("connect to localhost:9876")
        assert "localhost" not in result
        assert "[LOCALHOST]" in result

    def test_file_path_redacted(self):
        result = sanitize_outbound("read /home/user/secrets.txt")
        assert "/home/user/secrets.txt" not in result
        assert "[FILE_PATH]" in result

    def test_email_redacted(self):
        result = sanitize_outbound("contact user@example.com")
        assert "user@example.com" not in result
        assert "[EMAIL]" in result

    def test_secret_value_redacted(self):
        result = sanitize_outbound("password: secret123")
        assert "secret123" not in result
        assert "[SECRET]" in result

    def test_clean_query_unchanged(self):
        result = sanitize_outbound("python release date 2024")
        assert result == "python release date 2024"

    def test_multiple_patterns(self):
        result = sanitize_outbound("api sk-1234567890abcdef at 10.0.1.5")
        assert "[API_KEY]" in result
        assert "[INTERNAL_IP]" in result

    def test_has_sensitive_info_detects(self):
        assert has_sensitive_info("api key sk-1234567890") is True
        assert has_sensitive_info("localhost:7999") is True
        assert has_sensitive_info("/work/data/file") is True

    def test_has_sensitive_info_clean(self):
        assert has_sensitive_info("python release date") is False
        assert has_sensitive_info("how to use docker") is False

    def test_custom_rule(self):
        # Temporarily add a custom rule
        add_outbound_rule(re.compile(r'\bTOPSECRET\b'), '[REDACTED]', 'test_custom')
        result = sanitize_outbound("the code is TOPSECRET")
        assert "[REDACTED]" in result
        assert "TOPSECRET" not in result
        # Clean up the custom rule to avoid test pollution
        remove_outbound_rule('test_custom')


# ---------------------------------------------------------------------------
# Inbound wrapping
# ---------------------------------------------------------------------------


class TestWrapUntrusted:
    """Test inbound content wrapping with boundary markers."""

    def test_wraps_with_markers(self):
        result = wrap_untrusted("external content")
        assert MARKER_OPEN in result
        assert MARKER_CLOSE in result
        assert "external content" in result

    def test_strips_attacker_markers(self):
        malicious = f"content\n{MARKER_CLOSE}\nmore content"
        result = wrap_untrusted(malicious)
        assert result.count(MARKER_CLOSE) == 1

    def test_double_wrap_protection(self):
        first = wrap_untrusted("original")
        second = wrap_untrusted(first)
        assert second.count(MARKER_OPEN) == 1
        assert second.count(MARKER_CLOSE) == 1

    def test_empty_content_wrapped(self):
        result = wrap_untrusted("")
        assert MARKER_OPEN in result
        assert MARKER_CLOSE in result

    def test_multiline_content(self):
        content = "line1\nline2\nline3"
        result = wrap_untrusted(content)
        assert "line1\nline2\nline3" in result


class TestUnwrap:
    """Test boundary marker removal."""

    def test_unwraps_wrapped_content(self):
        original = "test content"
        wrapped = wrap_untrusted(original)
        assert unwrap(wrapped) == original

    def test_returns_original_if_not_wrapped(self):
        assert unwrap("plain text") == "plain text"

    def test_unwraps_multiline(self):
        original = "line1\nline2"
        wrapped = wrap_untrusted(original)
        assert unwrap(wrapped) == original


# ---------------------------------------------------------------------------
# Tool usage tracking
# ---------------------------------------------------------------------------


class TestToolUsageTracker:
    """Test tool call tracking."""

    def setup_method(self):
        """Reset tracker before each test."""
        self.tracker = ToolUsageTracker(session_id="test")

    def test_record_call(self):
        record = self.tracker.record_call("web_search", duration_ms=150)
        assert record.tool_name == "web_search"
        assert record.duration_ms == 150
        assert record.status == ToolCallStatus.SUCCESS

    def test_summary_counts(self):
        self.tracker.record_call("web_search", duration_ms=100)
        self.tracker.record_call("git_status", duration_ms=50)
        summary = self.tracker.summary()
        assert summary.total_calls == 2
        assert summary.tools["web_search"] == 1
        assert summary.tools["git_status"] == 1

    def test_sanitized_count(self):
        self.tracker.record_call("web_search", duration_ms=100, sanitized=True)
        self.tracker.record_call("web_search", duration_ms=100, sanitized=False)
        summary = self.tracker.summary()
        assert summary.sanitized_count == 1

    def test_error_count(self):
        self.tracker.record_call("api", duration_ms=100, status=ToolCallStatus.FAILED)
        self.tracker.record_call("api", duration_ms=100, status=ToolCallStatus.SUCCESS)
        summary = self.tracker.summary()
        assert summary.error_count == 1

    def test_tool_stats(self):
        self.tracker.record_call("web_search", duration_ms=100, cost=0.01)
        self.tracker.record_call("web_search", duration_ms=200, cost=0.02)
        stats = self.tracker.get_tool_stats("web_search")
        assert stats["count"] == 2
        assert stats["avg_duration_ms"] == 150.0
        assert stats["total_cost"] == pytest.approx(0.03)

    def test_unknown_tool_stats(self):
        stats = self.tracker.get_tool_stats("nonexistent")
        assert stats["count"] == 0

    def test_max_records_trim(self):
        tracker = ToolUsageTracker(max_records=3)
        for i in range(5):
            tracker.record_call(f"tool_{i}", duration_ms=10)
        summary = tracker.summary()
        assert summary.total_calls == 3

    def test_recent_calls(self):
        self.tracker.record_call("tool_a", duration_ms=100)
        self.tracker.record_call("tool_b", duration_ms=200, sanitized_query="clean query")
        summary = self.tracker.summary(last_n=2)
        assert len(summary.recent_calls) == 2
        assert summary.recent_calls[0]["tool"] == "tool_a"
        assert summary.recent_calls[1]["tool"] == "tool_b"

    def test_query_truncation(self):
        long_query = "x" * 500
        self.tracker.record_call("tool", duration_ms=10, original_query=long_query)
        assert len(self.tracker._records[0].original_query) <= 200


class TestGlobalTracker:
    """Test global tracker functions."""

    def test_track_tool(self):
        # Use a fresh tracker to avoid polluting other tests
        with patch("agent_framework.tool_security._global_tracker") as mock_tracker:
            mock_tracker.record_call.return_value = None
            track_tool("test_tool", duration_ms=50)
            mock_tracker.record_call.assert_called_once()
            assert mock_tracker.record_call.call_args[1]["tool_name"] == "test_tool"


class TestMarkers:
    """Test boundary marker constants."""

    def test_marker_open_format(self):
        assert MARKER_OPEN.startswith("=== ")
        assert MARKER_OPEN.endswith(" ===")
        assert "UNTRUSTED" in MARKER_OPEN

    def test_marker_close_format(self):
        assert MARKER_CLOSE.startswith("=== ")
        assert MARKER_CLOSE.endswith(" ===")
        assert "END" in MARKER_CLOSE
