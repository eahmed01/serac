"""Tests for agent_framework.tracing module."""

import json
import tempfile
from unittest.mock import patch

import pytest

from agent_framework.tracing import (
    Tracer,
    Span,
    set_trace_id,
    get_trace_id,
    clear_trace,
    _current_trace_id,
)


class TestSpan:
    """Test Span dataclass."""

    def test_basic_span(self):
        span = Span(
            trace_id="abc123",
            span_id="span01",
            parent_span_id=None,
            operation="model_call",
        )
        assert span.trace_id == "abc123"
        assert span.kind == "internal"
        assert span.status == "ok"

    def test_span_finish(self):
        span = Span(
            trace_id="abc123",
            span_id="span01",
            parent_span_id=None,
            operation="model_call",
        )
        span.finish()
        assert span.duration_ms is not None
        assert span.end_time is not None

    def test_span_to_dict(self):
        span = Span(
            trace_id="abc123",
            span_id="span01",
            parent_span_id=None,
            operation="model_call",
            kind="model",
        )
        span.finish()
        d = span.to_dict()
        assert d["trace_id"] == "abc123"
        assert d["kind"] == "model"
        assert "duration_ms" in d

    def test_span_to_json(self):
        span = Span(
            trace_id="abc123",
            span_id="span01",
            parent_span_id=None,
            operation="model_call",
        )
        span.finish()
        j = span.to_json()
        parsed = json.loads(j)
        assert parsed["trace_id"] == "abc123"

    def test_span_error(self):
        span = Span(
            trace_id="abc123",
            span_id="span01",
            parent_span_id=None,
            operation="model_call",
            status="error",
            error="connection refused",
        )
        assert span.status == "error"
        assert span.error == "connection refused"


class TestTracer:
    """Test Tracer class."""

    def test_basic_tracer(self):
        tracer = Tracer(enable=True)
        assert tracer.run_id is not None
        assert tracer.enable is True

    def test_tracer_disabled(self):
        tracer = Tracer(enable=False)
        assert tracer.enable is False

    def test_custom_run_id(self):
        tracer = Tracer(run_id="test-run-123")
        assert tracer.run_id == "test-run-123"

    def test_span_context_manager(self):
        tracer = Tracer(enable=True)
        with tracer.span("test_op", kind="model") as span:
            assert span.trace_id is not None
            assert span.span_id is not None
        assert span.duration_ms is not None

    def test_span_context_manager_captures_error(self):
        tracer = Tracer(enable=True)
        with pytest.raises(ValueError):
            with tracer.span("failing_op", kind="tool"):
                raise ValueError("test error")

    def test_record_model_call(self):
        tracer = Tracer(enable=True)
        tracer.record_model_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            latency_ms=500.0,
        )
        spans = tracer.get_spans()
        assert len(spans) == 1
        assert spans[0]["operation"] == "model.gpt-4"
        assert spans[0]["attributes"]["prompt_tokens"] == 100

    def test_record_tool_call(self):
        tracer = Tracer(enable=True)
        tracer.record_tool_call(
            tool_name="read_file",
            latency_ms=10.0,
        )
        spans = tracer.get_spans()
        assert len(spans) == 1
        assert spans[0]["operation"] == "tool.read_file"

    def test_record_compaction(self):
        tracer = Tracer(enable=True)
        tracer.record_compaction(
            messages_before=100,
            messages_after=10,
            method="summary",
        )
        spans = tracer.get_spans()
        assert len(spans) == 1
        assert spans[0]["operation"] == "compaction"
        assert spans[0]["attributes"]["reduction"] == 90

    def test_summary(self):
        tracer = Tracer(enable=True)
        tracer.record_model_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            latency_ms=500.0,
        )
        tracer.record_tool_call(tool_name="read_file", latency_ms=10.0)

        summary = tracer.summary()
        assert summary["total_spans"] == 2
        assert summary["model_calls"] == 1
        assert summary["tool_calls"] == 1
        assert summary["total_tokens"] == 150
        assert summary["total_cost"] == 0.01

    def test_tracer_disabled_no_spans(self):
        tracer = Tracer(enable=False)
        tracer.record_model_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            latency_ms=500.0,
        )
        assert len(tracer.get_spans()) == 0

    def test_file_output(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tracer = Tracer(enable=True, output_path=f.name)
            tracer.record_model_call(
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                cost=0.01,
                latency_ms=500.0,
            )
            tracer.close()

            with open(f.name) as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["run_id"] == tracer.run_id
            assert parsed["operation"] == "model.gpt-4"

    def test_trace_id_propagation(self):
        tracer = Tracer(enable=True)
        trace_id = tracer.trace_id
        assert trace_id is not None
        # Second call should return same trace_id
        assert tracer.trace_id == trace_id


class TestTraceContext:
    """Test trace context management."""

    def test_set_get_trace_id(self):
        clear_trace()
        trace_id = set_trace_id("custom-trace")
        assert get_trace_id() == "custom-trace"

    def test_clear_trace(self):
        clear_trace()
        assert get_trace_id() is None

    def test_trace_id_context_var(self):
        clear_trace()
        trace_id = set_trace_id("test-trace")
        assert _current_trace_id.get() == "test-trace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
