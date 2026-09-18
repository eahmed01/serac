"""Structured tracing for correlated spans across agent operations.

Provides trace_id/run_id propagation and span emission for:
- Model calls (latency, tokens, cost)
- Tool execution
- Context compaction
- Worker dispatch

Traces are written as structured JSON lines to a file or logger.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Context variable for current trace — propagates across threads via context
_current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)


@dataclass
class Span:
    """A single trace span with timing and metadata."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    kind: str = "internal"  # internal, model, tool, compaction, dispatch
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok, error
    error: Optional[str] = None

    def finish(self) -> None:
        """Mark the span as completed."""
        self.end_time = time.monotonic()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


class Tracer:
    """Structured tracer for agent operations.

    Emits spans to a file or logger. Supports correlated trace_ids
    across provider calls, tool execution, and compaction.

    Args:
        run_id: Optional run identifier. Auto-generated if not provided.
        output_path: Optional file path for JSONL span output.
        enable: Whether tracing is enabled (default True).
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        output_path: Optional[str] = None,
        enable: bool = True,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.output_path = output_path
        self.enable = enable
        self._spans: list[Span] = []
        self._lock = threading.Lock()

        if output_path and enable:
            # Open in append mode — safe for long-running processes
            self._file = open(output_path, "a", buffering=1)  # line-buffered
        else:
            self._file = None

    @property
    def trace_id(self) -> str:
        """Current trace ID or a new one."""
        trace = _current_trace_id.get()
        if not trace:
            trace = str(uuid.uuid4())[:12]
            _current_trace_id.set(trace)
        return trace

    @property
    def span_id(self) -> str:
        """Current span ID or a new one."""
        span = _current_span_id.get()
        if not span:
            span = str(uuid.uuid4())[:8]
            _current_span_id.set(span)
        return span

    def start_span(
        self,
        operation: str,
        kind: str = "internal",
        attributes: Optional[dict[str, Any]] = None,
    ) -> Span:
        """Create and return a new span."""
        if not self.enable:
            return Span(
                trace_id="", span_id="", parent_span_id=None,
                operation=operation, kind=kind,
            )

        parent = _current_span_id.get()
        span_id = str(uuid.uuid4())[:8]
        _current_span_id.set(span_id)

        span = Span(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent,
            operation=operation,
            kind=kind,
            attributes=attributes or {},
        )
        with self._lock:
            self._spans.append(span)
        return span

    def _emit(self, span: Span) -> None:
        """Write span to output."""
        payload = span.to_dict()
        payload["run_id"] = self.run_id

        if self._file:
            self._file.write(json.dumps(payload) + "\n")
        else:
            logger.debug("TRACE %s", json.dumps(payload))

    @contextmanager
    def span(
        self,
        operation: str,
        kind: str = "internal",
        attributes: Optional[dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Context manager for a traced operation.

        Usage:
            with tracer.span("model_call", kind="model", attrs) as s:
                ...
        """
        span = self.start_span(operation, kind, attributes)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = str(exc)
            raise
        finally:
            span.finish()
            self._emit(span)

    def record_model_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a completed model call as a span."""
        if not self.enable:
            return

        span = Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=_current_span_id.get(),
            operation=f"model.{model}",
            kind="model",
            attributes={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "latency_ms": latency_ms,
            },
            status="error" if error else "ok",
            error=error,
        )
        span.finish()
        with self._lock:
            self._spans.append(span)
        self._emit(span)

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a completed tool call as a span."""
        if not self.enable:
            return

        span = Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=_current_span_id.get(),
            operation=f"tool.{tool_name}",
            kind="tool",
            attributes={
                "tool": tool_name,
                "latency_ms": latency_ms,
            },
            status="error" if error else "ok",
            error=error,
        )
        span.finish()
        with self._lock:
            self._spans.append(span)
        self._emit(span)

    def record_compaction(
        self,
        messages_before: int,
        messages_after: int,
        method: str = "summary",
    ) -> None:
        """Record a context compaction as a span."""
        if not self.enable:
            return

        span = Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=_current_span_id.get(),
            operation="compaction",
            kind="compaction",
            attributes={
                "messages_before": messages_before,
                "messages_after": messages_after,
                "method": method,
                "reduction": messages_before - messages_after,
            },
        )
        span.finish()
        with self._lock:
            self._spans.append(span)
        self._emit(span)

    def close(self) -> None:
        """Close the tracer and flush output."""
        if self._file:
            self._file.close()
            self._file = None

    def get_spans(self) -> list[dict[str, Any]]:
        """Return all recorded spans as dicts."""
        with self._lock:
            return [s.to_dict() for s in self._spans]

    def summary(self) -> dict[str, Any]:
        """Return a summary of all spans."""
        with self._lock:
            model_spans = [s for s in self._spans if s.kind == "model"]
            tool_spans = [s for s in self._spans if s.kind == "tool"]

        total_cost = sum(
            s.attributes.get("cost", 0) for s in model_spans
        )
        total_tokens = sum(
            s.attributes.get("prompt_tokens", 0) + s.attributes.get("completion_tokens", 0)
            for s in model_spans
        )

        with self._lock:
            total_spans = len(self._spans)
            errors = sum(1 for s in self._spans if s.status == "error")

        return {
            "run_id": self.run_id,
            "total_spans": total_spans,
            "model_calls": len(model_spans),
            "tool_calls": len(tool_spans),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Thread-safe propagation helpers
# ---------------------------------------------------------------------------


def set_trace_id(trace_id: str) -> str:
    """Set the current trace ID and return it."""
    _current_trace_id.set(trace_id)
    _current_span_id.set(None)  # Reset span when setting new trace
    return trace_id


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return _current_trace_id.get()


def clear_trace() -> None:
    """Clear the current trace context."""
    _current_trace_id.set(None)
    _current_span_id.set(None)
