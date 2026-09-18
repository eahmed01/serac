"""Tool registry — definition, schema generation, and execution.

Tools are Python callables registered with JSON Schema parameters.
The registry produces OpenAI/Anthropic-compatible tool schemas and
executes tool calls concurrently.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_TOOL_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def _validate_tool_name(name: str) -> None:
    """Validate a tool name against the provider-compatible naming convention."""
    if not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            "Invalid tool name {!r}: must start with an ASCII letter, contain "
            "only ASCII letters, digits, underscores, or hyphens, and be at "
            "most 64 characters".format(name)
        )


@dataclass(frozen=True)
class ToolDef:
    """Definition of a single tool.

    Args:
        name: Unique tool name (e.g., "memory_search").
        description: Human-readable description for the model.
        parameters: JSON Schema object describing the tool's input.
        executor: Python callable that implements the tool.
        role: Optional role filter. When set, the tool is only visible
            to agents declaring this role via ``ToolRegistry.for_role()``.
        execution_mode: Where the tool executes. ``"sandbox"`` means the
            executor is safe to load with a sandbox; ``"host"`` is the
            conservative default for tools that execute on the host.
        requires_sandbox: Whether the tool may only be admitted when a
            sandbox is configured. This is distinct from ``execution_mode``:
            guarded host fallbacks can remain available without a sandbox.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    executor: Callable[..., Any]
    role: Optional[str] = None
    execution_mode: str = "host"
    requires_sandbox: bool = False

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if not isinstance(self.execution_mode, str) or self.execution_mode not in {"sandbox", "host"}:
            raise ValueError(
                f"Invalid execution_mode {self.execution_mode!r}: "
                "must be 'sandbox' or 'host'"
            )
        if not isinstance(self.requires_sandbox, bool):
            raise ValueError(
                f"Invalid requires_sandbox {self.requires_sandbox!r}: must be a boolean"
            )


class ToolRegistry:
    """Registry of tools with schema generation and execution.

    Maintains a single collection of ``ToolDef`` objects. Provides
    OpenAI-format tool schemas, role-filtered views, and concurrent
    execution of tool calls.

    Supports pluggable tool name resolution via ``ToolResolver``:
    when a model calls a tool by a name that doesn't match exactly,
    the resolver tries aliases, fuzzy matching, then LLM resolution.
    """

    _shared_pool: Optional["ToolRegistry"] = None

    def __init__(self, resolver=None) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._resolver = resolver
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = Lock()

    def register(self, tool: ToolDef) -> None:
        """Register a tool definition.

        Args:
            tool: Tool definition to register.

        Raises:
            ValueError: If the tool name is invalid or a tool with the same name is already registered.
        """
        # Validate at the registry boundary as well as ToolDef construction;
        # callers can otherwise bypass dataclass initialization.
        _validate_tool_name(tool.name)
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    @property
    def tools(self) -> dict[str, ToolDef]:
        """Read-only mapping of tool name → ToolDef."""
        return dict(self._tools)

    @property
    def schema(self) -> list[dict[str, Any]]:
        """OpenAI-format tool schema for all registered tools.

        Returns:
            List of tool schema dicts compatible with OpenAI/Anthropic APIs.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
            for td in self._tools.values()
        ]

    def for_role(self, role_name: str) -> ToolRegistry:
        """Return a filtered view containing only tools for the given role.

        Tools with ``role=None`` are included in every role filter.

        Args:
            role_name: Role name to filter by.

        Returns:
            New ToolRegistry with filtered tools (and same resolver).
        """
        filtered = ToolRegistry(resolver=self._resolver)
        for td in self._tools.values():
            if td.role is None or td.role == role_name:
                filtered.register(td)
        return filtered

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
        context: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a list of tool calls, returning results.

        Tool calls with matching names are dispatched to their executors.
        Multiple calls are executed concurrently using a thread pool.

        Args:
            tool_calls: List of tool call dicts with ``name`` and ``arguments``.
            context: Optional conversation history passed as ``context`` kwarg.

        Returns:
            List of result dicts with ``role``, ``content``, and
            ``tool_call_id`` keys.
        """
        if not tool_calls:
            return []

        def _run_one(tc: dict[str, Any]) -> dict[str, Any]:
            # Support both OpenAI format (function.name) and flat format (name)
            if "function" in tc:
                name = tc["function"].get("name", "unknown")
                args_raw = tc["function"].get("arguments", "{}")
            else:
                name = tc.get("name", "unknown")
                args_raw = tc.get("arguments", {})
            tool_id = tc.get("id", tc.get("tool_call_id", ""))

            # Parse arguments (may be raw JSON string or already a dict)
            if isinstance(args_raw, str):
                if not args_raw.strip():
                    args = {}  # Genuine empty — safe
                else:
                    try:
                        import json as _json
                        args = _json.loads(args_raw)
                    except (ValueError, TypeError) as exc:
                        # Return structured error instead of running with empty args
                        return {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": f"Error: tool '{name}' received invalid JSON arguments: {args_raw[:100]}",
                            "_tool_name": name,
                        }
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}

            if name not in self._tools:
                # Try resolver if available
                if self._resolver:
                    resolved = self._resolver.resolve(name)
                    if resolved and resolved in self._tools:
                        logger.info("Resolved '%s' -> '%s'", name, resolved)
                        name = resolved
                    else:
                        logger.warning("Unknown tool called: %s", name)
                        return {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": f"Error: tool '{name}' not found",
                            "_tool_name": name,
                        }
                else:
                    logger.warning("Unknown tool called: %s", name)
                    return {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": f"Error: tool '{name}' not found",
                        "_tool_name": name,
                    }

            executor = self._tools[name].executor
            required = self._tools[name].parameters.get("required", [])
            missing = [key for key in required if key not in args]
            if missing:
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: tool '{name}' missing required arguments: {', '.join(missing)}",
                    "_tool_name": name,
                    "_error": "missing_required_arguments",
                }
            try:
                kwargs = dict(args)
                if context is not None:
                    kwargs["context"] = context
                result = executor(**kwargs)
                content = str(result) if result is not None else ""
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content,
                    "_tool_name": name,
                }
            except Exception as exc:
                logger.exception("Tool execution failed: %s", name)
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error executing '{name}': {exc}",
                    "_tool_name": name,
                }

        # P2 fix: Use a shared thread pool instead of creating one per call
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=8)

        results = list(self._pool.map(_run_one, tool_calls))
        return results

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
