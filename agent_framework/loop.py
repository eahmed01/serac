"""Agent message loop — the core orchestration engine.

Handles model calls, tool dispatch, result coalescing, steering injection,
and autonomous context compaction.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
from typing import Any, Optional

from agent_framework.providers import ChatResponse, Provider, UsageStats
from agent_framework.tools import ToolRegistry
from agent_framework.tracing import Tracer

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""
    provider: Provider
    system_prompt: str
    tool_registry: ToolRegistry
    max_turns: int = 50
    max_context_tokens: int = 128_000
    coalesce_results: bool = True
    compaction_threshold: float = 0.75
    steering_queue: Optional[Queue] = None
    name: str = "agent"
    max_total_tokens: int = 0              # 0 = unlimited
    max_wall_clock_seconds: float = 0.0   # 0 = unlimited
    tracer: Optional[Tracer] = None       # Structured tracing (optional)


class AgentLoop:
    """Main agent message loop.

    Orchestrates the conversation: model call → tool dispatch → repeat.
    Supports mid-turn steering, parallel tool coalescing, and self-compaction.

    Args:
        provider: LLM backend (VLLM, Anthropic, OpenAI).
        system_prompt: System message for the agent.
        tool_registry: Registered tools available to the agent.
        max_turns: Maximum loop iterations before force-stop.
        max_context_tokens: Token budget for context window.
        coalesce_results: Combine parallel tool results into one message.
        finalize_empty_tool_response: Make one bounded tool-free call after empty post-tool output.
        finalization_prompt: Optional caller-specific instruction for that final call.
        compaction_threshold: Fraction of context window at which to compact.
        steering_queue: Queue for mid-turn steering messages.
        name: Agent name for logging.
        max_total_tokens: Token budget limit (0 = unlimited).
        max_wall_clock_seconds: Wall clock timeout (0 = unlimited).
        tracer: Optional Tracer instance for structured tracing.
    """

    def __init__(
        self,
        provider: Provider,
        system_prompt: str,
        tool_registry: ToolRegistry,
        max_turns: int = 50,
        max_context_tokens: int = 128_000,
        coalesce_results: bool = True,
        finalize_empty_tool_response: bool = False,
        finalization_prompt: Optional[str] = None,
        compaction_threshold: float = 0.75,
        steering_queue: Optional[Queue] = None,
        name: str = "agent",
        max_total_tokens: int = 0,
        max_wall_clock_seconds: float = 0.0,
        tracer: Optional[Tracer] = None,
        session_store: Optional["SessionStore"] = None,  # type: ignore[name-defined]
        session_id: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        self.coalesce_results = coalesce_results
        self.finalize_empty_tool_response = finalize_empty_tool_response
        self.finalization_prompt = finalization_prompt or (
            "The read-only tool results are complete. Return the final answer now "
            "as content only in the requested format. Use only the information "
            "already in this conversation. Do not call tools."
        )
        self.compaction_threshold = compaction_threshold
        self.steering_queue = steering_queue
        self.name = name
        self.max_total_tokens = max_total_tokens
        self.max_wall_clock_seconds = max_wall_clock_seconds
        self.tracer = tracer
        self.session_store = session_store
        self.session_id = session_id

        # State
        self.messages: list[dict[str, Any]] = []
        self.total_usage = UsageStats()
        self.turn_count = 0
        self.finalization_attempted = False
        self.last_response_rejected = False
        self.last_rejection_reason: Optional[str] = None
        self._run_id: Optional[str] = None  # Set when run() starts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Run the agent loop starting from a user message.

        Args:
            user_message: The initial user prompt.

        Returns:
            The final text response from the agent.
        """
        self.reset()
        reset_conversation = getattr(self.provider, "reset_conversation", None)
        if callable(reset_conversation):
            reset_conversation()
        start_time = time.monotonic()

        # P4.2: Initialize tracing
        if self.tracer:
            self._run_id = self.tracer.run_id
            from agent_framework.tracing import set_trace_id, clear_trace
            set_trace_id(self._run_id)
            try:
                return self._run_loop(user_message, start_time)
            finally:
                clear_trace()
        else:
            return self._run_loop(user_message, start_time)

    def _run_loop(self, user_message: str, start_time: float) -> str:
        """Internal loop implementation."""
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        final_text = ""
        tools_executed = False
        response = None

        for self.turn_count in range(1, self.max_turns + 1):
            # Check for mid-turn steering
            steering = self._check_steering()
            if steering:
                self._inject_steering(steering, self.messages)

            # Compact if needed
            self._maybe_compact()

            # Model call
            tools = self.tool_registry.schema if self.tool_registry.tools else None

            # P4.2: Trace model call
            t0 = time.monotonic()
            response = self.provider.chat(self.messages, tools=tools)
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            self.total_usage = self.total_usage + response.usage

            # P4.2: Emit model call span
            if self.tracer and response:
                self.tracer.record_model_call(
                    model=getattr(self.provider, "model", "unknown"),
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost=response.usage.cost,
                    latency_ms=latency_ms,
                    error=str(response.error) if response.error else None,
                )

            # P1: Check for provider errors — break on non-retryable failures
            if response.error is not None:
                if response.error.is_retryable:
                    logger.error(
                        "[%s] provider error (retry exhausted): %s — stopping",
                        self.name, response.error,
                    )
                else:
                    logger.error(
                        "[%s] non-retryable provider error: %s — stopping",
                        self.name, response.error,
                    )
                break

            # A length-truncated completion is untrusted: the payload may be
            # partial reasoning or a cut-off structured answer.  Fail closed
            # instead of returning a possibly-incomplete answer.
            if response.finish_reason == "length":
                reason = "completion truncated (finish_reason=length)"
                if response.reasoning_content:
                    reason += (
                        "; reasoning/thinking may have consumed the entire "
                        "max_tokens budget — increase max_tokens or disable thinking"
                    )
                logger.error(
                    "[%s] turn %d: %s — failing closed",
                    self.name, self.turn_count, reason,
                )
                final_text = ""
                self.last_response_rejected = True
                self.last_rejection_reason = reason
                break

            final_text = response.text

            # P3.6: Loop termination guards
            # Check token budget
            if self.max_total_tokens > 0:
                used = self.total_usage.prompt_tokens + self.total_usage.completion_tokens
                if used >= self.max_total_tokens:
                    logger.warning(
                        "[%s] token budget exceeded (%d/%d)",
                        self.name, used, self.max_total_tokens,
                    )
                    break

            # Check wall clock
            if self.max_wall_clock_seconds > 0 and (time.monotonic() - start_time) >= self.max_wall_clock_seconds:
                logger.warning(
                    "[%s] wall clock timeout (%.1fs/%.1fs)",
                    self.name, time.monotonic() - start_time, self.max_wall_clock_seconds,
                )
                break

            logger.info(
                "[%s] turn %d: text=%d tool_calls=%d usage=%s",
                self.name, self.turn_count,
                len(response.text), len(response.tool_calls),
                response.usage,
            )

            # Append assistant response
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.text:
                assistant_msg["content"] = response.text
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            self.messages.append(assistant_msg)

            # No tool calls → we're done
            if not response.tool_calls:
                if (
                    not response.text
                    and tools_executed
                    and self.finalize_empty_tool_response
                    and not self.finalization_attempted
                ):
                    final_text = self._finalize_empty_tool_response(start_time)
                break

            # Execute tools
            tools_executed = True
            t0 = time.monotonic()
            results = self.tool_registry.execute(response.tool_calls, context=self.messages)
            latency_ms = round((time.monotonic() - t0) * 1000, 2)

            # P4.2: Emit tool call spans
            if self.tracer and results:
                for res in results:
                    tool_name = res.get("_tool_name", "unknown")
                    error = res.get("_error")
                    self.tracer.record_tool_call(
                        tool_name=tool_name,
                        latency_ms=latency_ms / max(len(results), 1),
                        error=str(error) if error else None,
                    )

            provider_coalesces = getattr(self.provider, "coalesce_tool_results", True)
            if self.coalesce_results and provider_coalesces and len(results) > 1:
                coalesced = self._coalesce_results(results)
                self.messages.append(coalesced)
            else:
                for res in results:
                    self.messages.append(res)

        # max_turns exhausted while the model was still requesting tools:
        # attempt one bounded tool-free final answer instead of silently
        # returning the last tool-only turn's (empty) text.  If finalization
        # yields nothing, the previous final_text stands unchanged.
        if (
            response is not None
            and response.tool_calls
            and response.error is None
            and self.finalize_empty_tool_response
            and not self.finalization_attempted
        ):
            fin_text = self._finalize_empty_tool_response(start_time)
            if fin_text:
                final_text = fin_text

        return final_text

    def _finalize_empty_tool_response(self, start_time: float | None = None) -> str:
        """Request one tool-free final response after an empty post-tool turn.

        This is intentionally a bounded escape hatch for providers that emit
        reasoning or an empty content channel after completing tool calls.  A
        finalization call cannot execute tools, and its result is accepted only
        from ``ChatResponse.text`` by callers enforcing a structured contract.
        """
        self.finalization_attempted = True
        used = self.total_usage.prompt_tokens + self.total_usage.completion_tokens
        remaining = self.max_total_tokens - used if self.max_total_tokens > 0 else 0
        if self.max_total_tokens > 0 and remaining <= 0:
            logger.warning("[%s] finalization skipped: token budget exhausted (%d/%d)", self.name, used, self.max_total_tokens)
            self.last_response_rejected = True
            self.last_rejection_reason = "finalization token budget exhausted"
            return ""
        if (
            start_time is not None
            and self.max_wall_clock_seconds > 0
            and time.monotonic() - start_time >= self.max_wall_clock_seconds
        ):
            logger.warning("[%s] finalization skipped: wall clock budget exhausted", self.name)
            self.last_response_rejected = True
            self.last_rejection_reason = "finalization wall clock budget exhausted"
            return ""
        self.messages.append({
            "role": "user",
            "content": self.finalization_prompt,
        })
        t0 = time.monotonic()
        original_max_tokens = getattr(self.provider, "max_tokens", None)
        if self.max_total_tokens > 0 and isinstance(original_max_tokens, int):
            setattr(self.provider, "max_tokens", min(original_max_tokens, remaining))
        try:
            response = self.provider.chat(self.messages, tools=None)
        finally:
            if isinstance(original_max_tokens, int):
                setattr(self.provider, "max_tokens", original_max_tokens)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        self.total_usage = self.total_usage + response.usage
        if self.max_total_tokens > 0:
            used_after = self.total_usage.prompt_tokens + self.total_usage.completion_tokens
            if used_after > self.max_total_tokens:
                logger.warning("[%s] finalization exceeded token budget (%d/%d); rejecting response", self.name, used_after, self.max_total_tokens)
                self.last_response_rejected = True
                self.last_rejection_reason = "finalization exceeded token budget"
                return ""
        if self.tracer:
            self.tracer.record_model_call(
                model=getattr(self.provider, "model", "unknown"),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost=response.usage.cost,
                latency_ms=latency_ms,
                error=str(response.error) if response.error else None,
            )
        if response.error is not None:
            logger.error("[%s] tool-free finalization failed: %s", self.name, response.error)
            self.last_response_rejected = True
            self.last_rejection_reason = "finalization provider error"
            return ""
        if response.tool_calls:
            logger.error("[%s] tool-free finalization unexpectedly returned tool calls", self.name)
            self.last_response_rejected = True
            self.last_rejection_reason = "finalization returned unexpected tool calls"
            return ""
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if response.text:
            assistant_msg["content"] = response.text
        self.messages.append(assistant_msg)
        return response.text

    # ------------------------------------------------------------------
    # Result coalescing
    # ------------------------------------------------------------------

    def _coalesce_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine parallel tool results into a single user message.

        For OpenAI/Anthropic APIs, we preserve tool_call_id associations
        by embedding them in a structured format the model can parse.

        Args:
            results: List of tool result messages from ToolRegistry.execute().

        Returns:
            A single user message containing all tool results with
            tool_call_id associations preserved.
        """
        blocks: list[str] = []
        for r in results:
            tool_name = r.get("_tool_name", "unknown")
            tool_id = r.get("tool_call_id", "no-id")
            content = r.get("content", "")
            blocks.append(
                f"## {tool_name} (id={tool_id})\n{content}"
            )

        return {
            "role": "user",
            "content": "[TOOL_RESULTS — batch of {}]\n\n{}".format(
                len(results), "\n\n---\n\n".join(blocks),
            ),
        }

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def _check_steering(self) -> Optional[str]:
        """Non-blocking check for steering messages on the queue."""
        if not self.steering_queue:
            return None
        try:
            return self.steering_queue.get_nowait()
        except Empty:
            return None

    def _inject_steering(
        self, steering_msg: str, messages: list[dict[str, Any]],
    ) -> None:
        """Inject a steering message mid-conversation.

        Wraps the message in a special marker so the model treats it
        as a high-priority redirection.

        Args:
            steering_msg: The steering instruction.
            messages: Current conversation history (modified in place).
        """
        wrapped = (
            f"[OUT-OF-BAND STEERING]\n{steering_msg}\n"
            f"[END STEERING]\n\n"
            f"Please adjust your approach accordingly and continue."
        )
        messages.append({"role": "user", "content": wrapped})
        logger.info("[%s] steering injected: %s", self.name, steering_msg[:80])

    # ------------------------------------------------------------------
    # Self-compaction
    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token estimate for the message list.

        Uses character count / 3.5 as a fast approximation.
        For production, swap in tiktoken-based counting.

        Args:
            messages: Conversation history.

        Returns:
            Approximate token count.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(str(c)) for c in content)
            # tool_calls add overhead
            for tc in msg.get("tool_calls", []):
                total += len(json.dumps(tc))
        return max(1, total // 4)

    def _maybe_compact(self) -> None:
        """Compact context if utilization exceeds the threshold."""
        token_count = self._estimate_tokens(self.messages)
        ratio = token_count / self.max_context_tokens

        if ratio >= self.compaction_threshold:
            logger.info(
                "[%s] context at %.0f%% — compacting",
                self.name, ratio * 100,
            )
            msgs_before = len(self.messages)
            try:
                self._self_compact(self.messages)
            except Exception:
                logger.exception("[%s] compaction failed — truncating", self.name)
                # P4.3: Fallback: hard truncation of oldest messages
                self._hard_truncate()
            finally:
                msgs_after = len(self.messages)
                # P4.2: Emit compaction span
                if self.tracer:
                    self.tracer.record_compaction(
                        messages_before=msgs_before,
                        messages_after=msgs_after,
                        method="summary" if msgs_after > 0 else "truncate",
                    )

    def _self_compact(self, messages: list[dict[str, Any]]) -> None:
        """Autonomous context compaction with session preservation.

        Replaces the conversation history with a carryover summary
        generated by the model, but preserves the full history in the
        session store for later recall.

        Args:
            messages: Current conversation history (modified in place).
        """
        # Preserve full history before compaction
        if self.session_store and self.session_id:
            try:
                from agent_framework.session import SessionMessage

                for msg in messages:
                    self.session_store.append_message(
                        self.session_id,
                        SessionMessage(
                            role=msg.get("role", "unknown"),
                            content=msg.get("content", ""),
                            timestamp=time.time(),
                            tool_calls=msg.get("tool_calls"),
                            tool_call_id=msg.get("tool_call_id"),
                        ),
                    )
            except Exception:
                logger.exception("Failed to preserve messages before compaction")

        # Build compaction prompt from current history
        user_msg = messages[-1] if messages[-1].get("role") == "user" else None
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

        # Serialize messages for summarization (last 30 only to keep it lean)
        history_text = json.dumps(messages[-30:], default=str)
        compact_prompt = (
            "You are helping compress this conversation for context window management.\n"
            "Summarize the key decisions, findings, and current state in 10-15 bullet points.\n"
            "Preserve specific values, file paths, and conclusions.\n\n"
            f"Conversation history:\n{history_text}"
        )

        compact_messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": compact_prompt},
        ]
        try:
            resp = self.provider.chat(compact_messages)
        except Exception:
            logger.exception("[%s] compaction failed — skipping", self.name)
            return  # Keep existing messages, skip compaction

        carryover = (
            "[COMPACTED CONTEXT]\n"
            f"{resp.text}\n"
            "[END COMPACTED CONTEXT]"
        )

        # Record compaction point
        if self.session_store and self.session_id:
            try:
                self.session_store.record_compaction(
                    session_id=self.session_id,
                    summary=resp.text,
                    messages_before=len(messages),
                    messages_after=3,  # system + carryover + user
                    method="summary",
                )
            except Exception:
                logger.exception("Failed to record compaction point")

        new_messages: list[dict[str, Any]] = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({"role": "user", "content": carryover})
        if user_msg:
            new_messages.append(user_msg)

        messages.clear()
        messages.extend(new_messages)
        self.messages = messages

        logger.info(
            "[%s] compacted: %d → %d messages (full history preserved in session store)",
            self.name, len(messages) + 1, len(new_messages),
        )

    def _hard_truncate(self) -> None:
        """Fallback: keep last 10 messages + system prompt."""
        system_msg = None
        if self.messages and self.messages[0].get("role") == "system":
            system_msg = self.messages[0]
            self.messages = self.messages[-10:]
        else:
            self.messages = self.messages[-10:]
        if system_msg:
            self.messages.insert(0, system_msg)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the loop state for a new conversation."""
        self.messages = []
        self.total_usage = UsageStats()
        self.turn_count = 0
        self.finalization_attempted = False
        self.last_response_rejected = False
        self.last_rejection_reason = None

    @property
    def message_count(self) -> int:
        """Number of messages in the current conversation."""
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"AgentLoop(name={self.name!r}, turns={self.turn_count}, "
            f"messages={len(self.messages)}, usage={self.total_usage})"
        )
