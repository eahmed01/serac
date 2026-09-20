"""Provider abstraction for LLM backends.

Supports vLLM (OpenAI-compatible), Anthropic Messages API, and both OpenAI
Chat Completions and Responses APIs. All providers include retry/backoff and
structured error reporting.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from anthropic import Anthropic

from agent_framework.retry import (
    DEFAULT_RETRY,
    ProviderError,
    RetryConfig,
    classify_exception,
    retry_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UsageStats:
    """Token usage and cost for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost=self.cost + other.cost,
        )

    def __repr__(self) -> str:
        return (
            f"UsageStats(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, cost=${self.cost:.4f})"
        )


@dataclass
class ChatResponse:
    """Unified response from any LLM provider.

    Attributes:
        text: Model text output.
        tool_calls: List of tool call dicts (OpenAI or normalized format).
        usage: Token usage and cost.
        error: ProviderError if the call failed, None on success.
    """
    text: str
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: UsageStats = field(default_factory=UsageStats)
    error: Optional[ProviderError] = None


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Base class for LLM providers.

    Subclasses implement ``chat()`` which accepts a conversation history
    and an optional tool schema, returning a ``ChatResponse``. Providers
    whose endpoint requires typed tool-result items set
    ``coalesce_tool_results`` to ``False``.
    """

    # Chat Completions-compatible providers can use the compact batch format;
    # Responses providers override this so call_id associations remain typed.
    coalesce_tool_results = True

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Send a chat request and return the response."""
        ...


# ---------------------------------------------------------------------------
# vLLM provider (OpenAI-compatible API)
# ---------------------------------------------------------------------------

class VLLMProvider(Provider):
    """Provider for local vLLM server using OpenAI-compatible API.

    Args:
        base_url: Base URL of the vLLM server. Defaults to http://localhost:7999.
        model: Model name to use.
        api_key: API key (can be dummy for local servers).
        max_tokens: Maximum tokens in the response.
        reasoning_effort: Optional thinking/reasoning effort level. When set,
            the model produces a thinking pass whose tokens count against
            ``max_tokens`` — a long thinking pass can exhaust the entire
            budget, leaving 0 tokens for the visible answer (the turn then
            finishes with ``finish_reason="length"``). Disable thinking for
            non-reasoning tasks (``None``) or raise ``max_tokens`` accordingly.
        chat_template_kwargs: Optional chat-template kwargs sent as extra body.
        retry_config: Retry configuration. None = no retries.
        timeout: Optional HTTP timeout in seconds (applied to connect/read/
            write/pool) passed to the OpenAI client. None = SDK default
            (~600s read). Raise this for legitimately long generations that
            exceed the SDK default (e.g. very large prompts).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7999/v1",
        model: str = "local-model",
        api_key: str = "sk-dummy",
        max_tokens: int = 4096,
        reasoning_effort: str | None = "high",
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: Optional[float] = None,
    ) -> None:
        import openai  # noqa: PLC0415  # lazy import

        if reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("reasoning_effort must be low, medium, high, xhigh, or None")
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = "xhigh" if reasoning_effort == "high" else reasoning_effort
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.retry_config = retry_config or DEFAULT_RETRY
        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client: Any = openai.OpenAI(**client_kwargs)

    def _create(self, kwargs: dict[str, Any]):
        """Wrapper for API call — used by retry logic."""
        extra_body = {}
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self._client.chat.completions.create(**kwargs)

    def _parse_response(self, resp) -> ChatResponse:
        """Parse OpenAI-format response into ChatResponse."""
        choice = resp.choices[0]
        text = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if not reasoning_content:
            reasoning_content = getattr(choice.message, "reasoning", "") or ""
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        usage = UsageStats(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None

        return ChatResponse(text=text, reasoning_content=reasoning_content, tool_calls=tool_calls,
                            finish_reason=finish_reason, usage=usage)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        def make_call():
            return self._create(kwargs)

        try:
            resp = retry_call(make_call, self.retry_config, "VLLMProvider")
            return self._parse_response(resp)
        except Exception as exc:
            err = getattr(exc, "__provider_error__", None) or classify_exception(exc)
            logger.error("VLLMProvider.chat failed after retries: %s", err)
            return ChatResponse(text=f"Error: {exc}", usage=UsageStats(), error=err)


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(Provider):
    """Provider for Anthropic Messages API with tool_use support.

    Args:
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY if not provided.
        model: Model name. Defaults to claude-sonnet-4-20250514.
        max_tokens: Maximum tokens in the response.
        retry_config: Retry configuration. None = no retries.
        timeout: Optional HTTP timeout in seconds passed to the Anthropic
            client. None = SDK default.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        retry_config: Optional[RetryConfig] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.retry_config = retry_config or DEFAULT_RETRY
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = Anthropic(**client_kwargs)

    # -- internal helpers --

    def _build_system(self, messages: list[dict[str, Any]]) -> Optional[str]:
        """Extract system message from message list."""
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return None

    def _to_anthropic_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                continue
            role = msg["role"]
            if role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg.get("content", ""),
                    }],
                })
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if isinstance(msg.get("content"), str) and msg["content"]:
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("arguments", {}),
                    })
                out.append({"role": "assistant", "content": content})
            else:
                out.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })
        return out

    def _anthropic_tools(self, tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
        """Convert OpenAI tool schema to Anthropic tool_use format."""
        if not tools:
            return None
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

    def _parse_response(self, resp) -> ChatResponse:
        """Parse Anthropic response into ChatResponse."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        usage = UsageStats(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

        return ChatResponse(text="\n".join(text_parts), tool_calls=tool_calls, usage=usage)

    # -- main method --

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        system = self._build_system(messages)
        anthropic_msgs = self._to_anthropic_messages(messages)
        a_tools = self._anthropic_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_msgs,
        }
        if system:
            kwargs["system"] = system
        if a_tools:
            kwargs["tools"] = a_tools

        def make_call():
            return self._client.messages.create(**kwargs)

        try:
            resp = retry_call(make_call, self.retry_config, "AnthropicProvider")
            return self._parse_response(resp)
        except Exception as exc:
            err = getattr(exc, "__provider_error__", None) or classify_exception(exc)
            logger.error("AnthropicProvider.chat failed after retries: %s", err)
            return ChatResponse(text=f"Error: {exc}", usage=UsageStats(), error=err)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(Provider):
    """Provider for OpenAI Chat Completions or Responses with function calling.

    ``chat_completions`` preserves the existing ``messages`` protocol. The
    ``responses`` mode uses typed function-call continuations and
    ``previous_response_id`` so reasoning items remain associated with local
    tool results.

    Args:
        timeout: Optional HTTP timeout in seconds (applied to connect/read/
            write/pool) passed to the OpenAI client. None = SDK default
            (~600s read). Raise this for legitimately long generations that
            exceed the SDK default.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        max_tokens_parameter: str = "max_tokens",
        api_mode: str = "chat_completions",
        reasoning_effort: str | None = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: Optional[float] = None,
    ) -> None:
        import openai  # noqa: PLC0415  # lazy import

        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(f"unsupported api_mode: {api_mode}")
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"unsupported max_tokens_parameter: {max_tokens_parameter}")
        if reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("reasoning_effort must be low, medium, high, xhigh, or None")

        self.model = model
        self.max_tokens = max_tokens
        self.max_tokens_parameter = max_tokens_parameter
        self.api_mode = api_mode
        self.reasoning_effort = reasoning_effort
        # Responses function-call continuations must preserve typed output
        # items; AgentLoop must not collapse them into a synthetic user message.
        self.coalesce_tool_results = api_mode != "responses"
        self._responses_id: Optional[str] = None
        self._responses_conversation_key: Optional[str] = None
        self._responses_message_count = 0
        self._responses_instructions = ""
        self.retry_config = retry_config or DEFAULT_RETRY
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = openai.OpenAI(**client_kwargs)

    def _create(self, kwargs: dict[str, Any]):
        """Wrapper for Chat Completions API calls — used by retry logic."""
        return self._client.chat.completions.create(**kwargs)

    def _create_response(self, kwargs: dict[str, Any]):
        """Wrapper for Responses API calls — used by retry logic."""
        return self._client.responses.create(**kwargs)

    def _parse_response(self, resp) -> ChatResponse:
        """Parse Chat Completions output into ChatResponse."""
        choice = resp.choices[0]
        text = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", "") or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        usage = UsageStats(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None

        return ChatResponse(text=text, reasoning_content=reasoning_content, tool_calls=tool_calls,
                            finish_reason=finish_reason, usage=usage)

    @staticmethod
    def _conversation_key(messages: list[dict[str, Any]]) -> str:
        """Build a stable key for detecting a new AgentLoop conversation."""
        return json.dumps(messages[:2], sort_keys=True, default=str)

    def reset_conversation(self) -> None:
        """Forget server-side Responses chaining before a new AgentLoop run."""
        self._responses_id = None
        self._responses_conversation_key = None
        self._responses_message_count = 0
        self._responses_instructions = ""

    def _reset_responses_state(self, messages: list[dict[str, Any]]) -> None:
        """Reset Responses chaining when a new conversation starts."""
        key = self._conversation_key(messages)
        if key == self._responses_conversation_key:
            return
        self._responses_id = None
        self._responses_message_count = 0
        self._responses_conversation_key = key
        self._responses_instructions = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system" and message.get("content")
        )

    @staticmethod
    def _responses_tools(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
        """Convert Chat Completions function schemas to Responses schemas."""
        if not tools:
            return None
        converted = []
        for tool in tools:
            function = tool["function"]
            converted.append({
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                "strict": False,
            })
        return converted

    @staticmethod
    def _response_function_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        """Convert a framework tool call to a Responses input item."""
        function = tool_call.get("function", tool_call)
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, sort_keys=True)
        return {
            "type": "function_call",
            "call_id": tool_call.get("id", tool_call.get("call_id", "")),
            "name": function.get("name", ""),
            "arguments": arguments,
        }

    @classmethod
    def _responses_input(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert the framework chat history to initial Responses input."""
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": str(message.get("content", "")),
                })
                continue
            content = message.get("content", "")
            if content:
                items.append({"role": role, "content": content})
            for tool_call in message.get("tool_calls", []) or []:
                items.append(cls._response_function_call(tool_call))
        return items

    @staticmethod
    def _responses_continuation_input(
        messages: list[dict[str, Any]],
        previous_message_count: int,
    ) -> list[dict[str, Any]]:
        """Return new local tool outputs or user steering for a chained response."""
        items: list[dict[str, Any]] = []
        for message in messages[previous_message_count:]:
            role = message.get("role")
            if role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": str(message.get("content", "")),
                })
            elif role == "user" and message.get("content"):
                items.append({"role": "user", "content": message["content"]})
        return items

    @staticmethod
    def _parse_responses_response(resp) -> ChatResponse:
        """Parse Responses output items into the framework ChatResponse."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in getattr(resp, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls.append({
                    "id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", "{}"),
                    },
                })
            elif item_type == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text = getattr(content, "text", "")
                        if text:
                            text_parts.append(text)
        output_text = getattr(resp, "output_text", "") or ""
        if output_text and not text_parts:
            text_parts.append(output_text)
        usage_obj = getattr(resp, "usage", None)
        usage = UsageStats(
            prompt_tokens=getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
            completion_tokens=getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
        )
        return ChatResponse(text="\n".join(text_parts), tool_calls=tool_calls, usage=usage)

    def _chat_responses(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
    ) -> ChatResponse:
        """Send one initial or chained Responses API request."""
        self._reset_responses_state(messages)
        if self._responses_id:
            input_items = self._responses_continuation_input(
                messages, self._responses_message_count,
            )
        else:
            input_items = self._responses_input(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
            "store": True,
        }
        if self._responses_instructions:
            kwargs["instructions"] = self._responses_instructions
        response_tools = self._responses_tools(tools)
        if response_tools:
            kwargs["tools"] = response_tools
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self._responses_id:
            kwargs["previous_response_id"] = self._responses_id

        def make_call():
            return self._create_response(kwargs)

        resp = retry_call(make_call, self.retry_config, "OpenAIResponsesProvider")
        self._responses_id = getattr(resp, "id", None)
        self._responses_message_count = len(messages)
        return self._parse_responses_response(resp)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        if self.api_mode == "responses":
            try:
                return self._chat_responses(messages, tools)
            except Exception as exc:
                err = getattr(exc, "__provider_error__", None) or classify_exception(exc)
                logger.error("OpenAIProvider.responses failed after retries: %s", err)
                return ChatResponse(text=f"Error: {exc}", usage=UsageStats(), error=err)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_parameter: self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        def make_call():
            return self._create(kwargs)

        try:
            resp = retry_call(make_call, self.retry_config, "OpenAIProvider")
            return self._parse_response(resp)
        except Exception as exc:
            err = getattr(exc, "__provider_error__", None) or classify_exception(exc)
            logger.error("OpenAIProvider.chat failed after retries: %s", err)
            return ChatResponse(text=f"Error: {exc}", usage=UsageStats(), error=err)
