"""Tests for providers.py — Provider ABC, ChatResponse, UsageStats."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from unittest.mock import MagicMock, patch

from agent_framework.providers import (
    AnthropicProvider,
    ChatResponse,
    OpenAIProvider,
    Provider,
    UsageStats,
    VLLMProvider,
)


class TestUsageStats:
    def test_default(self):
        u = UsageStats()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.cost == 0.0

    def test_add(self):
        a = UsageStats(prompt_tokens=100, completion_tokens=50, cost=0.1)
        b = UsageStats(prompt_tokens=200, completion_tokens=100, cost=0.2)
        c = a + b
        assert c.prompt_tokens == 300
        assert c.completion_tokens == 150
        assert c.cost == pytest.approx(0.3)


class TestChatResponse:
    def test_defaults(self):
        r = ChatResponse(text="hello")
        assert r.text == "hello"
        assert r.tool_calls == []
        assert isinstance(r.usage, UsageStats)


class TestVLLMProvider:
    def test_init(self):
        p = VLLMProvider(base_url="http://localhost:8000/v1", model="test-model")
        assert p.model == "test-model"
        assert p.max_tokens == 4096
        assert p.chat_template_kwargs == {}

    def test_chat_template_kwargs_are_sent_as_extra_body(self):
        p = VLLMProvider(
            reasoning_effort=None,
            chat_template_kwargs={"enable_thinking": False},
        )
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "ok"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.usage.prompt_tokens = 1
        mock_resp.usage.completion_tokens = 1

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp) as create:
            p.chat([{"role": "user", "content": "hi"}])

        assert create.call_args.kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_reasoning_field_is_preserved(self):
        p = VLLMProvider(reasoning_effort=None)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = ""
        mock_resp.choices[0].message.reasoning_content = ""
        mock_resp.choices[0].message.reasoning = "structured fallback"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.usage.prompt_tokens = 1
        mock_resp.usage.completion_tokens = 1

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp):
            result = p.chat([{"role": "user", "content": "hi"}])

        assert result.reasoning_content == "structured fallback"

    def test_finish_reason_is_preserved(self):
        p = VLLMProvider(reasoning_effort=None)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "partial"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "length"
        mock_resp.usage.prompt_tokens = 1
        mock_resp.usage.completion_tokens = 2

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp):
            result = p.chat([{"role": "user", "content": "hi"}])

        assert result.finish_reason == "length"

    def test_finish_reason_defaults_to_none(self):
        p = VLLMProvider(reasoning_effort=None)
        out = ChatResponse(text="x")
        assert out.finish_reason is None

    def test_chat_text_only(self):
        p = VLLMProvider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Hello!"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 20

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp):
            result = p.chat([{"role": "user", "content": "hi"}])

        assert result.text == "Hello!"
        assert result.tool_calls == []
        assert result.usage.prompt_tokens == 10

    def test_chat_with_tool_calls(self):
        p = VLLMProvider()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search"
        tc.function.arguments = '{"query": "test"}'

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = ""
        mock_resp.choices[0].message.tool_calls = [tc]
        mock_resp.usage.prompt_tokens = 15
        mock_resp.usage.completion_tokens = 30

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp):
            result = p.chat([{"role": "user", "content": "search something"}])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "search"
        assert result.tool_calls[0]["function"]["arguments"] == '{"query": "test"}'
        assert result.tool_calls[0]["type"] == "function"
        assert result.tool_calls[0]["id"] == "call_1"

class TestOpenAIProvider:
    def test_init(self):
        p = OpenAIProvider(api_key="test-key")
        assert p.model == "gpt-4.1"
        assert p.api_mode == "chat_completions"
        assert p.coalesce_tool_results is True

    def test_init_with_base_url(self):
        p = OpenAIProvider(api_key="test-key", base_url="https://api.example.test/v1")
        assert str(p._client.base_url).rstrip("/") == "https://api.example.test/v1"

    def test_chat_uses_completion_token_parameter(self):
        p = OpenAIProvider(api_key="test-key", max_tokens_parameter="max_completion_tokens")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "OpenAI response"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 15

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp) as create:
            p.chat([{"role": "user", "content": "hi"}])

        assert create.call_args.kwargs["max_completion_tokens"] == 4096
        assert "max_tokens" not in create.call_args.kwargs

    def test_responses_tool_schema_and_function_call(self):
        p = OpenAIProvider(
            api_key="test-key",
            model="gpt-test-model",
            api_mode="responses",
            reasoning_effort="medium",
        )
        response = SimpleNamespace(
            id="resp_1",
            output=[SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="search",
                arguments='{"query":"test"}',
            )],
            output_text="",
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search documents",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]

        with patch.object(p._client.responses, "create", return_value=response) as create:
            result = p.chat([
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Find test."},
            ], tools=tools)

        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "gpt-test-model"
        assert kwargs["max_output_tokens"] == 4096
        assert kwargs["reasoning"] == {"effort": "medium"}
        assert kwargs["instructions"] == "Use tools."
        assert kwargs["input"] == [{"role": "user", "content": "Find test."}]
        assert kwargs["tools"] == [{
            "type": "function",
            "name": "search",
            "description": "Search documents",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            "strict": False,
        }]
        assert result.tool_calls == [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query":"test"}'},
        }]
        assert result.usage.prompt_tokens == 11
        assert p.coalesce_tool_results is False

    def test_responses_explicit_reset_clears_identical_new_run(self):
        p = OpenAIProvider(api_key="test-key", api_mode="responses")
        first = SimpleNamespace(
            id="resp_1", output=[SimpleNamespace(type="message", content=[])], output_text="one",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        second = SimpleNamespace(
            id="resp_2", output=[SimpleNamespace(type="message", content=[])], output_text="two",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        messages = [{"role": "user", "content": "identical"}]
        with patch.object(p._client.responses, "create", side_effect=[first, second]) as create:
            assert p.chat(messages).text == "one"
            p.reset_conversation()
            assert p.chat(messages).text == "two"
        second_kwargs = create.call_args_list[1].kwargs
        assert "previous_response_id" not in second_kwargs
        assert second_kwargs["input"] == messages

    def test_responses_continuation_sends_typed_tool_output(self):
        p = OpenAIProvider(api_key="test-key", api_mode="responses", reasoning_effort="medium")
        first = SimpleNamespace(
            id="resp_1",
            output=[SimpleNamespace(
                type="function_call", id="fc_1", call_id="call_1",
                name="search", arguments='{"query":"test"}',
            ), SimpleNamespace(
                type="function_call", id="fc_2", call_id="call_2",
                name="fetch", arguments='{"id":"test"}',
            )],
            output_text="",
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        )
        second = SimpleNamespace(
            id="resp_2",
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="done")],
            )],
            output_text="done",
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Find test."},
        ]
        with patch.object(p._client.responses, "create", side_effect=[first, second]) as create:
            first_result = p.chat(messages, tools=[])
            messages.extend([
                {"role": "assistant", "tool_calls": first_result.tool_calls},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                {"role": "tool", "tool_call_id": "call_2", "content": "fetched"},
            ])
            second_result = p.chat(messages, tools=[])

        assert first_result.tool_calls[0]["id"] == "call_1"
        assert first_result.tool_calls[1]["id"] == "call_2"
        assert second_result.text == "done"
        kwargs = create.call_args_list[1].kwargs
        assert kwargs["previous_response_id"] == "resp_1"
        assert kwargs["input"] == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "fetched",
            },
        ]

    def test_chat_text_only(self):
        p = OpenAIProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "OpenAI response"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 15

        with patch.object(p._client.chat.completions, "create", return_value=mock_resp):
            result = p.chat([{"role": "user", "content": "hi"}])

        assert result.text == "OpenAI response"
        assert len(result.tool_calls) == 0


class TestAnthropicProvider:
    def test_init(self):
        p = AnthropicProvider(api_key="test-key")
        assert p.model == "claude-sonnet-4-20250514"

    def test_build_system(self):
        p = AnthropicProvider(api_key="test-key")
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
        ]
        assert p._build_system(msgs) == "Be helpful"

    def test_build_system_none(self):
        p = AnthropicProvider(api_key="test-key")
        msgs = [{"role": "user", "content": "hi"}]
        assert p._build_system(msgs) is None

    def test_to_anthropic_messages_basic(self):
        p = AnthropicProvider(api_key="test-key")
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        converted = p._to_anthropic_messages(msgs)
        # System should be stripped
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_to_anthropic_messages_with_tool_use(self):
        p = AnthropicProvider(api_key="test-key")
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tu_1", "name": "search", "arguments": {"q": "x"}},
            ]},
            {"role": "tool", "content": "results", "tool_call_id": "tu_1"},
        ]
        converted = p._to_anthropic_messages(msgs)
        assert len(converted) == 2
        assert converted[0]["content"][0]["type"] == "tool_use"
        assert converted[1]["content"][0]["type"] == "tool_result"

    def test_anthropic_tools_conversion(self):
        p = AnthropicProvider(api_key="test-key")
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        a_tools = p._anthropic_tools(openai_tools)
        assert a_tools[0]["name"] == "search"
        assert a_tools[0]["input_schema"]["type"] == "object"

    def test_chat_text_only(self):
        p = AnthropicProvider(api_key="test-key")
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Claude says hello"

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 20

        with patch.object(p._client.messages, "create", return_value=mock_resp):
            result = p.chat([{"role": "user", "content": "hi"}])

        assert result.text == "Claude says hello"
        assert result.tool_calls == []
        assert result.usage.prompt_tokens == 10
