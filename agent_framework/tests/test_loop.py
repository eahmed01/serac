"""Tests for AgentLoop — message loop, coalescing, steering, compaction."""

from __future__ import annotations

from queue import Queue
from unittest.mock import MagicMock, patch

from agent_framework.loop import AgentLoop
from agent_framework.providers import ChatResponse, Provider, UsageStats
from agent_framework.retry import ErrorKind, ProviderError
from agent_framework.tools import ToolRegistry


class MockProvider(Provider):
    """Controllable mock provider for testing."""

    coalesce_tool_results = True

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = responses
        self.index = 0
        self.calls: list[tuple[list, list | None]] = []

    def chat(
        self,
        messages: list,
        tools=None,
    ) -> ChatResponse:
        self.calls.append((messages, tools))
        resp = self.responses[self.index]
        self.index += 1
        return resp


class TestAgentLoopBasic:
    def test_run_no_tools(self):
        provider = MockProvider([
            ChatResponse(text="That's the answer!", usage=UsageStats(prompt_tokens=10, completion_tokens=20)),
        ])
        registry = ToolRegistry()
        loop = AgentLoop(
            provider=provider,
            system_prompt="You are helpful.",
            tool_registry=registry,
        )

        result = loop.run("What is 2+2?")
        assert result == "That's the answer!"
        assert loop.turn_count == 1
        assert loop.total_usage.completion_tokens == 20

    def test_run_with_tool_call(self):
        registry = ToolRegistry()

        call_count = 0
        def add_tool(a, b, **kwargs):
            nonlocal call_count
            call_count += 1
            return a + b

        from agent_framework.tools import ToolDef
        registry.register(ToolDef(
            name="add",
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            executor=add_tool,
        ))

        provider = MockProvider([
            ChatResponse(
                text="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "add",
                    "arguments": {"a": 3, "b": 4},
                }],
                usage=UsageStats(prompt_tokens=5, completion_tokens=10),
            ),
            ChatResponse(
                text="The answer is 7.",
                usage=UsageStats(prompt_tokens=20, completion_tokens=5),
            ),
        ])

        loop = AgentLoop(
            provider=provider,
            system_prompt="Calculator.",
            tool_registry=registry,
        )

        result = loop.run("What is 3+4?")
        assert result == "The answer is 7."
        assert call_count == 1
        assert loop.turn_count == 2

    def _lookup_registry(self, executed):
        registry = ToolRegistry()
        registry.register(ToolDef(
            name="lookup",
            description="Look up a value",
            parameters={"type": "object", "properties": {}},
            executor=lambda **kwargs: executed.append("executed") or "lookup result",
        ))
        return registry

    def test_opt_in_finalizes_empty_post_tool_response_without_turn_budget(self):
        executed = []
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text=""),
            ChatResponse(text="final answer"),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry(executed),
            max_turns=2,
            finalize_empty_tool_response=True,
            finalization_prompt="Return exactly one JSON object.",
        )

        assert loop.run("go") == "final answer"
        assert executed == ["executed"]
        assert len(provider.calls) == 3
        assert provider.calls[2][1] is None
        assert any(
            "exactly one JSON object" in message.get("content", "")
            for message in provider.calls[2][0]
        )
        assert loop.turn_count == 2
        assert loop.finalization_attempted is True

    def test_finalization_empty_response_fails_closed(self):
        executed = []
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text=""),
            ChatResponse(text=""),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry(executed),
            finalize_empty_tool_response=True,
        )

        assert loop.run("go") == ""
        assert len(provider.calls) == 3
        assert provider.calls[2][1] is None
        assert executed == ["executed"]

    def test_no_tool_empty_response_does_not_finalize(self):
        provider = MockProvider([ChatResponse(text="")])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=ToolRegistry(),
            finalize_empty_tool_response=True,
        )

        assert loop.run("go") == ""
        assert len(provider.calls) == 1
        assert loop.finalization_attempted is False

    def test_default_finalization_prompt_is_format_neutral(self):
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text=""),
            ChatResponse(text="answer"),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry([]),
            finalize_empty_tool_response=True,
        )
        loop.run("go")
        prompt = provider.calls[2][0][-2]["content"]
        assert "exactly one JSON object" not in prompt
        assert "requested format" in prompt

    def test_finalization_is_skipped_when_token_budget_is_exhausted(self):
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text="", usage=UsageStats(prompt_tokens=10)),
            ChatResponse(text="should not be called"),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry([]),
            finalize_empty_tool_response=True,
            max_total_tokens=1,
        )
        assert loop.run("go") == ""
        assert len(provider.calls) == 2

    def test_finalization_rejects_post_call_token_overrun(self):
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text=""),
            ChatResponse(text="done", usage=UsageStats(prompt_tokens=100, completion_tokens=100)),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry([]),
            finalize_empty_tool_response=True,
            max_total_tokens=10,
        )
        assert loop.run("go") == ""
        assert len(provider.calls) == 3
        assert loop.total_usage.prompt_tokens + loop.total_usage.completion_tokens == 200

    def test_provider_error_after_tools_does_not_trigger_fallback_finalization(self):
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text="", error=ProviderError(ErrorKind.UNKNOWN, "provider failed")),
            ChatResponse(text="unsafe fallback"),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry([]),
            finalize_empty_tool_response=True,
        )
        assert loop.run("go") == ""
        assert len(provider.calls) == 2
        assert loop.finalization_attempted is False

    def test_default_loop_does_not_finalize_empty_post_tool_response(self):
        executed = []
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text=""),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry(executed),
        )

        assert loop.run("go") == ""
        assert len(provider.calls) == 2
        assert loop.finalization_attempted is False

    def test_finalization_does_not_execute_unexpected_tool_call(self):
        executed = []
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            ChatResponse(text=""),
            ChatResponse(text="", tool_calls=[{"id": "c2", "name": "lookup", "arguments": {}}]),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=self._lookup_registry(executed),
            finalize_empty_tool_response=True,
        )

        assert loop.run("go") == ""
        assert len(provider.calls) == 3
        assert executed == ["executed"]

    def test_max_turns(self):
        provider = MockProvider([
            ChatResponse(
                text="",
                tool_calls=[{"id": "c1", "name": "loop", "arguments": {}}],
                usage=UsageStats(),
            )
            for _ in range(10)
        ])
        registry = ToolRegistry()
        registry.register(ToolDef(
            name="loop",
            description="Loop tool",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "loop",
        ))

        loop = AgentLoop(
            provider=provider,
            system_prompt="Loop.",
            tool_registry=registry,
            max_turns=3,
        )

        result = loop.run("start")
        assert loop.turn_count <= 3


class TestCoalescing:
    def test_coalesce_results(self):
        loop = AgentLoop(
            provider=MockProvider([]),
            system_prompt="test",
            tool_registry=ToolRegistry(),
        )

        results = [
            {"role": "tool", "content": "result A", "_tool_name": "tool_a", "tool_call_id": "call_1"},
            {"role": "tool", "content": "result B", "_tool_name": "tool_b", "tool_call_id": "call_2"},
            {"role": "tool", "content": "result C", "_tool_name": "tool_c", "tool_call_id": "call_3"},
        ]

        coalesced = loop._coalesce_results(results)
        assert coalesced["role"] == "user"
        assert "## tool_a (id=call_1)" in coalesced["content"]
        assert "result B" in coalesced["content"]
        assert "batch of 3" in coalesced["content"]
        assert "---" in coalesced["content"]

    def test_coalesce_single_result(self):
        loop = AgentLoop(
            provider=MockProvider([]),
            system_prompt="test",
            tool_registry=ToolRegistry(),
        )
        results = [{"role": "tool", "content": "only one", "_tool_name": "solo"}]
        coalesced = loop._coalesce_results(results)
        assert "only one" in coalesced["content"]
        assert "---" not in coalesced["content"]

    def test_provider_can_require_structured_tool_results(self):
        registry = ToolRegistry()
        registry.register(ToolDef(
            name="first",
            description="First tool",
            parameters={"type": "object", "properties": {}},
            executor=lambda **kwargs: "first result",
        ))
        registry.register(ToolDef(
            name="second",
            description="Second tool",
            parameters={"type": "object", "properties": {}},
            executor=lambda **kwargs: "second result",
        ))
        provider = MockProvider([
            ChatResponse(text="", tool_calls=[
                {"id": "call_1", "name": "first", "arguments": {}},
                {"id": "call_2", "name": "second", "arguments": {}},
            ]),
            ChatResponse(text="done"),
        ])
        provider.coalesce_tool_results = False
        loop = AgentLoop(provider=provider, system_prompt="test", tool_registry=registry)

        assert loop.run("go") == "done"
        second_call_messages = provider.calls[1][0]
        assert [message["role"] for message in second_call_messages[2:5]] == [
            "assistant", "tool", "tool",
        ]


class TestSteering:
    def test_steering_injection(self):
        loop = AgentLoop(
            provider=MockProvider([]),
            system_prompt="test",
            tool_registry=ToolRegistry(),
        )

        msgs = [{"role": "user", "content": "original"}]
        loop._inject_steering("Stop and reconsider.", msgs)

        assert len(msgs) == 2
        assert "[OUT-OF-BAND STEERING]" in msgs[1]["content"]
        assert "Stop and reconsider." in msgs[1]["content"]

    def test_steering_queue(self):
        q: Queue = Queue()
        q.put("Redirect: focus on security.")

        loop = AgentLoop(
            provider=MockProvider([
                ChatResponse(text="done", usage=UsageStats()),
            ]),
            system_prompt="test",
            tool_registry=ToolRegistry(),
            steering_queue=q,
        )

        # _check_steering should pick up the message
        msg = loop._check_steering()
        assert msg == "Redirect: focus on security."

    def test_steering_queue_empty(self):
        q: Queue = Queue()
        loop = AgentLoop(
            provider=MockProvider([]),
            system_prompt="test",
            tool_registry=ToolRegistry(),
            steering_queue=q,
        )
        assert loop._check_steering() is None


class TestCompaction:
    def test_token_estimate(self):
        loop = AgentLoop(
            provider=MockProvider([]),
            system_prompt="test",
            tool_registry=ToolRegistry(),
        )

        msgs = [{"role": "user", "content": "x" * 300}]
        tokens = loop._estimate_tokens(msgs)
        assert tokens > 0

    def test_no_compaction_below_threshold(self):
        provider = MockProvider([
            ChatResponse(text="short", usage=UsageStats()),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=ToolRegistry(),
            max_context_tokens=128_000,
            compaction_threshold=0.75,
        )

        initial_msg_count = 2  # system + user
        loop.run("short question")
        # Should not have compacted
        assert loop.message_count >= initial_msg_count


class TestAgentLoopReset:
    def test_reset(self):
        provider = MockProvider([
            ChatResponse(text="ok", usage=UsageStats()),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=ToolRegistry(),
        )
        loop.run("hello")
        assert loop.turn_count > 0
        assert loop.message_count > 0

        loop.reset()
        assert loop.turn_count == 0
        assert loop.message_count == 0

    def test_repr(self):
        provider = MockProvider([])
        loop = AgentLoop(
            provider=provider,
            system_prompt="test",
            tool_registry=ToolRegistry(),
            name="test-agent",
        )
        r = repr(loop)
        assert "test-agent" in r


class TestAgentLoopExhaustionAndTruncation:
    """Regression tests for the empty-output failure mode:
    max_turns exhausted mid-tool-loop must not silently return "", and
    length-truncated completions must fail closed."""

    def _registry_with_noop(self):
        registry = ToolRegistry()
        registry.register(ToolDef(
            name="noop",
            description="no-op",
            parameters={"type": "object", "properties": {}},
            executor=lambda **kwargs: "ok",
        ))
        return registry

    @staticmethod
    def _tool_turn():
        return ChatResponse(
            text="",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "noop", "arguments": "{}"}}],
            usage=UsageStats(prompt_tokens=5, completion_tokens=5),
        )

    def test_max_turns_exhausted_triggers_tool_free_finalization(self):
        provider = MockProvider([
            self._tool_turn(),
            self._tool_turn(),
            ChatResponse(text='{"final": true}',
                         usage=UsageStats(prompt_tokens=5, completion_tokens=5)),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="s",
            tool_registry=self._registry_with_noop(),
            max_turns=2,
            finalize_empty_tool_response=True,
            finalization_prompt="FINALIZE NOW",
        )
        result = loop.run("go")
        assert result == '{"final": true}'
        assert loop.finalization_attempted is True
        assert len(provider.calls) == 3
        # The finalization call must be tool-free.
        assert provider.calls[2][1] is None

    def test_exhaustion_finalization_empty_preserves_prior_text(self):
        prior = "notes from last turn"
        provider = MockProvider([
            self._tool_turn(),
            ChatResponse(text=prior,
                         tool_calls=[{"id": "c2", "type": "function",
                                      "function": {"name": "noop", "arguments": "{}"}}],
                         usage=UsageStats(prompt_tokens=5, completion_tokens=5)),
            ChatResponse(text="", usage=UsageStats(prompt_tokens=5, completion_tokens=0)),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="s",
            tool_registry=self._registry_with_noop(),
            max_turns=2,
            finalize_empty_tool_response=True,
            finalization_prompt="FINALIZE NOW",
        )
        result = loop.run("go")
        # Finalization produced nothing: the last turn's text must stand.
        assert result == prior
        assert loop.finalization_attempted is True
        assert len(provider.calls) == 3

    def test_length_truncated_completion_fails_closed(self):
        provider = MockProvider([
            ChatResponse(text='{"partial": ', finish_reason="length",
                         usage=UsageStats(prompt_tokens=5, completion_tokens=10)),
        ])
        loop = AgentLoop(
            provider=provider,
            system_prompt="s",
            tool_registry=ToolRegistry(),
        )
        result = loop.run("go")
        assert result == ""
        assert loop.last_response_rejected is True
        assert "finish_reason=length" in (loop.last_rejection_reason or "")
        assert len(provider.calls) == 1


# Import ToolDef at module level for tests
from agent_framework.tools import ToolDef
