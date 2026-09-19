"""Tests for the 4 dogfood-run fixes.

- Provider clients accept an explicit ``timeout`` (openai/anthropic SDK
  default ~600s read otherwise kills legitimately long generations).
- Length-truncation failure message mentions the reasoning/max_tokens
  interaction when a thinking pass is present in the response.
- The ``read_file`` builtin supports a ``raw`` mode without ``N|`` prefixes.
- ``model_targets`` public surface is re-exported at the package root.
"""

from __future__ import annotations

import pytest

from agent_framework.builtins import _read_file_executor, read_file_factory
from agent_framework.providers import (
    AnthropicProvider,
    ChatResponse,
    OpenAIProvider,
    UsageStats,
    VLLMProvider,
)


# ---------------------------------------------------------------------------
# Finding 1: provider timeout parameter
# ---------------------------------------------------------------------------

class TestProviderTimeout:
    def test_vllm_timeout_is_passed_to_client(self):
        p = VLLMProvider(timeout=3600.0)
        assert p._client.timeout == 3600.0

    def test_vllm_default_timeout_is_sdk_default(self):
        p = VLLMProvider()
        # SDK default is a Timeout object (read=600), not our custom float.
        assert p._client.timeout != 3600.0
        assert getattr(p._client.timeout, "read", None) == 600

    def test_openai_timeout_is_passed_to_client(self):
        p = OpenAIProvider(api_key="sk-test", timeout=1800.0)
        assert p._client.timeout == 1800.0

    def test_openai_default_timeout_is_sdk_default(self):
        p = OpenAIProvider(api_key="sk-test")
        assert p._client.timeout != 1800.0
        assert getattr(p._client.timeout, "read", None) == 600

    def test_anthropic_timeout_is_passed_to_client(self):
        p = AnthropicProvider(api_key="sk-test", timeout=900.0)
        assert p._client.timeout == 900.0

    def test_anthropic_default_timeout_is_sdk_default(self):
        p = AnthropicProvider(api_key="sk-test")
        assert p._client.timeout != 900.0


# ---------------------------------------------------------------------------
# Finding 2: truncation message mentions reasoning/max_tokens
# ---------------------------------------------------------------------------

class TestTruncationReasoningMessage:
    def _run_with_length_response(self, reasoning_content: str) -> str:
        from agent_framework.loop import AgentLoop
        from agent_framework.tools import ToolRegistry

        class _Provider:
            coalesce_tool_results = True

            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, tools=None) -> ChatResponse:
                self.calls += 1
                return ChatResponse(
                    text="partial",
                    reasoning_content=reasoning_content,
                    finish_reason="length",
                    usage=UsageStats(prompt_tokens=5, completion_tokens=10),
                )

        provider = _Provider()
        loop = AgentLoop(
            provider=provider,  # type: ignore[arg-type]
            system_prompt="s",
            tool_registry=ToolRegistry(),
        )
        loop.run("go")
        return loop.last_rejection_reason or ""

    def test_reasoning_truncation_message_mentions_budget(self):
        reason = self._run_with_length_response("a lot of thinking")
        assert "finish_reason=length" in reason
        assert "max_tokens" in reason
        assert "reasoning" in reason.lower()

    def test_plain_truncation_message_stays_generic(self):
        reason = self._run_with_length_response("")
        assert reason == "completion truncated (finish_reason=length)"


# ---------------------------------------------------------------------------
# Finding 3: read_file raw mode
# ---------------------------------------------------------------------------

class TestReadFileRawMode:
    @pytest.fixture
    def sample_file(self, tmp_path, monkeypatch):
        p = tmp_path / "sample.txt"
        p.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n")
        monkeypatch.chdir(tmp_path)
        return str(p)

    def test_raw_false_matches_current_format(self, sample_file):
        out = _read_file_executor(sample_file)
        assert out == (
            "Lines 1-5 of 5:\n"
            "1|alpha\n2|beta\n3|gamma\n4|delta\n5|epsilon\n"
        )

    def test_raw_true_has_no_prefixes_same_content(self, sample_file):
        out = _read_file_executor(sample_file, raw=True)
        assert "Lines " not in out
        for prefix in ("1|", "2|", "3|", "4|", "5|"):
            assert prefix not in out
        assert out == "alpha\nbeta\ngamma\ndelta\nepsilon\n"

    def test_raw_true_respects_offset_and_limit(self, sample_file):
        out = _read_file_executor(sample_file, offset=2, limit=2, raw=True)
        assert out == "beta\ngamma\n"

    def test_default_mode_respects_offset_and_limit(self, sample_file):
        out = _read_file_executor(sample_file, offset=2, limit=2)
        assert out == "Lines 2-3 of 5:\n2|beta\n3|gamma\n"

    def test_schema_declares_raw_property(self):
        tool = read_file_factory()
        props = tool.parameters["properties"]
        raw = props["raw"]
        assert raw["type"] == "boolean"
        assert raw["default"] is False
        assert "raw" not in tool.parameters["required"]

    def test_factory_executor_forwards_raw(self, tmp_path, monkeypatch):
        p = tmp_path / "n.txt"
        p.write_text("one\ntwo\n")
        monkeypatch.chdir(tmp_path)
        tool = read_file_factory()
        assert tool.executor(str(p)) == "Lines 1-2 of 2:\n1|one\n2|two\n"
        assert tool.executor(str(p), raw=True) == "one\ntwo\n"


# ---------------------------------------------------------------------------
# Finding 4: model_targets re-exported at package root
# ---------------------------------------------------------------------------

class TestModelTargetsRootExports:
    def test_imports_from_package_root(self):
        from agent_framework import (
            SUPPORTED_PROVIDERS,
            SUPPORTED_VERSION,
            ModelTarget,
            ModelTargetError,
            load_model_targets,
            make_provider,
            resolve_model_target,
        )

        assert SUPPORTED_VERSION == 1
        assert {"vllm", "openai"} == SUPPORTED_PROVIDERS
        assert ModelTargetError.__bases__[0] is ValueError

    def test_exports_are_in_all(self):
        import agent_framework

        for name in (
            "ModelTarget",
            "ModelTargetError",
            "SUPPORTED_VERSION",
            "SUPPORTED_PROVIDERS",
            "load_model_targets",
            "resolve_model_target",
            "make_provider",
        ):
            assert name in agent_framework.__all__
