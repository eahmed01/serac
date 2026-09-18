#!/usr/bin/env python3
"""Tests for tool name resolution."""

import pytest
from agent_framework.tools import ToolRegistry, ToolDef
from agent_framework.tool_resolver import (
    ToolResolver,
    AliasPolicy,
    FuzzyPolicy,
    LLMPolicy,
    build_default_resolver,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Registry with common tools."""
    reg = ToolRegistry()
    reg.register(ToolDef(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        executor=lambda path="test.py", **kw: f"Read {path}",
    ))
    reg.register(ToolDef(
        name="code_search",
        description="Search code",
        parameters={"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        executor=lambda pattern="test", **kw: f"Found {pattern}",
    ))
    reg.register(ToolDef(
        name="execute_terminal",
        description="Run shell command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        executor=lambda command="ls", **kw: f"Ran {command}",
    ))
    reg.register(ToolDef(
        name="write_file",
        description="Write a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        executor=lambda path="test.py", content="", **kw: f"Wrote {path}",
    ))
    return reg


@pytest.fixture
def default_resolver(registry):
    """Default resolver with standard aliases."""
    resolver = build_default_resolver(registry=registry)
    return resolver


# ---------------------------------------------------------------------------
# AliasPolicy tests
# ---------------------------------------------------------------------------


class TestAliasPolicy:
    def test_exact_alias(self, registry):
        policy = AliasPolicy({"file_read": "read_file"})
        assert policy.resolve("file_read") == "read_file"

    def test_no_match(self, registry):
        policy = AliasPolicy({"file_read": "read_file"})
        assert policy.resolve("nonexistent") is None

    def test_alias_not_in_registry(self, registry):
        policy = AliasPolicy({"fake_tool": "nonexistent_tool"})
        # The resolver checks registry, so this returns None
        assert policy.resolve("fake_tool") == "nonexistent_tool"  # Policy itself returns raw

    def test_name_property(self):
        policy = AliasPolicy({})
        assert policy.name == "alias"


# ---------------------------------------------------------------------------
# FuzzyPolicy tests
# ---------------------------------------------------------------------------


class TestFuzzyPolicy:
    def test_exact_match(self, registry):
        policy = FuzzyPolicy(threshold=0.8)
        assert policy.resolve("read_file", registry) == "read_file"

    def test_similar_match(self, registry):
        # Fuzzy handles typos, not reordered words (that's alias territory)
        policy = FuzzyPolicy(threshold=0.8)
        assert policy.resolve("read_filee", registry) == "read_file"  # Typo

    def test_reordered_words(self, registry):
        # Reordered words (file_read vs read_file) need aliases, not fuzzy
        policy = FuzzyPolicy(threshold=0.6)
        # This won't match — aliases are the right tool for this
        assert policy.resolve("file_read", registry) is None

    def test_no_match(self, registry):
        policy = FuzzyPolicy(threshold=0.9)
        assert policy.resolve("xyz_nonexistent", registry) is None

    def test_threshold_too_high(self, registry):
        policy = FuzzyPolicy(threshold=0.99)
        assert policy.resolve("file_read", registry) is None

    def test_name_property(self):
        policy = FuzzyPolicy()
        assert policy.name == "fuzzy"


# ---------------------------------------------------------------------------
# LLMPolicy tests
# ---------------------------------------------------------------------------


class TestLLMPolicy:
    def test_custom_resolver(self, registry):
        def my_resolve(name, available):
            if "file" in name:
                return "read_file"
            return None

        policy = LLMPolicy(my_resolve)
        assert policy.resolve("get_file", registry) == "read_file"

    def test_no_resolver(self, registry):
        policy = LLMPolicy(None)
        assert policy.resolve("anything", registry) is None

    def test_resolver_returns_none(self, registry):
        policy = LLMPolicy(lambda n, a: None)
        assert policy.resolve("anything", registry) is None

    def test_name_property(self):
        policy = LLMPolicy(None)
        assert policy.name == "llm"


# ---------------------------------------------------------------------------
# ToolResolver tests
# ---------------------------------------------------------------------------


class TestToolResolver:
    def test_alias_resolution(self, default_resolver):
        assert default_resolver.resolve("file_read") == "read_file"

    def test_fuzzy_resolution(self, default_resolver):
        # "grep" is aliased, but test fuzzy with something not aliased
        assert default_resolver.resolve("code_search") == "code_search"  # Exact match
        assert default_resolver.resolve("search_code") == "code_search"  # Alias

    def test_exact_match(self, default_resolver):
        assert default_resolver.resolve("read_file") == "read_file"

    def test_no_match(self, default_resolver):
        assert default_resolver.resolve("nonexistent_tool_xyz") is None

    def test_custom_aliases(self, registry):
        resolver = build_default_resolver(
            aliases={"my_custom_read": "read_file"},
            registry=registry,
        )
        assert resolver.resolve("my_custom_read") == "read_file"

    def test_cache(self, default_resolver):
        # First call
        result1 = default_resolver.resolve("file_read")
        assert result1 == "read_file"
        # Second call should use cache
        result2 = default_resolver.resolve("file_read")
        assert result2 == "read_file"

    def test_add_aliases(self, registry):
        resolver = build_default_resolver(registry=registry)
        resolver.add_aliases({"new_alias": "write_file"})
        assert resolver.resolve("new_alias") == "write_file"

    def test_set_registry(self, registry):
        resolver = build_default_resolver()
        resolver.set_registry(registry)
        # Should now validate against the registry
        assert resolver.resolve("read_file") == "read_file"


# ---------------------------------------------------------------------------
# Integration: ToolRegistry with resolver
# ---------------------------------------------------------------------------


class TestRegistryWithResolver:
    def test_alias_resolution_in_execute(self, registry, default_resolver):
        registry._resolver = default_resolver
        result = registry.execute([{
            "id": "call_1",
            "type": "function",
            "function": {"name": "file_read", "arguments": '{"path": "test.py"}'},
        }])
        assert len(result) == 1
        assert "Read test.py" in result[0]["content"]

    def test_fuzzy_resolution_in_execute(self, registry, default_resolver):
        registry._resolver = default_resolver
        result = registry.execute([{
            "id": "call_1",
            "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }])
        assert len(result) == 1
        assert "Found test" in result[0]["content"]

    def test_exact_match_no_resolver(self, registry):
        result = registry.execute([{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "test.py"}'},
        }])
        assert len(result) == 1
        assert "Read test.py" in result[0]["content"]

    def test_unknown_tool_no_resolver(self, registry):
        result = registry.execute([{
            "id": "call_1",
            "type": "function",
            "function": {"name": "file_read", "arguments": '{"path": "test.py"}'},
        }])
        assert len(result) == 1
        assert "not found" in result[0]["content"]

    def test_resolver_preserved_in_for_role(self, registry, default_resolver):
        registry._resolver = default_resolver
        filtered = registry.for_role("worker")
        assert filtered._resolver is default_resolver
