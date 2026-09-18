"""Tests for ToolRegistry and ToolDef."""

from __future__ import annotations

import pytest

from agent_framework.tools import ToolDef, ToolRegistry


def _make_tool(name: str) -> ToolDef:
    return ToolDef(
        name=name,
        description="test",
        parameters={"type": "object", "properties": {}},
        executor=lambda: None,
    )


class TestToolDef:
    def test_create(self):
        td = ToolDef(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "ok",
        )
        assert td.name == "test_tool"
        assert td.description == "A test tool"
        assert td.role is None
        assert td.execution_mode == "host"
        assert td.requires_sandbox is False

    @pytest.mark.parametrize("requires_sandbox", [True, False])
    def test_requires_sandbox_boolean(self, requires_sandbox):
        td = ToolDef(
            name="sandbox_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            executor=lambda: None,
            requires_sandbox=requires_sandbox,
        )
        assert td.requires_sandbox is requires_sandbox

    @pytest.mark.parametrize("requires_sandbox", [None, 0, 1, "true", []])
    def test_invalid_requires_sandbox_raises(self, requires_sandbox):
        with pytest.raises(ValueError, match="Invalid requires_sandbox"):
            ToolDef(
                name="sandbox_tool",
                description="test",
                parameters={"type": "object", "properties": {}},
                executor=lambda: None,
                requires_sandbox=requires_sandbox,
            )

    @pytest.mark.parametrize("execution_mode", ["sandbox", "host"])
    def test_execution_mode(self, execution_mode):
        td = ToolDef(
            name="mode_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            executor=lambda: None,
            execution_mode=execution_mode,
        )
        assert td.execution_mode == execution_mode

    @pytest.mark.parametrize("execution_mode", ["", "docker", None, 1])
    def test_invalid_execution_mode_raises(self, execution_mode):
        with pytest.raises(ValueError, match="Invalid execution_mode"):
            ToolDef(
                name="mode_tool",
                description="test",
                parameters={"type": "object", "properties": {}},
                executor=lambda: None,
                execution_mode=execution_mode,
            )

    def test_with_role(self):
        td = ToolDef(
            name="admin_tool",
            description="Admin only",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "admin",
            role="admin",
        )
        assert td.role == "admin"

    @pytest.mark.parametrize(
        "name",
        [
            "a",
            "memory_search",
            "web-query",
            "Tool123",
            "a1_b-2",
            "a" + "x" * 63,
        ],
    )
    def test_valid_names(self, name):
        assert _make_tool(name).name == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            ".",
            "tool.name",
            "tool name",
            "tool/name",
            "tool$name",
            "1tool",
            "_tool",
            "-tool",
            "étool",
            "a" * 65,
        ],
    )
    def test_invalid_names_raise(self, name):
        with pytest.raises(ValueError, match="Invalid tool name"):
            _make_tool(name)


class TestToolRegistry:
    @pytest.mark.parametrize("name", ["tool.name", "1tool", "a" * 65])
    def test_register_revalidates_tool_name(self, name):
        tool = _make_tool("valid_tool")
        object.__setattr__(tool, "name", name)

        with pytest.raises(ValueError, match="Invalid tool name"):
            ToolRegistry().register(tool)

    def test_register_and_schema(self):
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            executor=lambda a, b: a + b,
        ))
        assert len(reg) == 1
        assert "add" in reg

        schema = reg.schema
        assert len(schema) == 1
        assert schema[0]["function"]["name"] == "add"
        assert schema[0]["type"] == "function"

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="x",
            description="x",
            parameters={"type": "object", "properties": {}},
            executor=lambda: None,
        ))
        import pytest
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ToolDef(
                name="x",
                description="x2",
                parameters={"type": "object", "properties": {}},
                executor=lambda: None,
            ))

    def test_execute_single(self):
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            executor=lambda name: f"Hello {name}",
        ))

        results = reg.execute([{"name": "greet", "arguments": {"name": "world"}}])
        assert len(results) == 1
        assert "Hello world" in results[0]["content"]

    def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        results = reg.execute([{"name": "no_such_tool", "arguments": {}}])
        assert len(results) == 1
        assert "not found" in results[0]["content"]

    def test_execute_exception_handling(self):
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="crash",
            description="Always crashes",
            parameters={"type": "object", "properties": {}},
            executor=lambda: 1 / 0,
        ))

        results = reg.execute([{"name": "crash", "arguments": {}}])
        assert len(results) == 1
        assert "Error" in results[0]["content"]

    def test_execute_empty(self):
        reg = ToolRegistry()
        results = reg.execute([])
        assert results == []

    def test_for_role(self):
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="public",
            description="Public tool",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "public",
        ))
        reg.register(ToolDef(
            name="admin_only",
            description="Admin tool",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "admin",
            role="admin",
        ))

        # "anyone" role: only role=None tools + tools matching "anyone"
        anyone_view = reg.for_role("anyone")
        assert len(anyone_view) == 1  # only "public"
        assert "public" in anyone_view

        # "admin" role: role=None tools + tools matching "admin"
        admin_view = reg.for_role("admin")
        assert len(admin_view) == 2  # "public" + "admin_only"

    def test_context_passed_to_executor(self):
        reg = ToolRegistry()
        received_context = {}

        def _ctx_tool(context=None):
            received_context["got"] = context
            return "done"

        reg.register(ToolDef(
            name="ctx_tool",
            description="Test context passing",
            parameters={"type": "object", "properties": {}},
            executor=_ctx_tool,
        ))

        ctx = [{"role": "user", "content": "hello"}]
        reg.execute([{"name": "ctx_tool", "arguments": {}}], context=ctx)
        assert received_context["got"] is ctx
