"""Tests for sandbox-aware tool loading."""

import pytest

import agent_framework.consult as consult_module
import agent_framework.builtins as builtins_module
from agent_framework.builtins import (
    _guard_read_path,
    _validate_write_path,
    execute_python_factory,
    execute_terminal_factory,
    patch_file_factory,
    read_file_factory,
    register_builtin_tools,
    write_file_factory,
)
from agent_framework.consult import load_workspace_tools
from agent_framework.tool_loader import ToolLoader, load_tools_from_config


class FakeSandbox:
    """Marker-only sandbox; no Docker or tool execution is needed here."""


class FalseySandbox:
    def __bool__(self):
        return False

    def read_file(self, path):
        return "old"

    def write_file(self, path, content):
        self.written = (path, content)

    def execute_shell(self, command, timeout=180):
        return "sandbox shell"

    def execute_python(self, code, timeout=60):
        return "sandbox python"


def test_falsey_sandbox_is_used_by_all_sandbox_executors():
    sandbox = FalseySandbox()
    assert read_file_factory(sandbox).executor("ignored") == "old"
    assert write_file_factory(sandbox).executor("ignored", "content").startswith("Written")
    assert patch_file_factory(sandbox).executor("ignored", "old", "new") == "Patched ignored"
    assert sandbox.written == ("ignored", "new")
    assert execute_terminal_factory(sandbox).executor("true") == "sandbox shell"
    assert execute_python_factory(sandbox).executor("print('x')") == "sandbox python"


def test_loader_omits_requires_sandbox_tools_from_non_sandbox_defaults():
    registry = ToolLoader().load()

    assert "execute_terminal" not in registry.tools
    assert "execute_python" not in registry.tools
    assert {"read_file", "write_file", "patch_file"}.issubset(registry.tools)


def test_loader_rejects_explicit_execute_terminal_without_sandbox():
    with pytest.raises(ValueError, match="execute_terminal.*requires a sandbox"):
        ToolLoader().load(tool_names=["execute_terminal"])


def test_loader_rejects_explicit_execute_python_without_sandbox():
    with pytest.raises(ValueError, match="execute_python.*requires a sandbox"):
        ToolLoader().load(tool_names=["execute_python"])


def test_loader_loads_explicit_execution_tools_with_sandbox():
    registry = ToolLoader(sandbox=FakeSandbox()).load(
        tool_names=["execute_terminal", "execute_python"]
    )

    assert set(registry.tools) == {"execute_terminal", "execute_python"}
    assert all(tool.requires_sandbox for tool in registry.tools.values())


def test_sandbox_loader_filters_host_only_default_tools():
    registry = ToolLoader(sandbox=FakeSandbox()).load()

    assert set(registry.tools) == {
        "read_file",
        "write_file",
        "patch_file",
        "execute_terminal",
        "execute_python",
    }
    assert all(tool.execution_mode == "sandbox" for tool in registry.tools.values())


def test_sandbox_loader_rejects_explicit_host_only_request():
    with pytest.raises(ValueError, match="code_search.*host-only"):
        ToolLoader(sandbox=FakeSandbox()).load(tool_names=["code_search"])


def test_sandbox_loader_filters_defaults_with_safe_override():
    registry = ToolLoader(sandbox=FakeSandbox()).load(
        overrides={"read_file": "agent_framework.builtins:read_file_factory"}
    )

    assert "read_file" in registry.tools
    assert "code_search" not in registry.tools


def test_sandbox_loader_rejects_host_only_override():
    with pytest.raises(ValueError, match="code_search.*host-only"):
        ToolLoader(sandbox=FakeSandbox()).load(
            overrides={"code_search": "agent_framework.builtins:code_search_factory"}
        )


def test_config_loader_skips_host_only_tools_with_sandbox():
    registry = load_tools_from_config(
        {
            "read_file": "agent_framework.builtins:read_file_factory",
            "git_status": "agent_framework.builtins:git_status_factory",
        },
        sandbox=FakeSandbox(),
    )

    assert set(registry.tools) == {"read_file"}


def test_loader_preserves_host_tools_without_sandbox():
    registry = ToolLoader().load(tool_names=["code_search", "git_status"])

    assert set(registry.tools) == {"code_search", "git_status"}
    assert all(tool.execution_mode == "host" for tool in registry.tools.values())


def test_workspace_tools_omitted_names_filter_host_only_for_sandbox():
    registry = load_workspace_tools("/tmp", sandbox=FakeSandbox())

    assert set(registry.tools) == {"read_file"}


def test_workspace_tools_explicit_sandbox_write_name_is_allowed():
    registry = load_workspace_tools("/tmp", tool_names=["write_file"], sandbox=FakeSandbox())

    assert set(registry.tools) == {"write_file"}


def test_workspace_tools_omitted_names_preserve_safe_defaults_without_sandbox(tmp_path):
    registry = load_workspace_tools(tmp_path)

    assert set(registry.tools) == {
        "read_file", "find_files", "code_search", "web_search", "sec_search", "sec_fetch"
    }


def test_workspace_tools_empty_explicit_list_loads_no_tools():
    registry = load_workspace_tools("/tmp", tool_names=[])

    assert registry.tools == {}


def test_workspace_tools_explicit_sec_tools_are_authoritative():
    registry = load_workspace_tools("/tmp", tool_names=["sec_search", "sec_fetch"])

    assert set(registry.tools) == {"sec_search", "sec_fetch"}


def test_workspace_tools_explicit_host_only_name_raises():
    with pytest.raises(ValueError, match="code_search.*host-only"):
        load_workspace_tools("/tmp", tool_names=["code_search"], sandbox=FakeSandbox())


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("code_search", {"pattern": "secret", "path": "/etc"}),
        ("code_search", {"pattern": "secret", "path": "../outside"}),
        ("find_files", {"pattern": "*", "path": "/etc"}),
        ("find_files", {"pattern": "*", "path": "../outside"}),
    ],
)
def test_workspace_tools_reject_host_search_escape_without_running_command(
    tmp_path, monkeypatch, tool_name, arguments
):
    calls = []
    monkeypatch.setattr(
        builtins_module, "_run_command", lambda *args, **kwargs: calls.append(args) or ""
    )
    registry = load_workspace_tools(tmp_path / "workspace", tool_names=[tool_name])

    result = registry.tools[tool_name].executor(**arguments)

    assert "outside the workspace" in result
    assert calls == []


def test_workspace_tools_reject_symlink_escape(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    calls = []
    monkeypatch.setattr(
        builtins_module, "_run_command", lambda *args, **kwargs: calls.append(args) or ""
    )

    registry = load_workspace_tools(workspace, tool_names=["code_search"])
    result = registry.tools["code_search"].executor(pattern="secret", path="link")

    assert "outside the workspace" in result
    assert calls == []


@pytest.mark.parametrize("tool_name", ["code_search", "find_files"])
def test_workspace_search_tools_resolve_relative_path_from_workspace(
    tmp_path, monkeypatch, tool_name
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return b"inside\0" if tool_name == "find_files" else "inside"

    monkeypatch.setattr(builtins_module, "_run_command", fake_run)
    registry = load_workspace_tools(workspace, tool_names=[tool_name])
    arguments = {
        "pattern": "needle" if tool_name == "code_search" else "*.py",
        "path": "nested",
    }

    assert registry.tools[tool_name].executor(**arguments) == "inside"
    assert calls
    path_index = -1 if tool_name == "code_search" else 1
    assert calls[0][path_index] == str(workspace / "nested")


def test_workspace_read_file_resolves_relative_path(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello\n")
    # The built-in read executor has its separate project-root guard; this
    # test isolates the workspace wrapper's path resolution.
    monkeypatch.setattr(builtins_module, "_guard_read_path", lambda path: None)

    registry = load_workspace_tools(workspace, tool_names=["read_file"])

    assert "hello" in registry.tools["read_file"].executor(path="note.txt")


def test_consult_prompt_advertises_only_loaded_sandbox_tools(monkeypatch):
    captured = {}

    class FakeLoop:
        total_usage = None
        turn_count = 1

        def __init__(self, **kwargs):
            captured["system_prompt"] = kwargs["system_prompt"]

        def run(self, message):
            return "done"

    monkeypatch.setattr(consult_module, "create_provider", lambda model, max_tokens: object())
    monkeypatch.setattr(consult_module, "AgentLoop", FakeLoop)

    result = consult_module.consult(
        goal="inspect",
        workspace="/tmp",
        sandbox=FakeSandbox(),
        max_turns=1,
    )

    assert result["success"] is True
    assert "read_file" in captured["system_prompt"]
    assert "find_files, code_search, web_search" not in captured["system_prompt"]


def test_legacy_registration_omits_sandbox_required_tools_without_sandbox():
    registry = builtins_module.ToolRegistry()

    with pytest.warns(DeprecationWarning):
        register_builtin_tools(registry)

    assert "execute_terminal" not in registry.tools
    assert "execute_python" not in registry.tools


def test_legacy_registration_with_sandbox_omits_host_only_tools():
    registry = builtins_module.ToolRegistry()

    with pytest.warns(DeprecationWarning):
        register_builtin_tools(registry, sandbox=FakeSandbox())

    assert set(registry.tools) == {
        "read_file",
        "write_file",
        "patch_file",
        "execute_terminal",
        "execute_python",
    }
    assert not {
        "code_search",
        "find_files",
        "git_status",
        "git_diff",
        "git_log",
        "web_search",
        "todo",
    } & set(registry.tools)
    assert all(tool.execution_mode == "sandbox" for tool in registry.tools.values())


def test_direct_path_guards_reject_cwd_prefix_collision_and_allow_in_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    prefix_sibling = tmp_path / "project-sibling"
    root.mkdir()
    prefix_sibling.mkdir()
    inside = root / "nested" / "new.txt"
    inside.parent.mkdir()
    inside.write_text("safe\n")

    monkeypatch.chdir(root)
    _guard_read_path(inside)
    _validate_write_path(inside)

    with pytest.raises(PermissionError):
        _guard_read_path(prefix_sibling / "secret.txt")
    with pytest.raises(PermissionError):
        _validate_write_path(prefix_sibling / "new.txt")


def test_direct_path_guards_reject_symlink_escape_for_read_and_write(tmp_path, monkeypatch):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n")
    (root / "secret-link").symlink_to(outside / "secret.txt")
    (root / "out-link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)

    with pytest.raises(PermissionError):
        read_file_factory().executor("secret-link")
    with pytest.raises(PermissionError):
        write_file_factory().executor("out-link/new.txt", "escape")
