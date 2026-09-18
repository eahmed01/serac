"""Regression tests for workspace-confined consultant attachments."""

from __future__ import annotations

import pytest

import agent_framework.consult as consult_module


class _FakeLoop:
    total_usage = None
    turn_count = 1
    messages: list[str] = []
    system_prompt: str = ""

    def __init__(self, **kwargs):
        self.__class__.system_prompt = kwargs["system_prompt"]

    def run(self, message):
        self.__class__.messages.append(message)
        return "done"


def _run_consult(monkeypatch, workspace, attachments, sandbox=None):
    _FakeLoop.messages = []
    monkeypatch.setattr(
        consult_module, "create_provider", lambda model, max_tokens: object()
    )
    monkeypatch.setattr(consult_module, "AgentLoop", _FakeLoop)
    result = consult_module.consult(
        goal="inspect",
        workspace=workspace,
        attach_files=[str(path) for path in attachments],
        sandbox=sandbox,
        max_turns=1,
    )
    assert result["success"] is True
    return _FakeLoop.messages[-1]


def test_attachments_resolve_relative_to_workspace_and_accept_absolute(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    file_path = workspace / "docs" / "a.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("attachment content")

    message = _run_consult(
        monkeypatch,
        workspace,
        ["docs/a.md", file_path],
    )

    assert message.count("attachment content") == 2
    assert "=== ATTACHED FILE: docs/a.md ===" in message
    assert str(file_path) not in message


@pytest.mark.parametrize("attachment", ["../outside.txt", "outside.txt"])
def test_attachments_reject_traversal_and_prefix_collision(tmp_path, monkeypatch, attachment):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be read")
    prefix_collision = tmp_path / "workspace_evil" / "outside.txt"
    prefix_collision.parent.mkdir()
    prefix_collision.write_text("must not be read")

    requested = attachment
    if attachment == "outside.txt":
        requested = str(prefix_collision)
    message = _run_consult(monkeypatch, workspace, [requested])

    assert "must not be read" not in message
    assert "outside workspace" in message
    assert str(tmp_path) not in message
    assert requested not in message


def test_attachment_rejects_symlink_escape_without_disclosing_host_path(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("symlink secret")
    link = workspace / "docs-link"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    message = _run_consult(monkeypatch, workspace, ["docs-link"])

    assert "symlink secret" not in message
    assert "outside workspace" in message
    assert str(outside) not in message


def test_sandbox_attachment_label_uses_container_path(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    file_path = workspace / "docs" / "a.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("sandbox attachment")

    message = _run_consult(monkeypatch, workspace, [file_path], sandbox=object())

    assert "=== ATTACHED FILE: /repo/docs/a.md ===" in message
    assert "sandbox attachment" in message
    assert str(file_path) not in message
