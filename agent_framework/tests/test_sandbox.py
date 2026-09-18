"""Tests for Sandbox — Docker sandbox for agent tool execution.

Uses mock subprocess since Docker may not be running.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.sandbox import Sandbox, SandboxError

_FAKE_IMAGE = "python:3.12-slim"
_FAKE_REPO = "/home/xeio/dev/fonda/0"
_FAKE_WORKSPACE = "/tmp/agent_workspace"


def _make_sb() -> Sandbox:
    return Sandbox(
        image=_FAKE_IMAGE,
        repo_path=_FAKE_REPO,
        workspace_path=_FAKE_WORKSPACE,
    )


class TestBuildBaseArgs:
    def test_security_flags_present(self):
        sb = _make_sb()
        args = sb._build_base_args()
        assert "--network" in args
        assert "none" in args
        assert "--cap-drop" in args
        assert "ALL" in args
        assert "--security-opt" in args
        assert "no-new-privileges" in args
        assert "--read-only" in args

    def test_resource_limits_present(self):
        sb = _make_sb()
        args = sb._build_base_args()
        assert "--pids-limit" in args
        assert "--memory" in args
        assert "--cpus" in args

    def test_repo_mounted_readonly(self):
        sb = _make_sb()
        args = sb._build_base_args()
        assert "-v" in args
        assert f"{_FAKE_REPO}:/repo:ro" in args

    def test_workspace_bind_mount(self):
        sb = _make_sb()
        args = sb._build_base_args()
        # Workspace is a bind mount to host dir (persistent across calls)
        assert "-v" in args
        assert f"/tmp/agent_workspace:/tmp/workspace:rw" in args

    def test_user_flag(self):
        sb = _make_sb()
        args = sb._build_base_args()
        # Runs as host user, not root
        assert "--user" in args

    def test_extra_volume_mounts(self):
        sb = Sandbox(
            image=_FAKE_IMAGE,
            repo_path=_FAKE_REPO,
            volume_mounts=[("/host/data", "/container/data")],
        )
        args = sb._build_base_args()
        # Volume mounts are joined as "-v host:container"
        assert any("/container/data" in a for a in args)


class TestRun:
    def test_docker_not_found(self):
        sb = _make_sb()
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            import pytest

            with pytest.raises(SandboxError, match="docker is not installed"):
                sb._run("echo hi")

    def test_timeout(self):
        sb = _make_sb()
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=10),
        ):
            import pytest

            with pytest.raises(SandboxError, match="timed out"):
                sb._run("sleep 999", timeout=10)

    def test_command_failure_raises(self):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "something broke"
        with patch("subprocess.run", return_value=result):
            sb = _make_sb()
            import pytest

            with pytest.raises(SandboxError, match="something broke"):
                sb._run("bad_command")

    def test_command_success_returns_stdout(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "hello world"
        result.stderr = ""
        with patch("subprocess.run", return_value=result):
            sb = _make_sb()
            out = sb._run("echo hi")
            assert out == "hello world"


class TestExecutePython:
    def test_delegates_to_run(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "42\n"
        result.stderr = ""
        with patch.object(Sandbox, "_run", return_value=result.stdout) as mock_run:
            sb = _make_sb()
            out = sb.execute_python("print(42)")
            assert out == "42\n"
            # Verify _run was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "python" in call_args
            assert "_sandbox_script.py" in call_args


class TestExecuteShell:
    def test_delegates_to_run(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "/repo\n"
        result.stderr = ""
        with patch.object(Sandbox, "_run", return_value=result.stdout) as mock_run:
            sb = _make_sb()
            out = sb.execute_shell("ls /repo")
            assert out == "/repo\n"
            mock_run.assert_called_once()
            # Verify cd /repo prefix was added
            call_args = mock_run.call_args[0][0]
            assert "cd /repo &&" in call_args


class TestReadFile:
    def test_delegates_to_run(self):
        with patch.object(Sandbox, "_run", return_value="file contents") as mock_run:
            sb = _make_sb()
            out = sb.read_file("/repo/README.md")
            assert out == "file contents"
            mock_run.assert_called_once()


class TestWriteFile:
    def test_valid_path(self):
        with patch.object(Sandbox, "_run", return_value="") as mock_run:
            sb = _make_sb()
            ok = sb.write_file("/tmp/workspace/output.txt", "hello")
            assert ok is True
            mock_run.assert_called_once()
            # Verify the command contains mkdir and cat
            cmd = mock_run.call_args[0][0]
            assert "mkdir -p" in cmd
            assert "cat >" in cmd

    def test_nested_path_creates_dirs(self):
        with patch.object(Sandbox, "_run", return_value="") as mock_run:
            sb = _make_sb()
            ok = sb.write_file(
                "/tmp/workspace/subdir/nested.txt", "nested content"
            )
            assert ok is True
            cmd = mock_run.call_args[0][0]
            assert "/tmp/workspace/subdir" in cmd

    def test_path_traversal_rejected(self):
        sb = _make_sb()
        import pytest

        with pytest.raises(ValueError, match="must be within /tmp/workspace"):
            sb.write_file("/etc/passwd", "hacked")

    def test_path_traversal_dotslash(self):
        sb = _make_sb()
        import pytest

        with pytest.raises(ValueError, match="must be within /tmp/workspace"):
            sb.write_file("/tmp/workspace/../../../etc/passwd", "hacked")

    def test_path_prefix_collision_rejected(self):
        sb = _make_sb()
        with pytest.raises(ValueError, match="must be within /tmp/workspace"):
            sb.write_file("/tmp/workspace_evil/file.txt", "hacked")

    def test_symlink_escape_rejected(self, tmp_path):
        import os
        import shutil
        import tempfile

        os.makedirs("/tmp/workspace", exist_ok=True)
        workspace = tempfile.mkdtemp(prefix="test-sandbox-", dir="/tmp/workspace")
        workspace_path = os.path.abspath(workspace)
        try:
            link = os.path.join(workspace_path, "link")
            os.symlink(tmp_path / "outside", link)
            sb = _make_sb()
            with patch.object(Sandbox, "_run") as mock_run:
                with pytest.raises(ValueError, match="must be within /tmp/workspace"):
                    sb.write_file(os.path.join(link, "file.txt"), "hacked")
                mock_run.assert_not_called()
        finally:
            shutil.rmtree(workspace_path, ignore_errors=True)

        with patch.object(Sandbox, "_run", return_value="") as mock_run:
            sb = _make_sb()
            content = "some content to write"
            sb.write_file("/tmp/workspace/data.txt", content)
            # Verify stdin_data was passed
            assert mock_run.call_args[1]["stdin_data"] == content


class TestInit:
    def test_expands_tilde_in_repo_path(self):
        import os

        sb = Sandbox(repo_path="~/dev/fonda/0")
        assert sb.repo_path == os.path.expanduser("~/dev/fonda/0")

    def test_allows_repo_same_as_workspace(self):
        """Repo equal to workspace is allowed."""
        sb = Sandbox(
            image="python:3.12-slim",
            repo_path="/tmp/test_workspace",
            workspace_path="/tmp/test_workspace",
        )
        assert sb.repo_path == "/tmp/test_workspace"

    def test_rejects_repo_higher_than_workspace(self):
        """Repo higher than workspace is rejected."""
        with pytest.raises(SandboxError, match="mounted higher"):
            Sandbox(
                image="python:3.12-slim",
                repo_path="/tmp",
                workspace_path="/tmp/test_workspace",
            )

    def test_rejects_root_mount(self):
        """Mounting / as repo is always rejected."""
        with pytest.raises(SandboxError, match="Cannot mount / as repo"):
            Sandbox(
                image="python:3.12-slim",
                repo_path="/",
                workspace_path="/tmp/test_workspace",
            )

    def test_rejects_home_mount(self):
        """Mounting /home as repo when workspace is deeper is rejected."""
        with pytest.raises(SandboxError, match="mounted higher"):
            Sandbox(
                image="python:3.12-slim",
                repo_path="/home/xeio",
                workspace_path="/home/xeio/dev/fonda/0",
            )

    def test_allows_unrelated_paths(self):
        """Unrelated paths at same depth are allowed."""
        sb = Sandbox(
            image="python:3.12-slim",
            repo_path="/home/xeio/dev/fonda/0",
            workspace_path="/tmp/agent_workspace",
        )
        assert sb.repo_path == "/home/xeio/dev/fonda/0"

    def test_expands_tilde(self):
        """Tilde paths are expanded before checking."""
        import os

        sb = Sandbox(
            image="python:3.12-slim",
            repo_path="~/dev/fonda/0",
            workspace_path="/tmp/test_workspace",
        )
        assert sb.repo_path.startswith(os.path.expanduser("~"))

    def test_defaults(self):
        sb = Sandbox(
            repo_path="/repo",
            workspace_path="/tmp/ws",
        )
        assert sb.image == "python:3.12-slim"
        assert sb.workspace_path == "/tmp/ws"
        assert sb.volume_mounts == []
