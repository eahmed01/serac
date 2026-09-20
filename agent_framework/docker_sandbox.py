"""Docker sandbox for agent tool execution with security constraints.

All tool execution happens inside a Docker sandbox. The sandbox mounts
the project repo read-only at /repo and provides a writable workspace
at /tmp/workspace. The host tool layer mediates all writes back to the host.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Default resource limits
_DEFAULT_PIDS_LIMIT = 100
_DEFAULT_MEMORY = "1g"
_DEFAULT_CPUS = "1.0"
_DEFAULT_TIMEOUT = 60
_DEFAULT_IMAGE = "python:3.12-slim"

# Default repo path: pre-created sample workspace (small, read-only)
_DEFAULT_REPO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "default_workspace",
)


class SandboxError(RuntimeError):
    """Raised when a sandboxed command fails."""


class Sandbox:
    """Docker sandbox for agent tool execution.

    Security constraints:
    - --network none (no outbound network access)
    - Non-root user
    - Read-only root FS with only /tmp/workspace writable
    - --cap-drop=ALL, --security-opt=no-new-privileges
    - Resource limits: --pids-limit, memory, CPU
    - Ephemeral container per operation

    By default, mounts the pre-created sample workspace at
    agent_framework/default_workspace/ (small, safe subset).
    Pass ``repo_path`` to mount a different directory.

    Args:
        image: Docker image to use (default: python:3.12-slim)
        repo_path: Path to mount read-only at /repo.
            Defaults to agent_framework/default_workspace/
        workspace_path: Path to writable workspace (mounted at /tmp/workspace)
        volume_mounts: Extra volume mounts as list of (host, container) tuples
    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        repo_path: Optional[str] = None,
        workspace_path: str = "/tmp/agent_workspace",
        volume_mounts: Optional[list[tuple[str, str]]] = None,
    ):
        self.image = image
        # Default to the pre-created sample workspace in the repo
        self.repo_path = os.path.abspath(
            os.path.expanduser(repo_path or _DEFAULT_REPO)
        )
        self.workspace_path = workspace_path
        self.volume_mounts = volume_mounts or []

        # P2: Mount depth security check
        # The repo path must not be mounted higher (closer to root) than the
        # workspace. If workspace is /home/user/project, repo cannot be /home,
        # /, etc. This prevents accidentally exposing large portions of the
        # filesystem to the container.
        self._check_mount_depth()

    def _check_mount_depth(self) -> None:
        """Validate that repo_path is not mounted higher than workspace_path.

        The workspace path defines the minimum allowed depth for the repo mount.
        If workspace is /tmp/agent_workspace, repo cannot be /tmp, /, etc.
        If workspace is /home/user/project, repo cannot be /home or /.

        This prevents accidentally exposing large portions of the filesystem
        to the container.

        Raises:
            SandboxError: If repo_path is higher than workspace_path.
        """
        # Resolve both paths to absolute
        repo = os.path.abspath(self.repo_path)
        workspace = os.path.abspath(self.workspace_path)

        # If repo is "/" it's always too broad — check before any normalization
        if repo == "/":
            raise SandboxError(
                "Cannot mount / as repo. This would expose the entire filesystem "
                "to the container. Use a more specific path."
            )

        # Normalize to handle trailing slashes
        repo_norm = repo.rstrip("/")
        workspace_norm = workspace.rstrip("/")

        # Check if repo is higher than workspace
        if (repo_norm != workspace_norm and
            workspace_norm.startswith(repo_norm + "/")):
            raise SandboxError(
                f"Security violation: repo_path {repo!r} is mounted higher "
                f"than workspace_path {workspace!r}. The repo path must be at "
                f"the same depth or deeper than the workspace to limit filesystem "
                f"exposure."
            )

    def _build_base_args(self) -> list[str]:
        """Build the common docker run arguments with security constraints."""
        # Ensure workspace directory exists on host
        os.makedirs(self.workspace_path, exist_ok=True)

        args = [
            "docker", "run", "-i", "--rm",
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "-v", f"{self.repo_path}:/repo:ro",
            "-v", f"{self.workspace_path}:/tmp/workspace:rw",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--pids-limit", str(_DEFAULT_PIDS_LIMIT),
            "--memory", _DEFAULT_MEMORY,
            "--cpus", _DEFAULT_CPUS,
        ]
        for host_path, container_path in self.volume_mounts:
            # P2 fix: Default to read-only mounts for security
            if container_path.startswith("/tmp/"):
                args.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}:rw"])
            else:
                args.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}:ro"])
        return args

    def _run(
        self,
        command: str,
        timeout: int = _DEFAULT_TIMEOUT,
        stdin_data: Optional[str] = None,
    ) -> str:
        """Run a command inside the sandbox and return stdout.

        Args:
            command: The command string to execute inside the container.
            timeout: Maximum seconds before the process is killed.
            stdin_data: Optional data to pipe to stdin.

        Returns:
            stdout as a string.

        Raises:
            SandboxError: If docker is not available or the command fails.
        """
        args = self._build_base_args()
        args.extend([self.image, "sh", "-c", command])

        try:
            result = subprocess.run(
                args,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise SandboxError("docker is not installed or not in PATH")
        except subprocess.TimeoutExpired:
            raise SandboxError(
                f"Sandbox command timed out after {timeout}s: {command}"
            )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            logger.warning("Sandbox command failed: %s", stderr)
            raise SandboxError(
                f"Sandbox command failed (exit {result.returncode}): {stderr}"
            )

        return result.stdout

    def execute_python(self, code: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        """Execute Python code in the sandbox and return stdout.

        Code runs with /repo as the working directory (the project root).

        Args:
            code: Python code string.
            timeout: Max seconds to run.

        Returns:
            stdout from the execution.

        Raises:
            SandboxError: If execution fails or times out.
        """
        script_path = "/tmp/workspace/_sandbox_script.py"
        escaped_code = code.replace("'", "'\\''")
        command = (
            f"cat > {script_path} << 'SERAC_SANDBOX_EOF'\n"
            f"{code}\n"
            f"SERAC_SANDBOX_EOF\n"
            f"cd /repo && python {script_path}"
        )
        return self._run(command, timeout=timeout)

    def execute_shell(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command in the sandbox and return stdout.

        Commands run with /repo as the working directory (the project root).

        Args:
            command: Shell command string.
            timeout: Max seconds to run.

        Returns:
            stdout from the execution.

        Raises:
            SandboxError: If execution fails or times out.
        """
        # Prepend cd /repo so relative paths work
        full_command = f"cd /repo && {command}"
        return self._run(full_command, timeout=timeout)

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox (within /repo or /tmp/workspace).

        Relative paths are resolved against /repo (the mounted project root).
        Absolute host paths are translated to container paths.

        Args:
            path: File path — absolute or relative to /repo.

        Returns:
            File contents as a string.

        Raises:
            SandboxError: If the file doesn't exist or read fails.
        """
        # Translate absolute host paths to container paths
        # The repo is mounted at /repo, so /home/user/project/foo.py -> /repo/foo.py
        container_path = path
        if path.startswith("/"):
            # Check if this is an absolute path within the repo
            repo_prefix = self.repo_path.rstrip("/") + "/"
            if path == self.repo_path or path.startswith(repo_prefix):
                # Strip the repo path prefix and prepend /repo
                container_path = "/repo" + path[len(self.repo_path):]
            elif path.startswith("/tmp"):
                # /tmp paths map directly
                container_path = path
            else:
                # For other absolute paths, try as-is (may fail)
                container_path = path
        elif not path.startswith("/repo"):
            # Relative paths go under /repo
            container_path = f"/repo/{path}"

        escaped = container_path.replace("'", "'\\''")
        return self._run(f"cat '{escaped}'")

    def write_file(self, path: str, content: str) -> bool:
        """Write a file in the sandbox workspace.

        Path must be within /tmp/workspace or will be rejected.

        Args:
            path: File path within /tmp/workspace.
            content: File contents.

        Returns:
            True if successful.

        Raises:
            ValueError: If path escapes /tmp/workspace.
            SandboxError: If the write fails.
        """
        # Path traversal guard
        workspace_root = os.path.realpath("/tmp/workspace")
        normalized = os.path.realpath(os.path.normpath(path))
        try:
            within_workspace = os.path.commonpath([workspace_root, normalized]) == workspace_root
        except ValueError:
            within_workspace = False
        if not within_workspace:
            raise ValueError(
                f"Path must be within /tmp/workspace, got: {path}"
            )

        def _sh_escape(s: str) -> str:
            """Escape a string for safe use in single-quoted shell."""
            return s.replace("'", "'\\''")

        escaped = _sh_escape(normalized)
        dir_part = os.path.dirname(normalized)
        escaped_dir = _sh_escape(dir_part)
        command = f"mkdir -p '{escaped_dir}' && cat > '{escaped}'"
        self._run(command, stdin_data=content)
        return True
