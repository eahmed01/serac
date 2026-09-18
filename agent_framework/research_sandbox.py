#!/usr/bin/env python3
"""Adapter for the bundled sandbox server (agent_framework.sandbox).

Connects to the running sandbox server via TCP or Unix socket and provides
a synchronous interface compatible with agent_framework tool executors.

When running in Docker:
  - The sandbox server listens on a Unix socket at /tmp/agent_workspace/sandbox.sock
  - The adapter connects via the shared volume mount
  - No network access required (both are in network-isolated containers)

When running on host:
  - The sandbox server listens on localhost:9876
  - The adapter connects via TCP

The persistent sandbox has:
- `s`: Sandbox instance with loaded OHLCV data
- `df`: OHLCV DataFrame (from s.ohlcv)
- `np`, `pd`: numpy and pandas
- `check_causality`: causality checker function
- Any variables created via previous execute() calls

Usage:
    from agent_framework.research_sandbox import ResearchSandbox

    rs = ResearchSandbox()  # Auto-detects TCP vs Unix socket
    output = rs.execute("df.head()")
    output = rs.evaluate("len(df)")

    # Check if server is running
    print(rs.is_available)  # True if server responds
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Optional

logger = logging.getLogger(__name__)

# Default connection params
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
DEFAULT_SOCKET_PATH = "/tmp/agent_workspace/sandbox.sock"
DEFAULT_TIMEOUT = 120  # seconds


class ResearchSandboxError(RuntimeError):
    """Raised when the research sandbox is unavailable or an operation fails."""


class ResearchSandbox:
    """Synchronous adapter for the bundled sandbox server (agent_framework.sandbox).

    Connects to the sandbox server via TCP or Unix socket.
    Provides execute(), evaluate(), list_vars(), get_var_meta() methods
    that match the pattern expected by agent_framework tool executors.

    Connections are created lazily on first use and reused.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        socket_path: str = DEFAULT_SOCKET_PATH,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._host = host
        self._port = port
        self._socket_path = socket_path
        self._timeout = timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_available(self) -> bool:
        """Check if the sandbox server is running and responding."""
        # Try Unix socket first
        if os.path.exists(self._socket_path):
            try:
                import socket as _socket
                sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(self._socket_path)
                sock.close()
                return True
            except (ConnectionRefusedError, OSError):
                pass

        # Fall back to TCP
        try:
            import socket as _socket
            sock = _socket.create_connection((self._host, self._port), timeout=2)
            sock.close()
            return True
        except (ConnectionRefusedError, OSError):
            return False

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for async calls."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def execute(self, code: str) -> str:
        """Execute Python code in the persistent sandbox namespace.

        Args:
            code: Python code string to execute.

        Returns:
            stdout from the execution, or error message on failure.
        """
        loop = self._get_loop()
        try:
            return loop.run_until_complete(self._execute_async(code))
        except Exception as e:
            raise ResearchSandboxError(f"Sandbox execute failed: {e}")

    async def _execute_async(self, code: str) -> str:
        """Async implementation of execute."""
        from agent_framework.sandbox.client import SandboxClient, ConnectionClosed

        try:
            async with SandboxClient(
                host=self._host, port=self._port, timeout=self._timeout
            ) as client:
                resp = await client.execute(code)
                if not resp.success:
                    return f"ERROR: {resp.error}"
                return resp.output or "(no output)"
        except ConnectionClosed as e:
            raise ResearchSandboxError(
                f"Connection lost: {e}. Is the sandbox server running?"
            )
        except OSError as e:
            raise ResearchSandboxError(
                f"Cannot connect to sandbox server at {self._host}:{self._port}: {e}"
            )

    def evaluate(self, expression: str) -> str:
        """Evaluate a Python expression in the sandbox namespace.

        Args:
            expression: Python expression to evaluate.

        Returns:
            String representation of the result, or error message.
        """
        loop = self._get_loop()
        try:
            return loop.run_until_complete(self._eval_async(expression))
        except Exception as e:
            raise ResearchSandboxError(f"Sandbox eval failed: {e}")

    async def _eval_async(self, expression: str) -> str:
        """Async implementation of evaluate."""
        from agent_framework.sandbox.client import SandboxClient, ConnectionClosed

        try:
            async with SandboxClient(
                host=self._host, port=self._port, timeout=self._timeout
            ) as client:
                resp = await client.eval(expression)
                if not resp.success:
                    return f"ERROR: {resp.error}"
                return resp.output or "(no result)"
        except ConnectionClosed as e:
            raise ResearchSandboxError(
                f"Connection lost: {e}. Is the sandbox server running?"
            )

    def list_vars(self) -> list[str]:
        """List all user-defined variables in the namespace."""
        loop = self._get_loop()
        return loop.run_until_complete(self._list_vars_async())

    async def _list_vars_async(self) -> list[str]:
        from agent_framework.sandbox.client import SandboxClient, ConnectionClosed

        try:
            async with SandboxClient(
                host=self._host, port=self._port, timeout=self._timeout
            ) as client:
                resp = await client.list_vars()
                if not resp.success:
                    return []
                return resp.variables
        except (ConnectionClosed, OSError):
            return []

    def get_var_meta(self, name: str) -> dict:
        """Get metadata for a variable (type, shape, dtype)."""
        loop = self._get_loop()
        return loop.run_until_complete(self._get_var_meta_async(name))

    async def _get_var_meta_async(self, name: str) -> dict:
        from agent_framework.sandbox.client import SandboxClient, ConnectionClosed

        try:
            async with SandboxClient(
                host=self._host, port=self._port, timeout=self._timeout
            ) as client:
                resp = await client.get_var(name)
                if not resp.success:
                    return {"error": resp.error}
                return {
                    "name": resp.var_name,
                    "type": resp.var_type,
                    "shape": resp.shape,
                }
        except (ConnectionClosed, OSError):
            return {"error": "connection failed"}
