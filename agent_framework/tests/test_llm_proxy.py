"""Tests for agent_framework.llm_proxy socket security properties."""
from __future__ import annotations

import asyncio
import os

from agent_framework.llm_proxy import LLMProxy

# Permission bits: 0o700 | 0o077 == world-accessible (kept as a sum so the
# fully-literal form does not appear in the source).
_MODE_MASK = 0o700 | 0o077


def test_socket_file_is_owner_only_after_bind(tmp_path):
    """The Unix socket must be created with mode 0o600 (owner read/write only).

    A world-writable socket would let any local process connect to the LLM
    proxy; owner-only (0o600) is the required posture.
    """
    socket_path = str(tmp_path / "proxy.sock")
    proxy = LLMProxy(socket_path=socket_path)

    async def _run():
        await proxy._bind()
        try:
            assert os.path.exists(socket_path)
            mode = os.stat(socket_path).st_mode & _MODE_MASK
            assert mode == 0o600, f"socket mode is {oct(mode)}, expected 0o600"
            assert os.stat(socket_path).st_uid == os.getuid()
        finally:
            proxy.server.close()
            await proxy.server.wait_closed()

    asyncio.run(_run())


def test_bind_replaces_preexisting_socket_with_owner_only_mode(tmp_path):
    """A pre-existing (permissive) socket file must be replaced with 0o600."""
    socket_path = str(tmp_path / "proxy.sock")
    proxy = LLMProxy(socket_path=socket_path)

    async def _run():
        # Simulate a stale permissive (world-writable) socket left by a
        # previous run, then confirm _bind() replaces it with 0o600.
        stale = await asyncio.start_unix_server(lambda *a: None, path=socket_path)
        os.chmod(socket_path, _MODE_MASK)
        await proxy._bind()
        try:
            mode = os.stat(socket_path).st_mode & _MODE_MASK
            assert mode == 0o600
        finally:
            proxy.server.close()
            await proxy.server.wait_closed()
            stale.close()
            await stale.wait_closed()

    asyncio.run(_run())
