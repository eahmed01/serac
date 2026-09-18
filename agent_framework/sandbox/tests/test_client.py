"""
Client tests — async sandbox client for programmatic execution.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from agent_framework.sandbox.client import SandboxClient
from agent_framework.sandbox.server import SandboxServer
from agent_framework.sandbox.transport import ConnectionClosed


@pytest.fixture
async def running_server():
    """Start and stop a sandbox server for testing."""
    server = SandboxServer(port=9890)
    await server.start()
    yield server
    await server.stop()


class TestSandboxClient:

    async def test_connect(self, running_server):
        """Client should connect to server."""
        client = SandboxClient(host="127.0.0.1", port=9890)
        await client.connect()
        assert client.is_connected
        await client.close()

    async def test_execute(self, running_server):
        """Execute code on server."""
        client = SandboxClient(host="127.0.0.1", port=9890)
        await client.connect()
        try:
            resp = await client.execute("x = 42")
            assert resp.success
            # Verify state persisted
            resp2 = await client.execute("print(x * 2)")
            assert "84" in resp2.output
        finally:
            await client.close()

    async def test_eval(self, running_server):
        """Evaluate expression on server."""
        client = SandboxClient(host="127.0.0.1", port=9890)
        await client.connect()
        try:
            await client.execute("y = [1, 2, 3]")
            resp = await client.eval("len(y)")
            assert resp.success
            assert "3" in resp.output
        finally:
            await client.close()

    async def test_list_variables(self, running_server):
        """List variables on server."""
        client = SandboxClient(host="127.0.0.1", port=9890)
        await client.connect()
        try:
            await client.execute("alpha = 1")
            await client.execute("beta = 2")
            resp = await client.list_vars()
            assert resp.success
            assert "alpha" in resp.variables
            assert "beta" in resp.variables
        finally:
            await client.close()

    async def test_get_var_meta(self, running_server):
        """Get variable metadata."""
        client = SandboxClient(host="127.0.0.1", port=9890)
        await client.connect()
        try:
            await client.execute("df = pd.DataFrame({'a': [1,2,3]})")
            resp = await client.get_var("df")
            assert resp.success
            assert resp.shape == [3, 1]
        finally:
            await client.close()

    async def test_context_manager(self, running_server):
        """Context manager should handle connect/close."""
        async with SandboxClient(host="127.0.0.1", port=9890) as client:
            resp = await client.execute("z = 100")
            assert resp.success

    async def test_execute_error(self, running_server):
        """Execute error should return failure."""
        client = SandboxClient(host="127.0.0.1", port=9890)
        await client.connect()
        try:
            resp = await client.execute("raise ValueError('test')")
            assert not resp.success
            assert "ValueError" in resp.error
        finally:
            await client.close()


class TestClientTimeoutAndErrorHandling:
    """Tests for timeout enforcement and transport error handling.

    Note: Uses mock-based tests because asyncio.wait_for() cannot cancel
    StreamReader.readexactly() in Python 3.12 — a known asyncio limitation
    with TCP transports. The timeout logic is correct (asyncio.wait_for wraps
    the recv coroutine), but it requires the underlying coroutine to be
    cancellable, which readexactly() is not with active TCP connections.
    """

    async def test_timeout_enforcement_raises_on_hung_server(self):
        """Client should raise asyncio.TimeoutError when recv takes too long."""
        client = SandboxClient(timeout=0.5)
        client._transport = AsyncMock()
        async def slow_recv():
            await asyncio.sleep(60)
        client._transport.recv = slow_recv

        with patch.object(client._transport, 'send', new=AsyncMock()):
            with pytest.raises(asyncio.TimeoutError):
                await client.execute("x = 1")

    async def test_connection_error_normalized_to_connection_closed(self):
        """When transport raises ConnectionClosed, client should propagate it."""
        client = SandboxClient()
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv.side_effect = ConnectionClosed("test")

        with pytest.raises(ConnectionClosed, match="test"):
            await client.execute("x = 1")

    async def test_transport_os_error_normalized(self):
        """When transport raises OSError, client normalizes to ConnectionClosed."""
        client = SandboxClient()
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv.side_effect = ConnectionResetError("broken")

        with pytest.raises(ConnectionClosed):
            await client.execute("x = 1")

    async def test_timeout_propagates_to_transport_layer(self):
        """Timeout on execute() should wrap recv via asyncio.wait_for()."""
        client = SandboxClient(timeout=30.0)
        client._transport = AsyncMock()
        async def slow_recv():
            await asyncio.sleep(60)
        client._transport.recv = slow_recv

        with patch.object(client._transport, 'send', new=AsyncMock()):
            start = asyncio.get_event_loop().time()
            with pytest.raises(asyncio.TimeoutError):
                await client.execute("pass", timeout=0.1)
            elapsed = asyncio.get_event_loop().time() - start
            assert elapsed < 2.0, f"Timeout took {elapsed:.2f}s — should be ~0.1s"

    async def test_timeout_none_disables_guard(self):
        """When timeout is None on both instance and call, no wait_for()."""
        client = SandboxClient(timeout=None)
        client._transport = AsyncMock()
        async def mock_recv():
            from agent_framework.sandbox.transport import Message
            return Message("result", success=True, output="ok")
        client._transport.recv = mock_recv

        resp = await client.execute("pass", timeout=None)
        assert resp.success

    async def test_timeout_zero_raises_value_error(self):
        """timeout=0 should raise ValueError, not asyncio.TimeoutError."""
        client = SandboxClient(timeout=0)
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv = AsyncMock()

        with pytest.raises(ValueError, match="timeout must be > 0 or None"):
            await client.execute("x = 1")

    async def test_timeout_negative_raises_value_error(self):
        """timeout=-1 should raise ValueError, not an asyncio internal error."""
        client = SandboxClient(timeout=-1)
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv = AsyncMock()

        with pytest.raises(ValueError, match="timeout must be > 0 or None"):
            await client.execute("x = 1")

    async def test_per_call_timeout_none_uses_instance_timeout(self):
        """When per-call timeout=None and instance timeout=30.0, effective timeout is 30.0."""
        client = SandboxClient(timeout=30.0)
        client._transport = AsyncMock()

        # Mock recv to be slow — if the instance timeout (30s) is used,
        # asyncio.wait_for with 0.1s override on the mock level verifies
        # the effective_timeout chain. We instead verify by checking that
        # timeout=None on the call does NOT disable the guard.
        async def slow_recv():
            await asyncio.sleep(60)
        client._transport.recv = slow_recv

        with patch.object(client._transport, 'send', new=AsyncMock()):
            # Per-call timeout=None should fall through to instance timeout=30.0
            # We can't easily wait 30s, so verify by overriding the instance
            # timeout to a small value for the test
            client._timeout = 0.1
            with pytest.raises(asyncio.TimeoutError):
                await client.execute("x = 1", timeout=None)
