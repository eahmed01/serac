"""
Async sandbox client — programmatic Python execution over TCP.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .transport import TCPTransport, Message, ConnectionClosed


@dataclass
class ExecuteResponse:
    """Response from execute/eval commands."""
    success: bool
    output: str = ""
    error: str = ""
    variables: List[str] = field(default_factory=list)
    shape: List[int] = field(default_factory=list)
    var_name: str = ""
    var_type: str = ""


class SandboxClient:
    """Async client for sandbox server communication.

    Parameters
    ----------
    host : str
        Server hostname (default: 127.0.0.1).
    port : int
        Server port (default: 9876).
    timeout : float, optional
        Default timeout in seconds for request/response round-trips.
        Applied to execute, eval, list_vars, and get_var unless overridden.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9876, timeout: Optional[float] = 300.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._transport: Optional[TCPTransport] = None

    @property
    def is_connected(self) -> bool:
        return self._transport is not None

    async def connect(self) -> None:
        """Connect to the sandbox server."""
        self._transport = TCPTransport()
        await self._transport.connect(self._host, self._port)

    async def close(self) -> None:
        """Close the connection."""
        if self._transport:
            await self._transport.close()
            self._transport = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _send_and_recv(self, msg: Message) -> Message:
        """Send a message and receive the response.

        Catches transport-level errors (ConnectionResetError, BrokenPipeError,
        OSError) and normalizes them to ConnectionClosed so callers only need
        to handle one exception type for connection failures.
        """
        if not self._transport:
            raise RuntimeError("Not connected")
        try:
            await self._transport.send(msg)
            return await self._transport.recv()
        except ConnectionClosed:
            raise
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise ConnectionClosed(f"Communication error: {exc}") from exc

    def _make_response(self, msg: Message) -> ExecuteResponse:
        """Convert a server response message to ExecuteResponse."""
        payload = msg.payload
        success = payload.get("success", False)
        output = payload.get("output", payload.get("stdout", ""))
        error = payload.get("error", payload.get("stderr", ""))
        variables = payload.get("variables", [])

        # Handle eval result (value field) — convert to string for output
        if "value" in payload and payload["value"] is not None and not output:
            output = str(payload["value"])

        # Handle variable metadata (nested in var_meta)
        shape = []
        var_name = ""
        var_type = ""
        if "var_meta" in payload and payload["var_meta"]:
            meta = payload["var_meta"]
            shape = meta.get("shape", [])
            var_name = meta.get("name", "")
            var_type = meta.get("type", "")

        return ExecuteResponse(
            success=success,
            output=output,
            error=error,
            variables=variables,
            shape=shape,
            var_name=var_name,
            var_type=var_type,
        )

    async def _send_recv_with_timeout(self, msg: Message, timeout: Optional[float]) -> Message:
        """Send/receive with optional timeout enforcement.

        If timeout is set (including the instance default), wraps the
        _send_and_recv call in asyncio.wait_for() so that a hung server
        raises asyncio.TimeoutError instead of blocking forever.

        Note: asyncio.wait_for() requires the underlying coroutine to be
        cancellable. With Python 3.12+, StreamReader.readexactly() may not
        be fully cancellable in all cases, so timeout enforcement is best-effort
        with live TCP connections.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("timeout must be > 0 or None")
        recv_task = self._send_and_recv(msg)
        if effective_timeout is not None:
            return await asyncio.wait_for(recv_task, timeout=effective_timeout)
        return await recv_task

    async def execute(self, code: str, timeout: Optional[float] = None) -> ExecuteResponse:
        """Execute Python code on the server.

        Parameters
        ----------
        code : str
            Python code to execute.
        timeout : float, optional
            Timeout in seconds for this request. Overrides the client default.
            Pass None to use the instance timeout, or a specific value to override.

        Returns
        -------
        ExecuteResponse
            Response with success status, output, error, and variable metadata.

        Raises
        ------
        asyncio.TimeoutError
            If the request exceeds the timeout (instance or per-call).
        ConnectionClosed
            If the connection is lost during send or receive.
        ValueError
            If timeout is <= 0.
        """
        msg = Message("execute", code=code)
        resp = await self._send_recv_with_timeout(msg, timeout)
        return self._make_response(resp)

    async def eval(self, expression: str, timeout: Optional[float] = None) -> ExecuteResponse:
        """Evaluate a Python expression on the server.

        Parameters
        ----------
        expression : str
            Python expression to evaluate.
        timeout : float, optional
            Timeout in seconds for this request. Overrides the client default.

        Returns
        -------
        ExecuteResponse
            Response with success status, output, error, and variable metadata.

        Raises
        ------
        asyncio.TimeoutError
            If the request exceeds the timeout (instance or per-call).
        ConnectionClosed
            If the connection is lost during send or receive.
        ValueError
            If timeout is <= 0.
        """
        msg = Message("eval", expression=expression)
        resp = await self._send_recv_with_timeout(msg, timeout)
        return self._make_response(resp)

    async def list_vars(self, timeout: Optional[float] = None) -> ExecuteResponse:
        """List all user-defined variables on the server.

        Parameters
        ----------
        timeout : float, optional
            Timeout in seconds for this request. Overrides the client default.

        Returns
        -------
        ExecuteResponse
            Response with success status and list of variable names.

        Raises
        ------
        asyncio.TimeoutError
            If the request exceeds the timeout (instance or per-call).
        ConnectionClosed
            If the connection is lost during send or receive.
        ValueError
            If timeout is <= 0.
        """
        msg = Message("list_vars")
        resp = await self._send_recv_with_timeout(msg, timeout)
        return self._make_response(resp)

    async def get_var(self, name: str, timeout: Optional[float] = None) -> ExecuteResponse:
        """Get metadata for a specific variable.

        Parameters
        ----------
        name : str
            Variable name to look up.
        timeout : float, optional
            Timeout in seconds for this request. Overrides the client default.

        Returns
        -------
        ExecuteResponse
            Response with success status and variable metadata (shape, type).

        Raises
        ------
        asyncio.TimeoutError
            If the request exceeds the timeout (instance or per-call).
        ConnectionClosed
            If the connection is lost during send or receive.
        ValueError
            If timeout is <= 0.
        """
        msg = Message("get_var", name=name)
        resp = await self._send_recv_with_timeout(msg, timeout)
        return self._make_response(resp)

    async def check_causality(
        self,
        func_name: str,
        data_name: str = "df",
        trim_from_end_days: int = 30,
        tail_days: int = 10,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        timeout: Optional[float] = None,
    ) -> ExecuteResponse:
        """Check a feature function for future data leakage.

        The function must already exist in the server namespace (via
        ``execute`` or pre-loaded). The check compares outputs with and
        without future data — if they differ, the function leaks.

        Parameters
        ----------
        func_name : str
            Name of the function in the server namespace.
        data_name : str
            Name of the DataFrame variable in the server namespace
            (default: ``"df"``).
        trim_from_end_days : int
            Number of trailing days to remove for the truncated run.
        tail_days : int
            Number of days at the tail of truncated data to compare.
        atol : float
            Absolute tolerance for ``numpy.isclose``.
        rtol : float
            Relative tolerance for ``numpy.isclose``.
        timeout : float, optional
            Timeout in seconds for this request. Overrides the client default.

        Returns
        -------
        ExecuteResponse
            ``success=True`` if causality check passed (no leakage),
            ``success=False`` with ``error`` describing the violation.

        Raises
        ------
        asyncio.TimeoutError
            If the request exceeds the timeout (instance or per-call).
        ConnectionClosed
            If the connection is lost during send or receive.
        ValueError
            If timeout is <= 0.
        """
        msg = Message(
            "check_causality",
            func_name=func_name,
            data_name=data_name,
            trim_from_end_days=trim_from_end_days,
            tail_days=tail_days,
            atol=atol,
            rtol=rtol,
        )
        resp = await self._send_recv_with_timeout(msg, timeout)
        return self._make_response(resp)
