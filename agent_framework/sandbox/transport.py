"""
Modular transport layer for sandbox communication.

Abstracts the wire protocol so the server and client don't need to know
whether they're talking TCP, HTTP, or something else.

Message format: 4-byte big-endian length prefix + newline + JSON payload + newline.
"""

from __future__ import annotations

import asyncio
import json
import struct
from abc import ABC, abstractmethod
from typing import Any


class Message:
    """A sandbox protocol message."""

    __slots__ = ("type", "payload")

    def __init__(self, msg_type: str, **kwargs):
        self.type = msg_type
        self.payload = kwargs

    def to_json(self) -> str:
        return json.dumps({"type": self.type, **self.payload})

    @classmethod
    def from_json(cls, raw: str) -> Message:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Message expected JSON object, got {type(data).__name__}"
            )
        msg_type = data.pop("type", "unknown")
        return cls(msg_type, **data)


# ---------------------------------------------------------------------------
# Abstract transport
# ---------------------------------------------------------------------------

class Transport(ABC):
    """Abstract transport layer for sandbox communication."""

    @abstractmethod
    async def send(self, msg: Message) -> None:
        """Send a message to the remote end."""

    @abstractmethod
    async def recv(self) -> Message:
        """Receive and return a message from the remote end."""

    @abstractmethod
    async def close(self) -> None:
        """Shut down the transport."""


# ---------------------------------------------------------------------------
# Wire helpers (shared by all transports)
# ---------------------------------------------------------------------------

def _encode_frame(json_str: str) -> bytes:
    """Length-prefix a JSON string: <4-byte BE length>\\n<json>\\n"""
    encoded = json_str.encode("utf-8")
    header = struct.pack(">I", len(encoded)) + b"\n"
    return header + encoded + b"\n"


async def _decode_frame(reader: asyncio.StreamReader, *, max_message_size: int = 100 * 1024 * 1024) -> str:
    """Read a length-prefixed JSON string from a stream.

    Format: <4-byte big-endian length><newline><json payload><newline>
    We readexactly(4) for length, then readline() to consume the newline,
    then readexactly(length) for the payload, then readexactly(1) for trailing newline.

    Args:
        reader: AsyncIO stream reader.
        max_message_size: Maximum allowed payload size in bytes (default 100 MB).
    """
    # Read exactly 4 bytes for the length (binary-safe)
    length_bytes = await reader.readexactly(4)
    length = struct.unpack(">I", length_bytes)[0]

    # Guard against oversized frames (DoS protection)
    if length > max_message_size:
        raise ConnectionClosed(
            f"Frame too large: {length} bytes (max {max_message_size})"
        )

    # Consume the newline after the length prefix
    nl = await reader.readexactly(1)
    if nl != b"\n":
        raise ConnectionClosed(f"Malformed frame: expected newline, got {nl!r}")

    # Read the payload
    data = await reader.readexactly(length)
    payload = data.decode("utf-8")

    # Consume trailing newline
    trail = await reader.readexactly(1)
    if trail != b"\n":
        raise ConnectionClosed(f"Malformed frame: expected trailing newline, got {trail!r}")

    return payload


# ---------------------------------------------------------------------------
# TCP transport
# ---------------------------------------------------------------------------

class ConnectionClosed(Exception):
    """Raised when the transport connection is lost."""
    pass


class TCPTransport(Transport):
    """Raw TCP socket transport.

    Client side: connects to host:port.
    Server side: wraps an accepted (reader, writer) pair.
    """

    def __init__(self, *, max_message_size: int = 100 * 1024 * 1024) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: Any = None  # asyncio.Writer (3.11+) / asyncio.StreamWriter
        self._max_message_size: int = max_message_size

    async def connect(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        """Connect to a TCP server (client side)."""
        self._reader, self._writer = await asyncio.open_connection(host, port)

    def attach(self, reader: asyncio.StreamReader, writer: Any) -> None:
        """Attach to an accepted connection (server side)."""
        self._reader = reader
        self._writer = writer

    async def send(self, msg: Message) -> None:
        if self._writer is None:
            raise ConnectionClosed("Not connected")
        try:
            frame = _encode_frame(msg.to_json())
            self._writer.write(frame)
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError) as exc:
            raise ConnectionClosed(f"Connection lost while sending: {exc}") from exc

    async def recv(self) -> Message:
        if self._reader is None:
            raise ConnectionClosed("Not connected")
        try:
            raw = await _decode_frame(self._reader, max_message_size=self._max_message_size)
            return Message.from_json(raw)
        except (ConnectionResetError, asyncio.IncompleteReadError) as exc:
            raise ConnectionClosed(f"Connection lost: {exc}") from exc
        except ValueError as exc:
            raise ConnectionClosed(f"Invalid message: {exc}") from exc

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None


# ---------------------------------------------------------------------------
# Unix socket transport (for Docker container communication)
# ---------------------------------------------------------------------------


class UnixSocketTransport(Transport):
    """Unix domain socket transport.

    Used for communication between Docker containers that share a volume mount.
    The socket file is created in the shared volume, and both containers can
    access it without network access.

    Client side: connects to socket path.
    Server side: wraps an accepted (reader, writer) pair.
    """

    def __init__(self, *, max_message_size: int = 100 * 1024 * 1024) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: Any = None
        self._max_message_size: int = max_message_size

    async def connect(self, path: str = "/tmp/agent_workspace/sandbox.sock") -> None:
        """Connect to a Unix socket server (client side)."""
        self._reader, self._writer = await asyncio.open_unix_connection(path)

    def attach(self, reader: asyncio.StreamReader, writer: Any) -> None:
        """Attach to an accepted connection (server side)."""
        self._reader = reader
        self._writer = writer

    async def send(self, msg: Message) -> None:
        if self._writer is None:
            raise ConnectionClosed("Not connected")
        try:
            frame = _encode_frame(msg.to_json())
            self._writer.write(frame)
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError) as exc:
            raise ConnectionClosed(f"Connection lost while sending: {exc}") from exc

    async def recv(self) -> Message:
        if self._reader is None:
            raise ConnectionClosed("Not connected")
        try:
            raw = await _decode_frame(self._reader, max_message_size=self._max_message_size)
            return Message.from_json(raw)
        except (ConnectionResetError, asyncio.IncompleteReadError) as exc:
            raise ConnectionClosed(f"Connection lost: {exc}") from exc
        except ValueError as exc:
            raise ConnectionClosed(f"Invalid message: {exc}") from exc

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
