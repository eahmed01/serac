"""
Transport layer tests — TCP framing, JSON serialization, protocol messages.
"""

import asyncio
import json
import struct
from unittest import mock
import pytest
from agent_framework.sandbox.transport import (
    Message,
    TCPTransport,
    _encode_frame,
    _decode_frame,
    ConnectionClosed,
)


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------

class TestMessage:
    def test_roundtrip(self):
        msg = Message("execute", code="print('hello')")
        json_str = msg.to_json()
        decoded = Message.from_json(json_str)
        assert decoded.type == "execute"
        assert decoded.payload["code"] == "print('hello')"

    def test_empty_payload(self):
        msg = Message("shutdown")
        json_str = msg.to_json()
        decoded = Message.from_json(json_str)
        assert decoded.type == "shutdown"
        assert decoded.payload == {}

    def test_nested_payload(self):
        msg = Message("experiment", action="log", params={"lr": 1e-5, "d_model": 256})
        json_str = msg.to_json()
        decoded = Message.from_json(json_str)
        assert decoded.payload["params"]["lr"] == 1e-5


# ---------------------------------------------------------------------------
# Wire framing
# ---------------------------------------------------------------------------

class TestFraming:
    def test_encode_decode_roundtrip(self):
        original = '{"type": "execute", "code": "x = 1"}'
        frame = _encode_frame(original)
        # Parse: 4-byte big-endian length + newline + json + newline
        length = struct.unpack(">I", frame[:4])[0]
        json_bytes = frame[5:-1]  # skip length prefix + newline, skip trailing newline
        assert len(json_bytes) == length
        assert json_bytes.decode("utf-8") == original

    def test_large_message(self):
        """A 10KB message should frame correctly."""
        big = '{"type": "execute", "code": "' + "x = 1\n" * 200 + '"}'
        frame = _encode_frame(big)
        length = struct.unpack(">I", frame[:4])[0]
        json_bytes = frame[5:-1]
        assert len(json_bytes) == length


# ---------------------------------------------------------------------------
# Async decode_frame with in-memory pipes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAsyncTransport:
    async def test_tcp_roundtrip(self):
        """Server sends a message, client receives it."""
        server_reader = client_reader = None
        server_writer = client_writer = None

        async def server_task():
            nonlocal server_reader, server_writer
            reader, writer = await asyncio.open_connection("127.0.0.1", 9877)
            server_reader, server_writer = reader, writer

        # Start a simple echo server
        async def echo_handler(reader, writer):
            try:
                raw = await _decode_frame(reader)
                msg = Message.from_json(raw)
                # Echo back with result type
                response = Message("result", stdout=f"echo:{msg.type}", success=True)
                writer.write(_encode_frame(response.to_json()))
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        # Start server
        server = await asyncio.start_server(echo_handler, "127.0.0.1", 9877)

        try:
            # Connect client
            client = TCPTransport()
            await client.connect("127.0.0.1", 9877)

            # Send and receive
            await client.send(Message("test", value=42))
            resp = await client.recv()
            assert resp.type == "result"
            assert resp.payload["stdout"] == "echo:test"

            await client.close()
        finally:
            server.close()
            await server.wait_closed()

    async def test_connection_closed_raises(self):
        """recv() on a closed connection raises ConnectionClosed."""
        client = TCPTransport()
        # Not connected — should raise
        with pytest.raises(ConnectionClosed):
            await client.recv()

    async def test_send_before_connect_raises(self):
        with pytest.raises(ConnectionClosed):
            client = TCPTransport()
            await client.send(Message("test"))

    async def test_send_after_peer_disconnect_raises_connection_closed(self):
        """When the peer disconnects, send() raises ConnectionClosed, not BrokenPipeError."""
        client = TCPTransport()
        # Mock the writer to simulate a broken pipe during drain
        mock_writer = mock.AsyncMock()
        mock_writer.write = mock.Mock()
        mock_writer.drain = mock.AsyncMock(side_effect=BrokenPipeError("Broken pipe"))
        client.attach(asyncio.StreamReader(), mock_writer)
        with pytest.raises(ConnectionClosed, match="Connection lost while sending"):
            await client.send(Message("test", value=42))

    async def test_recv_non_dict_json_raises_connection_closed(self):
        """When the wire carries non-object JSON (list, string, etc.), recv() raises ConnectionClosed."""
        async def bad_json_handler(reader, writer):
            # Send a validly-framed but non-dict JSON payload
            raw = json.dumps([1, 2, 3])
            writer.write(_encode_frame(raw))
            await writer.drain()
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(bad_json_handler, "127.0.0.1", 9880)
        try:
            client = TCPTransport()
            await client.connect("127.0.0.1", 9880)
            with pytest.raises(ConnectionClosed):
                await client.recv()
            await client.close()
        finally:
            server.close()
            await server.wait_closed()

    async def test_recv_oversized_frame_raises_connection_closed(self):
        """When a frame claims a size exceeding max_message_size, recv() raises ConnectionClosed."""
        small_limit = 1024  # 1 KB limit for testing

        async def oversized_handler(reader, writer):
            # Write a frame header claiming 100 KB payload but send nothing
            header = struct.pack(">I", 100 * 1024) + b"\n"
            writer.write(header)
            await writer.drain()
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(oversized_handler, "127.0.0.1", 9881)
        try:
            client = TCPTransport(max_message_size=small_limit)
            await client.connect("127.0.0.1", 9881)
            with pytest.raises(ConnectionClosed, match="Frame too large"):
                await client.recv()
            await client.close()
        finally:
            server.close()
            await server.wait_closed()


# ---------------------------------------------------------------------------
# Non-dict JSON handling
# ---------------------------------------------------------------------------

class TestNonDictJSON:
    """Message.from_json() must reject non-dict JSON payloads gracefully."""

    def test_rejects_list(self):
        with pytest.raises(ValueError, match="got list"):
            Message.from_json(json.dumps([1, 2, 3]))

    def test_rejects_string(self):
        with pytest.raises(ValueError, match="got str"):
            Message.from_json(json.dumps("just a string"))

    def test_rejects_number(self):
        with pytest.raises(ValueError, match="got int"):
            Message.from_json(json.dumps(42))

    def test_rejects_null(self):
        with pytest.raises(ValueError, match="got NoneType"):
            Message.from_json(json.dumps(None))

    def test_rejects_boolean(self):
        with pytest.raises(ValueError, match="got bool"):
            Message.from_json(json.dumps(True))

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="got float"):
            Message.from_json(json.dumps(3.14))

    def test_accepts_valid_object(self):
        msg = Message.from_json('{"type": "ping"}')
        assert msg.type == "ping"

    def test_accepts_object_with_extra_keys(self):
        msg = Message.from_json('{"type": "data", "value": 1, "extra": "ok"}')
        assert msg.type == "data"
        assert msg.payload == {"value": 1, "extra": "ok"}


# ---------------------------------------------------------------------------
# Max message size enforcement
# ---------------------------------------------------------------------------

class TestMaxMessageSize:
    """_decode_frame() must reject frames exceeding max_message_size."""

    def test_default_max_is_100mb(self):
        """Default max_message_size on TCPTransport is 100 MB."""
        t = TCPTransport()
        assert t._max_message_size == 100 * 1024 * 1024

    def test_custom_max_message_size(self):
        """TCPTransport accepts custom max_message_size."""
        t = TCPTransport(max_message_size=512)
        assert t._max_message_size == 512

    @pytest.mark.asyncio
    async def test_decode_frame_rejects_oversized(self):
        """_decode_frame raises ConnectionClosed when frame exceeds limit."""
        reader = asyncio.StreamReader()
        # Feed a frame claiming 10 KB payload directly to the reader
        header = struct.pack(">I", 10240) + b"\n"
        reader.feed_data(header)
        with pytest.raises(ConnectionClosed, match="Frame too large"):
            await _decode_frame(reader, max_message_size=1024)

    @pytest.mark.asyncio
    async def test_decode_frame_allows_within_limit(self):
        """_decode_frame accepts frames within max_message_size."""
        reader = asyncio.StreamReader()
        payload = '{"type": "ok"}'
        frame = _encode_frame(payload)
        reader.feed_data(frame)
        result = await _decode_frame(reader, max_message_size=1024)
        assert result == payload
