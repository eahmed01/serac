"""Proxy server for LLM API calls from sandboxed containers.

The sandbox runs with --network none, which blocks all outbound network.
However, Unix sockets work fine. This proxy:

1. Listens on a Unix socket accessible from inside the container
2. Forwards requests to external LLM APIs (Anthropic, OpenAI, etc.)
3. Logs all requests for audit trails

This allows the consultant to make API calls while keeping the sandbox
network-isolated from the internet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMProxy:
    """Proxy server for LLM API calls."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.server = None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single client connection."""
        try:
            # Read request
            data = await reader.readline()
            if not data:
                return

            request = json.loads(data)
            method = request.get("method", "chat")
            model = request.get("model", "")
            messages = request.get("messages", [])

            # Forward to appropriate API
            if "anthropic" in model:
                response = await self._forward_to_anthropic(messages, model)
            elif "openai" in model or "gpt" in model:
                response = await self._forward_to_openai(messages, model)
            else:
                response = {"error": f"Unknown model: {model}"}

            # Send response
            response_data = json.dumps(response)
            writer.write(response_data.encode() + b"\n")
            await writer.drain()

        except Exception as e:
            logger.exception("Error handling client")
            error_response = {"error": str(e)}
            writer.write(json.dumps(error_response).encode() + b"\n")
            await writer.drain()

        finally:
            writer.close()
            await writer.wait_closed()

    async def _forward_to_anthropic(self, messages: list, model: str) -> dict:
        """Not yet implemented — returns a stub response.

        Real forwarding to the Anthropic API (with API key handling) is not
        implemented; callers currently receive a hardcoded stub payload.
        """
        return {
            "content": "This is a mock response from Anthropic API",
            "model": model,
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }

    async def _forward_to_openai(self, messages: list, model: str) -> dict:
        """Not yet implemented — returns a stub response.

        Real forwarding to the OpenAI API (with API key handling) is not
        implemented; callers currently receive a hardcoded stub payload.
        """
        return {
            "content": "This is a mock response from OpenAI API",
            "model": model,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }

    async def _bind(self) -> None:
        """Bind the Unix socket for the proxy server.

        The socket file is created with mode 0o600 so that only the owner
        (this user) can connect to the proxy; no other local process may.
        """
        # Remove existing socket if present
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Restrict the socket file to the owner before binding.
        old_umask = os.umask(0o177)
        try:
            self.server = await asyncio.start_unix_server(
                self.handle_client,
                path=self.socket_path
            )
        finally:
            os.umask(old_umask)

        # Belt and braces: ensure 0o600 regardless of umask handling.
        os.chmod(self.socket_path, 0o600)

    async def start(self) -> None:
        """Start the proxy server."""
        await self._bind()

        logger.info("LLM Proxy listening on %s", self.socket_path)

        # Handle shutdown
        loop = asyncio.get_event_loop()
        stop = loop.create_future()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set_result, None)
        await stop

        # Cleanup
        self.server.close()
        await self.server.wait_closed()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def run(self) -> None:
        """Run the proxy server."""
        asyncio.run(self.start())


def main() -> None:
    """Run the LLM proxy server."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM API proxy server")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    proxy = LLMProxy(socket_path=args.socket)
    proxy.run()


if __name__ == "__main__":
    main()
