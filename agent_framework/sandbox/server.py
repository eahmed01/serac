"""
Persistent sandbox server — shared execution namespace with TCP interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import io
import contextlib
import pathlib
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent_framework.sandbox.transport import (
    Transport,
    TCPTransport,
    Message,
    ConnectionClosed,
)
from agent_framework.sandbox.code_history import CodeHistory

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe builtins
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    # Import support (needed for pandas/numpy internals when exec'd)
    "__import__": __import__,
    # Types
    "len": len, "range": range, "int": int, "float": float, "str": str,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset, "bool": bool,
    "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
    "complex": complex, "object": object, "slice": slice, "super": super,
    "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
    # Constants
    "None": None, "True": True, "False": False,
    # Functions
    "issubclass": issubclass, "isinstance": isinstance, "type": type,
    "id": id, "hash": hash, "abs": abs, "min": min, "max": max,
    "sum": sum, "round": round, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "iter": iter, "next": next, "reversed": reversed,
    "any": any, "all": all, "callable": callable,
    "open": open, "print": print,
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    "format": format, "ascii": ascii, "repr": repr,
    # Exceptions
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "RuntimeError": RuntimeError, "AttributeError": AttributeError,
    "StopIteration": StopIteration, "AssertionError": AssertionError,
    "ImportError": ImportError, "OSError": OSError,
    "EOFError": EOFError, "FileNotFoundError": FileNotFoundError,
    "PermissionError": PermissionError,
    "ZeroDivisionError": ZeroDivisionError, "OverflowError": OverflowError,
    "RecursionError": RecursionError, "NameError": NameError,
    "SyntaxError": SyntaxError,
    "SystemExit": SystemExit, "KeyboardInterrupt": KeyboardInterrupt,
    "GeneratorExit": GeneratorExit,
    "Exception": Exception, "BaseException": BaseException,
}


# ---------------------------------------------------------------------------
# Shared namespace
# ---------------------------------------------------------------------------

class SharedNamespace:
    """A persistent Python execution namespace.

    Holds globals for exec() calls. State persists across all agent sessions
    until the server is restarted.

    The namespace is domain-neutral: domain capabilities (data panel,
    checker functions, ...) are injected via ``sandbox_factory``,
    ``extra_globals``, and ``causality_checker``. With all of them left at
    their defaults the namespace only exposes ``np``/``pd``/``df``/``s``
    (the latter two ``None``) plus the code-history helpers.
    """

    def __init__(
        self,
        tickers: int | List[str] = 100,
        start: str = "2020-01-01",
        end: str = "2025-01-01",
        sandbox_factory: Any = None,
        extra_globals: Optional[Dict[str, Any]] = None,
        causality_checker: Any = None,
        causality_exception: type = Exception,
    ) -> None:
        # Code history — persistent, file-backed
        self._code_history = CodeHistory(
            history_file=str(pathlib.Path.home() / ".sandbox_cache" / "code_history.jsonl"),
        )

        # Store the injected checker so the command handler can use it.
        self.causality_checker = causality_checker
        self.causality_exception = causality_exception

        self.globals: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
            "pd": pd,
            "df": None,  # Will be set when a data panel is loaded
            "s": None,   # Will be set when a sandbox factory is provided
        }

        # Optional extra globals (domain helpers, etc.)
        if extra_globals is not None:
            self.globals.update(extra_globals)

        # Initialize the sandbox instance with the configured data range.
        if sandbox_factory is not None:
            self.globals["s"] = sandbox_factory(tickers=tickers, start=start, end=end)
            self.globals["df"] = self.globals["s"].ohlcv

        # Optional checker registration
        if causality_checker is not None:
            self.globals["check_causality"] = causality_checker
            self.globals["CausalityViolation"] = causality_exception

        # Expose configuration so validation code can use the same panel.
        self.globals["sandbox_config"] = {
            "tickers": tickers, "start": start, "end": end,
        }

        # Wire code history view/export into the namespace
        self.globals["view_code"] = self._view_code
        self.globals["export_code"] = self._export_code
        self.globals["code_history"] = self._code_history

        self._lock = asyncio.Lock()
        self._system_vars = {"s", "df", "sandbox_config",
                             "view_code", "export_code", "code_history"}
        if extra_globals is not None:
            self._system_vars.update(extra_globals.keys())
        if causality_checker is not None:
            self._system_vars.update({"check_causality", "CausalityViolation"})

    async def execute(self, code: str) -> tuple[str, str, bool]:
        """Execute Python code in the shared namespace.

        Returns (stdout, stderr, success). Raises exceptions are caught
        and returned as stderr with success=False.
        """
        async with self._lock:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            success = True

            try:
                with contextlib.redirect_stdout(stdout_capture), \
                     contextlib.redirect_stderr(stderr_capture):
                    exec(code, self.globals)
            except Exception as e:
                success = False
                stderr_capture.write(f"{type(e).__name__}: {e}\n")

            # Record in code history
            self._code_history.record(
                code=code,
                entry_type="execute",
                result_preview=stdout_capture.getvalue(),
                success=success,
            )

            return stdout_capture.getvalue(), stderr_capture.getvalue(), success

    async def evaluate(self, expression: str) -> Any:
        """Evaluate a Python expression and return the result."""
        async with self._lock:
            result = eval(expression, self.globals)

            # Record in code history
            self._code_history.record(
                code=expression,
                entry_type="eval",
                success=True,
            )

            return result

    def _view_code(
        self,
        cell_id: Optional[int] = None,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """View code history from the namespace.

        Usage:
            view_code()           — list all cells with cell_id and timestamp
            view_code(5)          — show source of cell 5
            view_code(start_id=3, end_id=10) — show cells 3 through 10
        """
        entries = self._code_history.list_entries(
            cell_id=cell_id, start_id=start_id, end_id=end_id,
        )
        return entries

    def _export_code(
        self,
        output_path: str,
        cell_id: Optional[int] = None,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
    ) -> str:
        """Export code history to a Python file.

        Usage:
            export_code('/tmp/session.py')                — all cells
            export_code('/tmp/cell5.py', cell_id=5)       — single cell
            export_code('/tmp/range.py', start_id=3, end_id=10) — range
        """
        return self._code_history.export_to_file(
            output_path=output_path,
            cell_id=cell_id,
            start_id=start_id,
            end_id=end_id,
        )

    def list_variables(self) -> List[str]:
        """List user-defined variables (exclude builtins and dunder methods)."""
        skip = {
            "__builtins__", "np", "pd",
            "__name__", "__doc__", "__loader__", "__spec__",
            *self._system_vars,
        }
        return sorted(k for k in self.globals if k not in skip and not k.startswith("__"))

    def get_variable_meta(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a variable (type, shape, size)."""
        obj = self.globals.get(name)
        if obj is None:
            return None

        meta: Dict[str, Any] = {"name": name}

        # Handle pandas objects
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            meta["type"] = type(obj).__module__ + "." + type(obj).__name__
            meta["shape"] = list(obj.shape) if hasattr(obj, "shape") else None
            return meta

        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            meta["type"] = type(obj).__module__ + "." + type(obj).__name__
            meta["shape"] = list(obj.shape)
            meta["dtype"] = str(obj.dtype)
            return meta

        # Generic
        meta["type"] = type(obj).__module__ + "." + type(obj).__name__
        if hasattr(obj, "shape"):
            meta["shape"] = list(obj.shape)
        return meta


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _handle_execute(ns: SharedNamespace, msg: Message) -> Message:
    code = msg.payload.get("code", "")
    stdout, stderr, success = await ns.execute(code)
    return Message("result", stdout=stdout, stderr=stderr, success=success)


async def _handle_eval(ns: SharedNamespace, msg: Message) -> Message:
    expression = msg.payload.get("expression", "")
    try:
        result = await ns.evaluate(expression)
        # Convert result to JSON-serializable form
        if isinstance(result, (np.ndarray, pd.DataFrame, pd.Series)):
            value = {
                "type": type(result).__name__,
                "shape": list(result.shape) if hasattr(result, "shape") else None,
                "dtype": str(result.dtype) if hasattr(result, "dtype") else None,
            }
        elif isinstance(result, (np.integer,)):
            value = int(result)
        elif isinstance(result, (np.floating,)):
            value = float(result)
        elif isinstance(result, (np.bool_,)):
            value = bool(result)
        else:
            value = result
            # Validate JSON-serializable; fall back to repr for non-serializable types
            try:
                json.dumps(value)
            except TypeError:
                value = {"__repr__": repr(result), "type": type(result).__name__}
        return Message("result", success=True, value=value)
    except Exception as e:
        return Message("result", success=False, error=f"{type(e).__name__}: {e}")


async def _handle_list_vars(ns: SharedNamespace, msg: Message) -> Message:
    vars_list = ns.list_variables()
    return Message("result", success=True, variables=vars_list)


async def _handle_get_var(ns: SharedNamespace, msg: Message) -> Message:
    name = msg.payload.get("name", "")
    meta = ns.get_variable_meta(name)
    if meta is None:
        return Message("result", success=False, error=f"variable '{name}' not found")
    # Pass meta fields as payload, avoiding 'type' collision with Message.type
    return Message("result", success=True, var_meta=meta)


async def _handle_shutdown(ns: SharedNamespace, msg: Message) -> Message:
    return Message("result", success=True, message="shutdown requested")


async def _handle_check_causality(ns: SharedNamespace, msg: Message) -> Message:
    """Check a feature function for future data leakage.

    Uses the checker and exception configured on the namespace
    (``causality_checker`` / ``causality_exception``). The command is only
    registered when a checker is configured.

    Payload expects:
      - func_name: str — name of a function in the shared namespace
      - data_name: str (optional, default 'df') — name of DataFrame variable
      - trim_from_end_days: int (optional, default 30)
      - tail_days: int (optional, default 10)
      - atol: float (optional, default 1e-5)
      - rtol: float (optional, default 1e-5)
    """
    if ns.causality_checker is None:
        return Message("result", success=False,
                       error="no causality checker configured on this server")
    check_causality = ns.causality_checker
    CausalityViolation = ns.causality_exception

    func_name = msg.payload.get("func_name", "")
    data_name = msg.payload.get("data_name", "df")
    trim = msg.payload.get("trim_from_end_days", 30)
    tail = msg.payload.get("tail_days", 10)
    atol = msg.payload.get("atol", 1e-5)
    rtol = msg.payload.get("rtol", 1e-5)

    func = ns.globals.get(func_name)
    if func is None or not callable(func):
        return Message("result", success=False,
                       error=f"function '{func_name}' not found in namespace")

    data = ns.globals.get(data_name)
    if data is None:
        return Message("result", success=False,
                       error=f"DataFrame '{data_name}' not found in namespace")

    try:
        check_causality(func, data,
                        trim_from_end_days=trim, tail_days=tail,
                        atol=atol, rtol=rtol)
        return Message("result", success=True,
                       message=f"Causality check passed for '{func_name}'",
                       func_name=func_name, passed=True)
    except CausalityViolation as e:
        return Message("result", success=False,
                       error=f"CausalityViolation: {e}", func_name=func_name,
                       passed=False)
    except Exception as e:
        return Message("result", success=False,
                       error=f"check error: {type(e).__name__}: {e}")


# Command handler registry.
# "check_causality" is only registered when the namespace has a checker
# configured (see _command_handlers).
_BASE_COMMAND_HANDLERS = {
    "execute": _handle_execute,
    "eval": _handle_eval,
    "list_vars": _handle_list_vars,
    "get_var": _handle_get_var,
    "shutdown": _handle_shutdown,
}


def _command_handlers(ns: SharedNamespace) -> Dict[str, Any]:
    """Build the command handler registry for a namespace."""
    handlers = dict(_BASE_COMMAND_HANDLERS)
    if ns.causality_checker is not None:
        handlers["check_causality"] = _handle_check_causality
    return handlers


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class SandboxServer:
    """Persistent sandbox server with TCP or Unix socket interface.

    Maintains a shared Python namespace and accepts commands from clients.
    Can listen on either TCP or Unix socket for Docker container communication.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        socket_path: Optional[str] = None,
        tickers: int | List[str] = 100,
        start: str = "2020-01-01",
        end: str = "2025-01-01",
        sandbox_factory: Any = None,
        extra_globals: Optional[Dict[str, Any]] = None,
        causality_checker: Any = None,
        causality_exception: type = Exception,
    ) -> None:
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self.tickers = tickers
        self.start_date = start
        self.end_date = end
        self._namespace = SharedNamespace(
            tickers=tickers, start=start, end=end,
            sandbox_factory=sandbox_factory,
            extra_globals=extra_globals,
            causality_checker=causality_checker,
            causality_exception=causality_exception,
        )
        self._handlers = _command_handlers(self._namespace)
        self._server: Optional[asyncio.Server] = None
        self._accepting = False
        self._running = False

    async def start(self) -> None:
        """Start the server on TCP or Unix socket."""
        self._accepting = True
        self._running = True
        if self.socket_path:
            # Unix socket mode (for Docker containers)
            import os
            # Remove stale socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            self._server = await asyncio.start_unix_server(
                self._handle_client, self.socket_path,
            )
            log.info(f"Sandbox server started on Unix socket: {self.socket_path}")
        else:
            # TCP mode
            self._server = await asyncio.start_server(
                self._handle_client, self.host, self.port,
                reuse_address=True,
            )
            log.info(f"Sandbox server started on {self.host}:{self.port}")

    async def stop(self, timeout: float = 3.0) -> None:
        """Stop the server and close all connections."""
        self._accepting = False
        self._running = False
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("Server close timed out")
            # Clean up Unix socket file
            if self.socket_path:
                import os
                if os.path.exists(self.socket_path):
                    os.unlink(self.socket_path)
            log.info("Sandbox server stopped")

    async def _handle_client(self, reader, writer) -> None:
        """Handle a single client connection."""
        addr = writer.get_extra_info("peername")
        transport = TCPTransport()
        transport.attach(reader, writer)

        try:
            while self._running:
                try:
                    msg = await transport.recv()
                except ConnectionClosed:
                    log.info(f"Client {addr} disconnected")
                    break

                # Dispatch to handler
                handler = self._handlers.get(msg.type)
                if handler is None:
                    resp = Message("result", success=False,
                                   error=f"unknown command: {msg.type}")
                else:
                    try:
                        resp = await handler(self._namespace, msg)
                    except Exception as e:
                        resp = Message("result", success=False,
                                       error=f"handler error: {e}")

                # Check for shutdown
                if msg.type == "shutdown":
                    await transport.send(resp)
                    self._accepting = False
                    if self._server:
                        self._server.close()
                    break

                await transport.send(resp)
        except Exception as e:
            log.exception(f"Error handling client {addr}: {e}")
        finally:
            await transport.close()

    async def run_forever(self) -> None:
        """Run the server until interrupted. Calls start() if not yet running."""
        if not self._running:
            await self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self.stop()


def _parse_tickers(value: str) -> int | List[str]:
    """Parse a ticker count or comma-separated explicit ticker list."""
    value = value.strip()
    try:
        count = int(value)
    except ValueError:
        tickers = [ticker.strip() for ticker in value.split(",") if ticker.strip()]
        if not tickers:
            raise argparse.ArgumentTypeError("tickers must be a positive count or ticker list")
        return tickers
    if count < 1:
        raise argparse.ArgumentTypeError("ticker count must be positive")
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the parser for ``python -m agent_framework.sandbox.server``."""
    parser = argparse.ArgumentParser(description="Persistent sandbox server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--socket-path", default=None)
    parser.add_argument("--tickers", type=_parse_tickers, default=100,
                        help="Ticker count or comma-separated ticker symbols")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    return parser


async def _run_from_args(args: argparse.Namespace) -> None:
    server = SandboxServer(
        host=args.host, port=args.port, socket_path=args.socket_path,
        tickers=args.tickers, start=args.start, end=args.end,
    )
    try:
        await server.run_forever()
    finally:
        if server._running or server._server is not None:
            await server.stop()


def main(argv: Optional[List[str]] = None) -> None:
    """Run the persistent server from command-line arguments."""
    args = build_arg_parser().parse_args(argv)
    try:
        asyncio.run(_run_from_args(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
