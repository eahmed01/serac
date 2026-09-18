"""Focused tests for configurable sandbox server construction and CLI parsing."""

import pandas as pd

from agent_framework.sandbox.server import SharedNamespace, SandboxServer, build_arg_parser


class FakeSandbox:
    calls = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        self.ohlcv = pd.DataFrame({"ticker": [], "time": []})


def test_shared_namespace_propagates_configuration_without_loading_data():
    FakeSandbox.calls.clear()
    ns = SharedNamespace(
        tickers=["AAA", "BBB"],
        start="2010-01-01",
        end="2011-01-01",
        sandbox_factory=FakeSandbox,
    )
    assert FakeSandbox.calls == [{
        "tickers": ["AAA", "BBB"],
        "start": "2010-01-01",
        "end": "2011-01-01",
    }]
    assert ns.globals["sandbox_config"] == FakeSandbox.calls[0]


def test_server_defaults_and_propagation():
    FakeSandbox.calls.clear()
    server = SandboxServer(sandbox_factory=FakeSandbox)
    assert FakeSandbox.calls == [{
        "tickers": 100,
        "start": "2020-01-01",
        "end": "2025-01-01",
    }]
    assert (server.host, server.port, server.socket_path) == ("127.0.0.1", 9876, None)


def test_cli_parser_supports_all_server_and_data_options():
    args = build_arg_parser().parse_args([
        "--host", "0.0.0.0", "--port", "9999",
        "--socket-path", "/tmp/sandbox.sock", "--tickers", "AAA, BBB",
        "--start", "2010-01-01", "--end", "2026-01-01",
    ])
    assert vars(args) == {
        "host": "0.0.0.0", "port": 9999, "socket_path": "/tmp/sandbox.sock",
        "tickers": ["AAA", "BBB"], "start": "2010-01-01", "end": "2026-01-01",
    }


def test_cli_parser_accepts_ticker_count_and_preserves_defaults():
    parser = build_arg_parser()
    assert parser.parse_args([]).__dict__ == {
        "host": "127.0.0.1", "port": 9876, "socket_path": None,
        "tickers": 100, "start": "2020-01-01", "end": "2025-01-01",
    }
    assert parser.parse_args(["--tickers", "25"]).tickers == 25
