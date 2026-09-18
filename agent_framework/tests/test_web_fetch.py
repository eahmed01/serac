"""Permanent tests for the bounded web_fetch tool (agent_framework.builtins).

Network is fully mocked (socket.getaddrinfo + urllib.request.urlopen). The
tests verify scheme/host validation, private-address refusal, fail-closed
HTTP/binary errors, HTML text extraction, the max_chars cap, and the ToolDef
schema (event-tier, never identity).
"""
from __future__ import annotations

import json
import socket

import pytest

from agent_framework.builtins import _web_fetch_executor, web_fetch_factory

PUBLIC = "93.184.216.34"
PRIVATE = "10.0.0.1"
LOOPBACK = "127.0.0.1"


class _FakeResponse:
    def __init__(self, payload: bytes, status=200, content_type="text/html"):
        self._payload = payload
        self.status = status

        class _H:
            def get(self, key, default=None):
                return {"Content-Type": content_type}.get(key, default)

        self.headers = _H()

    def getcode(self):
        return self.status

    def read(self, n=-1):
        return self._payload[:n] if n and n > 0 else self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ai(ip: str):
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))


def _patch(monkeypatch, ip=PUBLIC, payload=b"", status=200, content_type="text/html",
           error=None, urlopen_error=None):
    monkeypatch.setattr("agent_framework.builtins.socket.getaddrinfo",
                        lambda *a, **k: [_ai(ip)])

    def fake_urlopen(req, timeout=None):
        if urlopen_error is not None:
            raise urlopen_error
        if error is not None:
            raise error
        return _FakeResponse(payload, status=status, content_type=content_type)

    monkeypatch.setattr("agent_framework.builtins.urllib.request.urlopen", fake_urlopen)


# --- ToolDef schema -------------------------------------------------------

def test_tool_def_schema_is_event_tier():
    tool = web_fetch_factory()
    assert tool.name == "web_fetch"
    assert "event" in tool.description.lower()
    assert "never" in tool.description.lower()  # never identity/price/action/adjustment
    props = tool.parameters["properties"]
    assert set(props) == {"url", "max_chars"}
    assert tool.parameters["required"] == ["url"]
    assert props["max_chars"]["default"] == 20000


# --- URL validation (fail closed) ----------------------------------------

def test_requires_url():
    assert "url is required" in _web_fetch_executor()


def test_rejects_non_http_scheme(monkeypatch):
    _patch(monkeypatch)
    out = json.loads(_web_fetch_executor("file:///etc/passwd"))
    assert "error" in out and "http" in out["error"]
    out = json.loads(_web_fetch_executor("ftp://example.com/x"))
    assert "error" in out and "http" in out["error"]


def test_rejects_localhost(monkeypatch):
    _patch(monkeypatch, ip=LOOPBACK)
    out = json.loads(_web_fetch_executor("http://localhost/article"))
    assert "error" in out and ("private" in out["error"] or "loopback" in out["error"].lower())


def test_rejects_private_address(monkeypatch):
    _patch(monkeypatch, ip=PRIVATE)
    out = json.loads(_web_fetch_executor("https://10.0.0.1/x"))
    assert "error" in out and "private" in out["error"]


def test_unresolvable_host_fails_closed(monkeypatch):
    def bad_getaddrinfo(*a, **k):
        raise socket.gaierror("no such host")
    monkeypatch.setattr("agent_framework.builtins.socket.getaddrinfo", bad_getaddrinfo)
    out = json.loads(_web_fetch_executor("https://nonexistent.invalid.example/x"))
    assert "error" in out and "resolve" in out["error"]


# --- fetch + extraction ---------------------------------------------------

HTML = b"""<html><head><style>body{color:red}</style>
<script>var x='secret';</script></head>
<body><h1>  Velo3D Announces Reverse Split  </h1>
<p>The company executed a <b>1-for-38</b> reverse split.</p></body></html>"""


def test_extracts_text_and_strips_script_style(monkeypatch):
    _patch(monkeypatch, payload=HTML)
    out = json.loads(_web_fetch_executor("https://example.com/article"))
    assert out["status"] == 200
    assert "Velo3D Announces Reverse Split" in out["text"]
    assert "1-for-38" in out["text"]
    assert "secret" not in out["text"]  # script stripped
    assert "color:red" not in out["text"]  # style stripped
    assert "<" not in out["text"]
    assert out["truncated"] is False


def test_caps_at_max_chars(monkeypatch):
    big = b"<p>" + b"x" * 50000 + b"</p>"
    _patch(monkeypatch, payload=big)
    out = json.loads(_web_fetch_executor("https://example.com/big", max_chars=5000))
    assert out["truncated"] is True
    assert out["chars"] == 5000
    assert len(out["text"]) == 5000


def test_default_cap_is_20k(monkeypatch):
    big = b"<p>" + b"x" * 40000 + b"</p>"
    _patch(monkeypatch, payload=big)
    out = json.loads(_web_fetch_executor("https://example.com/big"))
    assert out["truncated"] is True
    assert out["chars"] == 20000


def test_rejects_binary_content_type(monkeypatch):
    _patch(monkeypatch, payload=b"\x00\x01binary", content_type="application/octet-stream")
    out = json.loads(_web_fetch_executor("https://example.com/file.bin"))
    assert "error" in out and "text" in out["error"]
    assert out.get("content_type") == "application/octet-stream"


def test_http_error_fails_closed(monkeypatch):
    import urllib.error
    err = urllib.error.HTTPError("https://example.com/404", 404, "Not Found", None, None)
    _patch(monkeypatch, urlopen_error=err)
    out = json.loads(_web_fetch_executor("https://example.com/404"))
    assert "error" in out and "404" in out["error"]


def test_network_error_fails_closed(monkeypatch):
    _patch(monkeypatch, urlopen_error=ConnectionError("boom"))
    out = json.loads(_web_fetch_executor("https://example.com/x"))
    assert "error" in out and "web fetch failed" in out["error"]
