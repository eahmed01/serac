"""Permanent tests for the free Google News tool (agent_framework.builtins).

Network is mocked; the tests verify query construction, RSS parsing, the
client-side ``since`` filter, dedupe/sort/bounds, error handling, and the
ToolDef schema (event-only guidance, no identity claims).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from agent_framework.builtins import (
    _parse_rss_date,
    _unwrap_google_news_link,
    google_news_factory,
)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>"VELO" stock - Google News</title>
    <item>
      <title>Velo3D Secures $11.5M Defense Contract</title>
      <link>https://news.google.com/rss/articles/CBMx?oc=5</link>
      <pubDate>Wed, 15 Sep 2026 14:14:53 GMT</pubDate>
      <source url="https://stocktwits.com">Stocktwits</source>
    </item>
    <item>
      <title>Velo3D Stock Climbs Following Entry</title>
      <link>https://news.google.com/rss/articles/CBMx?oc=5</link>
      <pubDate>Wed, 15 Sep 2026 13:02:45 GMT</pubDate>
      <source url="https://stocktitan.net">Stock Titan</source>
    </item>
    <item>
      <title>Older Velo3D Story From March</title>
      <link>https://finance.example.com/velo3d-march</link>
      <pubDate>Tue, 03 Mar 2026 09:00:00 GMT</pubDate>
      <source url="https://finance.example.com">Example Finance</source>
    </item>
    <item>
      <title>Unwrapped Target Story</title>
      <link>https://news.google.com/rss/articles/CBMi?oc=5&amp;url=https%3A%2F%2Fseekingalpha.com%2Farticle%2Fvelo</link>
      <pubDate>Tue, 03 Mar 2026 08:00:00 GMT</pubDate>
      <source url="https://seekingalpha.com">Seeking Alpha</source>
    </item>
  </channel>
</rss>
"""


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, payload=None, error=None, urls=None):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if error is not None:
            raise error
        body = payload if payload is not None else ET.tostring(ET.fromstring(RSS))
        return _FakeResponse(body)

    monkeypatch.setattr("agent_framework.builtins.urllib.request.urlopen", fake_urlopen)
    return calls


def test_tool_def_schema_is_event_only():
    tool = google_news_factory()
    assert tool.name == "google_news"
    assert "event" in tool.description.lower()
    assert "never" in tool.description.lower()  # never for identity
    props = tool.parameters["properties"]
    assert set(props) == {"query", "ticker", "since", "limit"}
    assert tool.parameters["required"] == []
    assert props["limit"]["minimum"] == 1
    assert props["limit"]["maximum"] == 25


def test_executor_requires_query_or_ticker(monkeypatch):
    _patch_urlopen(monkeypatch)
    out = json.loads(google_news_factory().executor())
    assert "error" in out and "required" in out["error"]


def test_executor_builds_bounded_query_and_parses_items(monkeypatch):
    calls = _patch_urlopen(monkeypatch)
    out = json.loads(google_news_factory().executor(ticker="velo", limit=25))
    assert "error" not in out
    assert out["query"] == '"VELO" stock'
    assert "news.google.com/rss/search?" in out["rss_url"]
    # Deduped: 4 items, 1 duplicate link -> 3 unique; sorted newest-first.
    assert out["count"] == 3
    titles = [r["title"] for r in out["results"]]
    assert titles[0] == "Velo3D Secures $11.5M Defense Contract"
    assert all(r["published"] for r in out["results"])
    # The Google News redirect link without a ?url= target is returned as-is
    # (unwrap is a no-op); a real ?url= target is unwrapped to the destination.
    assert out["results"][0]["link"] == "https://news.google.com/rss/articles/CBMx?oc=5"
    assert out["results"][-1]["link"] == "https://seekingalpha.com/article/velo"
    # Request used the tool UA and the RSS Accept header.
    assert calls[0].headers["User-agent"].startswith("ai-agents/")
    assert "rss+xml" in calls[0].headers["Accept"]


def test_executor_applies_since_filter_and_limit(monkeypatch):
    _patch_urlopen(monkeypatch)
    out = json.loads(google_news_factory().executor(ticker="VELO", since="2026-04-01", limit=25))
    # Only the two September 2026 items survive the since cutoff (1 dup removed).
    assert out["count"] == 1
    assert out["results"][0]["published"] >= "2026-04-01"
    out = json.loads(google_news_factory().executor(ticker="VELO", limit=2))
    assert out["count"] == 2


def test_executor_rejects_bad_since(monkeypatch):
    _patch_urlopen(monkeypatch)
    out = json.loads(google_news_factory().executor(ticker="VELO", since="2026-04-01x"))
    assert "error" in out and "since" in out["error"]


def test_executor_passes_free_form_query_through(monkeypatch):
    calls = _patch_urlopen(monkeypatch)
    out = json.loads(google_news_factory().executor(query="Regencell Bioscience split", limit=3))
    assert out["query"] == "Regencell Bioscience split"
    assert "Regencell" in calls[0].full_url


def test_executor_caps_limit(monkeypatch):
    _patch_urlopen(monkeypatch)
    out = json.loads(google_news_factory().executor(ticker="VELO", limit=5000))
    assert out["count"] <= 3  # capped at 25, but only 3 unique items exist


def test_executor_fails_closed_on_bad_xml(monkeypatch):
    _patch_urlopen(monkeypatch, payload=b"<rss><channel><item>")
    out = json.loads(google_news_factory().executor(ticker="VELO"))
    assert "error" in out and "failed" in out["error"]


def test_executor_fails_closed_on_network_error(monkeypatch):
    _patch_urlopen(monkeypatch, error=OSError("boom"))
    out = json.loads(google_news_factory().executor(ticker="VELO"))
    assert "error" in out and "boom" in out["error"]


def test_parse_rss_date_handles_gmt_and_missing():
    assert _parse_rss_date("Wed, 15 Sep 2026 14:14:53 GMT") == "2026-09-15T14:14:53+00:00"
    assert _parse_rss_date("") == ""
    assert _parse_rss_date(None) == ""
    assert _parse_rss_date("not a date") == ""


def test_unwrap_google_news_link():
    # No redirect -> unchanged.
    assert _unwrap_google_news_link("https://example.com/a") == "https://example.com/a"
    # ?url= target -> unwrapped.
    target = _unwrap_google_news_link(
        "https://news.google.com/rss/articles/CBM?oc=5&url=https%3A%2F%2Fseekingalpha.com%2Fx"
    )
    assert target == "https://seekingalpha.com/x"
