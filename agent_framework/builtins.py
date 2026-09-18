"""Built-in tool implementations for the agent framework.

Each tool is a standalone factory function that returns a ToolDef.
This enables config-driven loading: swap implementations by pointing
config at a different factory function.

Usage:
    # Config-driven (recommended):
    from agent_framework.tool_loader import load_default_tools
    registry = load_default_tools(sandbox=my_sandbox)

    # Direct import:
    from agent_framework.builtins import read_file_factory
    registry.register(read_file_factory(sandbox=my_sandbox))
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import warnings
import ipaddress
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

from agent_framework.sanitize import wrap_untrusted
from agent_framework.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def _read_file_executor(
    path: str, offset: int = 1, limit: int = 500, sandbox: Any = None, **kwargs: Any
) -> str:
    """Read a file with line numbers and pagination."""
    if sandbox is not None:
        return sandbox.read_file(path)
    _guard_read_path(path)
    with open(path, "r") as f:
        lines = f.readlines()
    total = len(lines)
    selected = lines[offset - 1 : offset + limit - 1]
    result = []
    for i, line in enumerate(selected, start=offset):
        result.append(f"{i}|{line}")
    return f"Lines {offset}-{offset + len(selected) - 1} of {total}:\n" + "".join(result)


def read_file_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for read_file."""
    def executor(path: str, offset: int = 1, limit: int = 500, **kwargs: Any) -> str:
        return _read_file_executor(path, offset, limit, sandbox=sandbox)

    return ToolDef(
        name="read_file",
        description="Read a text file with line numbers and pagination.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "offset": {"type": "integer", "description": "Line number to start from (1-indexed)", "default": 1},
                "limit": {"type": "integer", "description": "Maximum number of lines to read", "default": 500},
            },
            "required": ["path"],
        },
        executor=executor,
        execution_mode="sandbox",
        requires_sandbox=False,
    )


def _write_file_executor(path: str, content: str, sandbox: Any = None, **kwargs: Any) -> str:
    """Write content to a file."""
    if sandbox is not None:
        sandbox.write_file(path, content)
        return f"Written {len(content)} chars to {path}"
    _guard_read_path(path)
    _validate_write_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def write_file_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for write_file."""
    def executor(path: str, content: str, **kwargs: Any) -> str:
        return _write_file_executor(path, content, sandbox=sandbox)

    return ToolDef(
        name="write_file",
        description="Write content to a file, completely replacing existing content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Complete content to write to the file"},
            },
            "required": ["path", "content"],
        },
        executor=executor,
        execution_mode="sandbox",
        requires_sandbox=False,
    )


def _patch_file_executor(
    path: str, old_string: str, new_string: str,
    replace_all: bool = False, sandbox: Any = None, **kwargs: Any
) -> str:
    """Find-and-replace patching."""
    if sandbox is not None:
        content = sandbox.read_file(path)
    else:
        _guard_read_path(path)
        with open(path, "r") as f:
            content = f.read()
        _validate_write_path(path)

    if replace_all:
        content = content.replace(old_string, new_string)
    else:
        content = content.replace(old_string, new_string, 1)

    if sandbox is not None:
        sandbox.write_file(path, content)
    else:
        with open(path, "w") as f:
            f.write(content)

    return f"Patched {path}"


def patch_file_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for patch_file."""
    def executor(
        path: str, old_string: str, new_string: str,
        replace_all: bool = False, **kwargs: Any
    ) -> str:
        return _patch_file_executor(path, old_string, new_string, replace_all, sandbox=sandbox)

    return ToolDef(
        name="patch_file",
        description="Find and replace text in a file. Uses fuzzy matching.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {
                    "type": "string",
                    "description": "Text to find and replace. Must be unique unless replace_all=true.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text. Pass empty string to delete the matched text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences instead of requiring a unique match",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        executor=executor,
        execution_mode="sandbox",
        requires_sandbox=False,
    )


# ---------------------------------------------------------------------------
# Code search
# ---------------------------------------------------------------------------


def _code_search_executor(
    pattern: str, path: str = ".", file_glob: str | None = None,
    context: int = 0, limit: int = 50, **kwargs: Any
) -> str:
    """Search file contents using ripgrep."""
    cmd = ["rg", "--color=never", f"-C{context}", "-n", "--max-count", str(limit), pattern]
    if file_glob:
        cmd.extend(["-g", file_glob])
    cmd.append(path)
    return _run_command(cmd)


def code_search_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for code_search."""
    return ToolDef(
        name="code_search",
        description="Search file contents using ripgrep (fast grep).",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in", "default": "."},
                "file_glob": {"type": "string", "description": "Filter files by glob pattern (e.g., '*.py')"},
                "context": {"type": "integer", "description": "Number of context lines before/after", "default": 0},
                "limit": {"type": "integer", "description": "Maximum number of results", "default": 50},
            },
            "required": ["pattern"],
        },
        executor=_code_search_executor,
    )


def _find_files_executor(pattern: str, path: str = ".", limit: int = 50, **kwargs: Any) -> str:
    """Find files by name pattern."""
    cmd = ["find", path, "-name", pattern, "-type", "f", "-print0"]
    result = _run_command(cmd, binary=True)

    # Handle bytes result (mytype: ignore[call-arg, attr])
    raw_bytes: bytes = result if isinstance(result, bytes) else result.encode("utf-8")
    files = [f.decode("utf-8", errors="replace") for f in raw_bytes.split(b"\0") if f]  # type: ignore[attr-defined]

    return "\n".join(files[:limit]) if files else "(no files found)"


def find_files_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for find_files."""
    return ToolDef(
        name="find_files",
        description="Find files by glob pattern. Equivalent to 'ls' or 'find -name'.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py', '*config*')"},
                "path": {"type": "string", "description": "Directory to search in", "default": "."},
                "limit": {"type": "integer", "description": "Maximum number of results", "default": 50},
            },
            "required": ["pattern"],
        },
        executor=_find_files_executor,
    )


# ---------------------------------------------------------------------------
# Terminal / code execution
# ---------------------------------------------------------------------------


def _execute_terminal_executor(command: str, timeout: int = 180, **kwargs: Any) -> str:
    """Execute a shell command."""
    # Use shell=True to support pipes, redirects, &&, etc.
    return _run_command(command, timeout=timeout, shell=True)


def execute_terminal_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for execute_terminal."""
    def executor(command: str, timeout: int = 180, **kwargs: Any) -> str:
        if sandbox is not None:
            return sandbox.execute_shell(command, timeout=timeout)
        return _execute_terminal_executor(command, timeout)

    return ToolDef(
        name="execute_terminal",
        description="Execute a shell command. Use for git, builds, tests, process management.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute"},
                "timeout": {"type": "integer", "description": "Max seconds to wait", "default": 180},
            },
            "required": ["command"],
        },
        executor=executor,
        execution_mode="sandbox",
        requires_sandbox=True,
        role="worker",
    )


def _execute_python_executor(code: str, timeout: int = 60, **kwargs: Any) -> str:
    """Execute Python code in isolated environment.

    Restricts imports to safe modules only. Blocks os, subprocess, socket, etc.
    """
    # Restrict imports to safe modules
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "isinstance": isinstance,
        "type": type,
        "Exception": Exception,
    }
    restricted_builtins = {"__builtins__": safe_builtins, "__name__": "__sandbox__"}
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, restricted_builtins)
        return sys.stdout.getvalue()
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout


def execute_python_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for execute_python."""
    def executor(code: str, timeout: int = 60, allow_unsafe: bool = False, **kwargs: Any) -> str:
        if sandbox is not None:
            return sandbox.execute_python(code, timeout=timeout)
        if not allow_unsafe:
            raise PermissionError(
                "execute_python is disabled in in-process mode for security. "
                "Use a Docker sandbox or set allow_unsafe=True."
            )
        logger.warning("INSECURE: execute_python running in-process with allow_unsafe=True")
        return _execute_python_executor(code, timeout)

    return ToolDef(
        name="execute_python",
        description="Execute Python code in an isolated environment. "
                    "Requires a Docker sandbox; in-process mode is disabled by default for security.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Max seconds to run", "default": 60},
                "allow_unsafe": {
                    "type": "boolean",
                    "description": "Allow in-process execution (disabled by default). Only set when no sandbox is available.",
                    "default": False,
                },
            },
            "required": ["code"],
        },
        executor=executor,
        execution_mode="sandbox",
        requires_sandbox=True,
    )


# ---------------------------------------------------------------------------
# Research sandbox (persistent Python namespace via model_flow sandbox server)
# ---------------------------------------------------------------------------


def _execute_research_executor(code: str, timeout: int = 120, **kwargs: Any) -> str:
    """Execute Python code in the persistent research sandbox."""
    from agent_framework.research_sandbox import ResearchSandbox

    rs = ResearchSandbox()
    if not rs.is_available:
        return (
            "ERROR: Research sandbox is not available. "
            "Cannot connect to localhost:9876. "
            "Is the sandbox server running? (python3 -m model_flow.sandbox.server)"
        )
    try:
        return rs.execute(code)
    except Exception as e:
        return f"ERROR: {e}"


def research_execute_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for research_execute.

    Connects to the model_flow sandbox server at localhost:9876.
    The persistent namespace has: s (Sandbox), df (OHLCV data),
    np, pd, check_causality, and any previously-defined variables.
    """
    return ToolDef(
        name="research_execute",
        description="Execute Python code in the persistent research sandbox "
                    "(model_flow sandbox server at localhost:9876). "
                    "Access to OHLCV data (df), sandbox instance (s), "
                    "numpy (np), pandas (pd), and causality checker.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to run",
                    "default": 120,
                },
            },
            "required": ["code"],
        },
        executor=_execute_research_executor,
    )


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def git_status_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for git_status."""
    return ToolDef(
        name="git_status",
        description="Show git status (short format).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Git repository path", "default": "."},
            },
            "required": [],
        },
        executor=lambda path=".", **kw: _run_command(["git", "status", "--short"], cwd=path),
    )


def git_diff_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for git_diff."""
    def executor(path: str = ".", stat_only: bool = False, **kwargs: Any) -> str:
        cmd = ["git", "diff"]
        if stat_only:
            cmd.append("--stat")
        return _run_command(cmd, cwd=path)

    return ToolDef(
        name="git_diff",
        description="Show git diff of unstaged changes.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Git repository path", "default": "."},
                "stat_only": {"type": "boolean", "description": "Show only diff stat", "default": False},
            },
            "required": [],
        },
        executor=executor,
    )


def git_log_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for git_log."""
    return ToolDef(
        name="git_log",
        description="Show recent git log entries.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of entries to show", "default": 5},
                "path": {"type": "string", "description": "Git repository path", "default": "."},
            },
            "required": [],
        },
        executor=lambda n=5, path=".", **kw: _run_command(["git", "log", f"-n{n}", "--oneline"], cwd=path),
    )


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


def _sec_search_executor(query: str = "", cik: str = "", max_results: int = 10, **kwargs: Any) -> str:
    """Search SEC submissions for a CIK/company query and return filing links."""
    try:
        if not cik:
            return json.dumps({"error": "cik is required for SEC search"})
        cik10 = str(cik).strip().upper().replace("CIK", "").zfill(10)
        if not cik10.isdigit() or len(cik10) != 10:
            return json.dumps({"error": "cik must be a numeric SEC CIK"})
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Rebo research contact research@example.com"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        recent = data.get("filings", {}).get("recent", {})
        results = []
        for i, form in enumerate(recent.get("form", [])):
            item = {k: recent.get(k, [None] * len(recent.get("form", [])))[i] for k in ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument")}
            haystack = json.dumps(item).lower()
            if not query or query.lower() in haystack:
                acc = str(item["accessionNumber"]).replace("-", "")
                item["url"] = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc}/{item['primaryDocument']}"
                results.append(item)
            if len(results) >= max_results:
                break
        return json.dumps({"company": data.get("name"), "cik": cik10, "query": query, "results": results})
    except Exception as exc:
        return json.dumps({"error": f"SEC search failed: {exc}"})


def sec_search_factory(sandbox: Optional[Any] = None) -> ToolDef:
    return ToolDef(
        name="sec_search",
        description="Search a company's SEC submissions by CIK and optional form/date/document query; returns filing metadata and SEC archive URLs.",
        parameters={"type": "object", "properties": {"cik": {"type": "string"}, "query": {"type": "string"}, "max_results": {"type": "integer", "default": 10}}, "required": ["cik"]},
        executor=_sec_search_executor,
    )


def sec_fetch_factory(sandbox: Optional[Any] = None) -> ToolDef:
    return ToolDef(
        name="sec_fetch",
        description="Fetch the text of one SEC filing from its CIK, accession number, and primary document name.",
        parameters={"type": "object", "properties": {"cik": {"type": "string"}, "accession_number": {"type": "string"}, "document_name": {"type": "string"}, "max_chars": {"type": "integer", "default": 50000}}, "required": ["cik", "accession_number", "document_name"]},
        executor=_sec_fetch_executor,
    )


# ---------------------------------------------------------------------------
# Google News (free RSS endpoint — no API key)
# ---------------------------------------------------------------------------


def _parse_rss_date(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        from datetime import timezone
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(value)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def _unwrap_google_news_link(url: str) -> str:
    try:
        from urllib.parse import urlparse, parse_qs
        p = urlparse(url)
        if p.netloc.endswith("news.google.com"):
            target = parse_qs(p.query).get("url", [None])[0]
            if target:
                return target
    except Exception:
        pass
    return url


def _google_news_executor(
    query: str = "",
    ticker: str = "",
    since: str = "",
    limit: int = 10,
    lang: str = "en-US",
    region: str = "US",
    **kwargs: Any,
) -> str:
    """Search the free Google News RSS endpoint. No API key required.

    Returns bounded JSON: query, rss_url, and a list of {title, link,
    published, source}. Google News redirect links are unwrapped to the target
    domain where possible. Results are deduped, sorted newest-first, and
    capped at limit.

    Query shape is deliberately simple — the RSS endpoint does not handle
    complex boolean queries: a bare ticker becomes '"TICKER" stock' (quoted
    ticker plus one finance context word), and a free-form query is passed
    through verbatim. '-site:' negations empirically degrade or return
    unrelated results on this endpoint, so they are not used.
    """
    try:
        import re as _re
        import xml.etree.ElementTree as ET
        from datetime import datetime
        from urllib.parse import urlencode

        q = (query or "").strip() or (ticker or "").strip().upper()
        if not q:
            return json.dumps({"error": "query or ticker is required"})

        final_q = q if (query or "").strip() else f'"{q}" stock'

        try:
            n = max(1, min(int(limit), 25))
        except Exception:
            n = 10
        if since and not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
            return json.dumps({"error": "since must be YYYY-MM-DD"})

        rss_url = "https://news.google.com/rss/search?" + urlencode({
            "q": final_q, "hl": lang, "gl": region, "ceid": f"{region}:{lang.split('-')[0]}",
        })
        headers = {
            "User-Agent": "ai-agents/google-news (contact@example.com)",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        }
        req = urllib.request.Request(rss_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            root = ET.fromstring(resp.read())

        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since + "T00:00:00+00:00")

        seen = set()
        items = []
        for item in root.findall(".//item"):
            def _text(tag):
                el = item.find(tag)
                return (el.text or "").strip() if el is not None and el.text else ""
            title = _text("title")[:200]
            link = _unwrap_google_news_link(_text("link"))
            published_iso = _parse_rss_date(_text("pubDate"))
            source_el = item.find("source")
            source = (source_el.text or "").strip()[:80] if source_el is not None and source_el.text else ""
            key = link or title
            if not key or key in seen:
                continue
            seen.add(key)
            if since_dt is not None and published_iso:
                try:
                    dt_item = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
                    if dt_item < since_dt:
                        continue
                except Exception:
                    pass
            items.append({"title": title, "link": link, "published": published_iso, "source": source})
        items.sort(key=lambda x: x.get("published") or "", reverse=True)
        items = items[:n]
        return json.dumps({"query": final_q, "rss_url": rss_url, "count": len(items), "results": items})
    except Exception as exc:
        return json.dumps({"error": f"google news search failed: {exc}"})


def google_news_factory(sandbox: Optional[Any] = None) -> ToolDef:
    return ToolDef(
        name="google_news",
        description="Search recent news headlines for a ticker or free-form query via the free Google News RSS endpoint. Returns bounded titles/links/dates/sources. Use only for event/catalyst context, never for legal-entity identity (use SEC tools for identity).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-form news query, e.g. 'Velo3D reverse split' or a ticker."},
                "ticker": {"type": "string", "description": "Ticker to search (quoted exact-match). Used when query is empty."},
                "since": {"type": "string", "description": "Only items published on/after this date (YYYY-MM-DD)."},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 25, "description": "Max headlines (default 10)."},
            },
            "required": [],
        },
        executor=_google_news_executor,
    )


def _sec_fetch_executor(cik: str, accession_number: str, document_name: str, max_chars: int = 50000, **kwargs: Any) -> str:
    """Fetch bounded SEC filing text from the SEC Archives."""
    try:
        cik10 = str(cik).strip().upper().replace("CIK", "").zfill(10)
        acc = str(accession_number).strip().replace("-", "")
        doc = str(document_name).strip()
        if not cik10.isdigit() or len(cik10) != 10 or not acc.isdigit() or not doc or "/" in doc or "\\" in doc or ".." in doc:
            return json.dumps({"error": "invalid CIK, accession number, or document name"})
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc}/{urllib.parse.quote(doc)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Rebo research contact research@example.com"})
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read(int(max_chars) * 4).decode("utf-8", errors="replace")
        import re
        text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()[:max(1000, min(int(max_chars), 200000))]
        return json.dumps({"cik": cik10, "accession_number": accession_number, "document_name": doc, "url": url, "text": text})
    except Exception as exc:
        return json.dumps({"error": f"SEC fetch failed: {exc}"})


# ---------------------------------------------------------------------------
# Web fetch (bounded, sanitized, event/explanation tier)
# ---------------------------------------------------------------------------

DEFAULT_WEB_FETCH_MAX_CHARS = 20000
_MAX_WEB_FETCH_RAW_BYTES = 400_000  # raw byte cap before text extraction


def _web_fetch_executor(url: str = "", max_chars: int = DEFAULT_WEB_FETCH_MAX_CHARS, **kwargs: Any) -> str:
    """Bounded fetch of one public URL, sanitized to plain text.

    Event/explanation tier only — same trust boundary as google_news: it may
    back a catalyst explanation, never identity, price, corporate-action, or
    adjustment claims. Only http/https to a public host is allowed; localhost,
    private/link-local ranges, and non-http schemes are refused (fail-closed).
    Text extraction strips script/style/tags, collapses whitespace, and caps
    output at max_chars (default 20k). Errors return fail-closed JSON.
    """
    import re
    import socket
    import urllib.error
    from urllib.parse import urlparse

    try:
        cap = max(1000, min(int(max_chars), 200_000))
    except Exception:
        cap = DEFAULT_WEB_FETCH_MAX_CHARS

    raw_url = (url or "").strip()
    if not raw_url:
        return json.dumps({"error": "url is required"})
    try:
        parsed = urlparse(raw_url)
    except Exception as exc:
        return json.dumps({"error": f"invalid url: {exc}"})

    if parsed.scheme not in {"http", "https"}:
        return json.dumps({"error": "only http/https URLs are allowed"})
    host = (parsed.hostname or "").lower()
    if not host:
        return json.dumps({"error": "url has no host"})

    # Resolve and refuse localhost / private / link-local / reserved ranges.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except Exception as exc:
        return json.dumps({"error": f"could not resolve host: {exc}"})
    try:
        for info in infos:
            ip = info[4][0]
            addr = ipaddress.ip_address(ip)
            if (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_reserved
                or addr.is_multicast
                or addr.is_unspecified
            ):
                return json.dumps({"error": "url resolves to a private or non-public address"})
    except Exception as exc:
        return json.dumps({"error": f"could not validate host address: {exc}"})

    headers = {
        "User-Agent": "Rebo research contact research@example.com",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
    }
    req = urllib.request.Request(raw_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read(_MAX_WEB_FETCH_RAW_BYTES)
            status = getattr(response, "status", None) or response.getcode()
    except urllib.error.HTTPError as exc:
        return json.dumps({"error": f"HTTP {exc.code}: {exc.reason}", "status": exc.code})
    except Exception as exc:
        return json.dumps({"error": f"web fetch failed: {exc}"})

    content_type = ""
    try:
        content_type = (response.headers.get("Content-Type", "") if hasattr(response, "headers") else "") or ""
    except Exception:
        content_type = ""

    # Binary / non-text content: fail closed.
    text_head = content_type.lower()
    if any(marker in text_head for marker in ("application/", "image/", "audio/", "video/", "octet-stream")) \
            and "text" not in text_head and "html" not in text_head and "xml" not in text_head:
        return json.dumps({"error": "content type is not fetchable text", "content_type": content_type})

    body = raw.decode("utf-8", errors="replace")
    text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", body, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    truncated = len(text) > cap
    text = text[:cap]
    result = {
        "url": raw_url,
        "status": status,
        "content_type": content_type,
        "chars": len(text),
        "truncated": truncated,
        "text": text,
    }
    return json.dumps(result)


def web_fetch_factory(sandbox: Optional[Any] = None) -> ToolDef:
    return ToolDef(
        name="web_fetch",
        description=(
            "Fetch and sanitize the text of one public http/https URL (an article or notice "
            "found via google_news or sec_search). Bounded to a max_chars text cap. Use ONLY for "
            "event/catalyst context — never for legal-entity identity, price, corporate-action, "
            "or adjustment verification (use SEC tools and the deterministic evidence packet for those)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full http/https URL to fetch."},
                "max_chars": {"type": "integer", "default": DEFAULT_WEB_FETCH_MAX_CHARS,
                               "minimum": 1000, "maximum": 200000,
                               "description": "Max characters of extracted text to return (default 20000)."},
            },
            "required": ["url"],
        },
        executor=_web_fetch_executor,
    )


def _web_search_executor(query: str, max_results: int = 5, **kwargs: Any) -> str:
    """Execute a web search — prefers Exa, falls back to DuckDuckGo.

    Outbound: queries sanitized via tool_security framework.
    Inbound: results wrapped with boundary markers.
    Usage: all calls tracked via global tool usage tracker.
    """
    from agent_framework.tool_security import (
        ToolCallStatus,
        has_sensitive_info,
        sanitize_outbound,
        track_tool,
    )
    from agent_framework.sanitize import wrap_untrusted

    # Sanitize query before sending
    was_sanitized = has_sensitive_info(query)
    clean_query = sanitize_outbound(query) if was_sanitized else query

    # Try Exa first (requires EXA_API_KEY)
    start_time = time.monotonic()
    result = _exa_search_executor(clean_query, max_results)
    duration_ms = (time.monotonic() - start_time) * 1000

    if result and not result.startswith("ERROR:"):
        result_count = len([l for l in result.split("\n\n") if l.strip()])
        track_tool(
            tool_name="web_search",
            duration_ms=duration_ms,
            sanitized=was_sanitized,
            original_query=query[:200],
            sanitized_query=clean_query[:200],
            result_count=result_count,
            status=ToolCallStatus.SUCCESS,
        )
        return wrap_untrusted(result)

    # Fall back to DuckDuckGo
    import json as _json
    import urllib.request as _urllib
    import urllib.parse as _urllib_parse

    url = "https://api.duckduckgo.com/?q={}&format=json&no_html=1".format(
        _urllib_parse.quote(clean_query)
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    req = _urllib.Request(url, headers=headers)

    start_time = time.monotonic()
    try:
        with _urllib.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                track_tool("web_search", duration_ms=(time.monotonic() - start_time) * 1000,
                          sanitized=was_sanitized, original_query=query[:200],
                          sanitized_query=clean_query[:200], status=ToolCallStatus.SUCCESS)
                return wrap_untrusted("Search returned no results.")
            data = _json.loads(raw)
    except Exception as exc:
        duration_ms = (time.monotonic() - start_time) * 1000
        track_tool("web_search", duration_ms=duration_ms,
                   sanitized=was_sanitized, original_query=query[:200],
                   sanitized_query=clean_query[:200], status=ToolCallStatus.FAILED,
                   error=str(exc)[:200])
        return wrap_untrusted(f"Search failed: {exc}")

    duration_ms = (time.monotonic() - start_time) * 1000

    # Format results
    lines = []

    if data.get("AbstractURL") or data.get("Text"):
        lines.append(
            f"Instant answer:\n"
            f"  Title: {data.get('Heading', 'N/A')}\n"
            f"  URL: {data.get('AbstractURL', '')}\n"
            f"  {data.get('Text', '')}"
        )

    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and topic.get("Text"):
            lines.append(
                f"- {topic['Text']}\n"
                f"  {topic.get('FirstURL', '')}"
            )
        elif isinstance(topic, dict) and topic.get("Topics"):
            for sub in topic["Topics"][:max_results]:
                lines.append(
                    f"- {sub.get('Text', '')}\n"
                    f"  {sub.get('FirstURL', '')}"
                )

    if not lines:
        track_tool("web_search", duration_ms=duration_ms,
                   sanitized=was_sanitized, original_query=query[:200],
                   sanitized_query=clean_query[:200], status=ToolCallStatus.SUCCESS)
        return wrap_untrusted("No results found.")

    result_text = "\n\n".join(lines[:max_results])
    result_count = len(lines[:max_results])
    track_tool("web_search", duration_ms=duration_ms,
               sanitized=was_sanitized, original_query=query[:200],
               sanitized_query=clean_query[:200], result_count=result_count,
               status=ToolCallStatus.SUCCESS)
    return wrap_untrusted(result_text)


def _exa_search_executor(query: str, max_results: int = 5, **kwargs: Any) -> str:
    """Execute a web search via Exa API (https://exa.ai)."""
    import os
    import urllib.request

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return ""  # Let caller fall back to DuckDuckGo

    url = "https://api.exa.ai/search"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    payload = {
        "query": query,
        "numResults": max_results,
        "type": "neural",
        "useAutoprompt": True,
        "contents": {"highlights": True},
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return f"ERROR: Exa search failed: {exc}"

    # Format results
    lines = []
    for i, item in enumerate(result.get("results", [])[:max_results], 1):
        # Get highlights if available
        highlights = item.get("highlights", [])
        highlight_text = "\n".join(highlights[:3]) if highlights else ""
        lines.append(
            f"{i}. {item.get('title', 'Untitled')}\n"
            f"   {item.get('url', '')}\n"
            f"   {highlight_text}"
        )

    if not lines:
        return "No results found."

    return "\n\n".join(lines)


def web_search_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for web_search."""
    return ToolDef(
        name="web_search",
        description="Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        executor=_web_search_executor,
    )


# ---------------------------------------------------------------------------
# Legacy compatibility: register_builtin_tools
# ---------------------------------------------------------------------------


def register_builtin_tools(registry: ToolRegistry, sandbox: Optional[Any] = None) -> None:
    """Legacy: register all built-in tools in a ToolRegistry.

    .. deprecated:: Use load_default_tools() instead.
    """
    warnings.warn(
        "register_builtin_tools() is deprecated. Use load_default_tools() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    factories = (
        read_file_factory,
        write_file_factory,
        patch_file_factory,
        code_search_factory,
        find_files_factory,
        execute_terminal_factory,
        execute_python_factory,
        git_status_factory,
        git_diff_factory,
        git_log_factory,
    )
    for factory in factories:
        tool = factory(sandbox)
        # Match ToolLoader's admission policy while preserving this deprecated
        # API's omission-based compatibility: sandbox-backed tools are the only
        # tools admitted when a sandbox is supplied, while sandbox-required
        # tools are omitted when no sandbox is available.
        if sandbox is not None and tool.execution_mode != "sandbox":
            continue
        if sandbox is None and tool.requires_sandbox:
            continue
        registry.register(tool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_command(
    cmd: list[str] | str, timeout: int = 180, binary: bool = False, **kwargs: Any
) -> str:
    """Run a command and return stdout."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=not binary,
        timeout=timeout,
        **kwargs,
    )
    if result.returncode != 0:
        return f"Exit {result.returncode}: {result.stderr or '(no error output)'}"
    return result.stdout


# ---------------------------------------------------------------------------
# Todo list (task management)
# ---------------------------------------------------------------------------


def todo_factory(sandbox: Optional[Any] = None) -> ToolDef:
    """Factory: returns ToolDef for todo (task management)."""
    def executor(**kwargs: Any) -> str:
        from agent_framework.todo import _todo_executor
        return _todo_executor(**kwargs)

    return ToolDef(
        name="todo",
        description="Manage a persistent todo list for task tracking. "
                    "Tasks can be created, updated, deleted, and reordered. "
                    "The list is shared and persists across turns.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'list' (default), 'add', 'update', 'delete', 'reorder', 'clear', 'set', 'summary'",
                    "default": "list",
                },
                "content": {
                    "type": "string",
                    "description": "Task content for 'add' or 'update' actions",
                    "default": "",
                },
                "item_id": {
                    "type": "string",
                    "description": "Item id for 'update', 'delete' actions",
                    "default": "",
                },
                "status": {
                    "type": "string",
                    "description": "Status for 'update' action: pending, in_progress, completed, cancelled",
                    "default": "",
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of item ids for 'reorder' action",
                },
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                    "description": "List of todo dicts for 'set' action (full replace or merge)",
                },
                "merge": {
                    "type": "boolean",
                    "description": "For 'set' action: if true, merge with existing instead of replacing",
                    "default": False,
                },
            },
            "required": [],
        },
        executor=executor,
    )


def _guard_read_path(path: str) -> None:
    """Guard against path traversal and symlink attacks on in-process file reads.

    Resolves symlinks and ensures the real path lives under allowed roots.
    Allowed: project root (cwd) only. Raises PermissionError otherwise.
    """
    resolved_path = Path(path).resolve(strict=False)
    cwd = Path.cwd().resolve()
    try:
        resolved_path.relative_to(cwd)
    except ValueError:
        raise PermissionError(
            f"Path outside project root: {path} -> {resolved_path}"
        ) from None


def _validate_write_path(path: str) -> None:
    """Validate that a write path is safe.

    Uses an allowlist: only paths under project root (cwd) are writable.
    """
    # strict=False is intentional: a new file (or its new parent) need not
    # exist yet. Existing symlinks are still resolved, so a symlinked parent
    # cannot redirect the write outside the project root.
    resolved_path = Path(path).resolve(strict=False)
    cwd = Path.cwd().resolve()
    try:
        resolved_path.relative_to(cwd)
    except ValueError:
        raise PermissionError(
            f"Write denied: {path} (outside project root)"
        ) from None
