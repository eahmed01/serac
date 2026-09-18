"""
Task queue — file-backed research idea tracking with flock protection.

Maintains a markdown table (human-readable) and a JSONL sidecar (machine-readable).
All operations use flock to prevent concurrent access issues between agents.
"""

import fcntl
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional


class TaskQueue:
    """File-backed task queue for research ideas.

    Parameters
    ----------
    queue_path : str
        Path to the markdown queue file.
    """

    def __init__(self, queue_path: str) -> None:
        self._path = queue_path
        self._jsonl_path = queue_path.rsplit(".", 1)[0] + ".jsonl"
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the queue file if it doesn't exist."""
        if not Path(self._path).exists():
            self._write_queue([])

    def _acquire_lock(self) -> int:
        """Acquire exclusive lock on the queue file."""
        fd = os.open(self._path, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _release_lock(self, fd: int) -> None:
        """Release the lock."""
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    # ── Markdown cell escaping ─────────────────────────────────────
    # Literal pipe `|` in cell content breaks markdown table parsing.
    # We escape on write and unescape on read so the round-trip is lossless.
    #
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _escape_cell(text: str) -> str:
        """Escape pipe characters for safe markdown table cells."""
        return text.replace("|", "&#124;")

    @staticmethod
    def _unescape_cell(text: str) -> str:
        """Undo pipe-escaping after reading a markdown cell."""
        return text.replace("&#124;", "|")

    def _read_queue(self) -> List[Dict]:
        """Read and parse the queue file."""
        tasks: List[Dict] = []
        content = Path(self._path).read_text()
        for line in content.split("\n"):
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|")]
            # Need at least 7 data columns (between pipes) for 6 fields
            if len(parts) < 7:
                continue
            # Skip header and separator rows
            if parts[1] in ("ID", ""):
                continue
            if parts[1].startswith("-"):
                continue
            try:
                # Validate that the ID starts with S
                if not parts[1].startswith("S"):
                    continue
                tasks.append({
                    "id": parts[1],
                    "status": parts[2],
                    "idea": self._unescape_cell(parts[3]),
                    "date": parts[4],
                    "agent": parts[5],
                    "result": self._unescape_cell(parts[6]) if len(parts) > 6 else "",
                    "asset": "",  # hydrated below from JSONL
                })
            except (ValueError, IndexError):
                continue
        # Hydrate asset field from the JSONL sidecar (it's not in markdown)
        self._hydrate_assets(tasks)
        return tasks

    def _hydrate_assets(self, tasks: List[Dict]) -> None:
        """Populate the asset field from the JSONL sidecar.

        The markdown table doesn't have an asset column, so we look up
        the most recent JSONL entry per task ID to fill in .asset.
        """
        if not Path(self._jsonl_path).exists():
            return
        asset_map: Dict[str, str] = {}
        content = Path(self._jsonl_path).read_text()
        for line in content.split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                tid = entry.get("id")
                if tid and entry.get("asset"):
                    asset_map[tid] = entry["asset"]
            except json.JSONDecodeError:
                continue
        for t in tasks:
            t["asset"] = asset_map.get(t["id"], "stocks")

    def _write_queue(self, tasks: List[Dict]) -> None:
        """Write tasks back to the queue file."""
        header = "# Research Queue\n"
        table_header = "\n| ID | Status | Idea | Date | Agent | Result |\n"
        table_sep = "|----|--------|------|------|-------|--------|\n"
        rows = ""
        for t in tasks:
            rows += f"| {t['id']} | {t['status']} | {self._escape_cell(t['idea'])} | {t['date']} | {t['agent']} | {self._escape_cell(t.get('result', ''))} |\n"

        Path(self._path).write_text(header + table_header + table_sep + rows)

    def _write_jsonl(self, task: Dict) -> None:
        """Append a task entry to the JSONL sidecar (atomic write).

        Uses a temp-file + os.replace pattern so a crash mid-write
        never leaves a truncated line on disk.
        """
        entry = json.dumps(task, ensure_ascii=False) + "\n"
        # os.O_APPEND + flock guarantees ordering even under concurrency
        fd = os.open(self._jsonl_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, entry.encode("utf-8"))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _next_id(self, tasks: List[Dict]) -> str:
        """Generate the next task ID."""
        if not tasks:
            return "S001"
        max_num = 0
        for t in tasks:
            try:
                num = int(t["id"][1:])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
        return f"S{max_num + 1:03d}"

    def add(self, idea: str, asset: str = "stocks") -> str:
        """Add a new research idea to the queue.

        Returns the task ID.
        """
        fd = self._acquire_lock()
        try:
            tasks = self._read_queue()
            task_id = self._next_id(tasks)
            new_task = {
                "id": task_id,
                "status": "pending",
                "idea": idea,
                "date": date.today().isoformat(),
                "agent": "",
                "result": "",
                "asset": asset,
            }
            tasks.append(new_task)
            self._write_queue(tasks)
            self._write_jsonl({**new_task, "action": "add"})
            return task_id
        finally:
            self._release_lock(fd)

    def claim(self, task_id: str, agent_id: str) -> Optional[Dict]:
        """Claim a pending task for testing.

        Returns the updated task dict, or None if not found.
        """
        fd = self._acquire_lock()
        try:
            tasks = self._read_queue()
            for t in tasks:
                if t["id"] == task_id and t["status"] == "pending":
                    t["status"] = "testing"
                    t["agent"] = agent_id
                    self._write_queue(tasks)
                    self._write_jsonl({**t, "action": "claim"})
                    return t
            return None
        finally:
            self._release_lock(fd)

    def complete(self, task_id: str, result_path: str) -> Optional[Dict]:
        """Mark a task as validated with a result link.

        Only tasks in "testing" status can be completed (enforces the
        pending → testing → validated state machine).

        Returns the updated task dict, or None if not found / invalid state.
        """
        fd = self._acquire_lock()
        try:
            tasks = self._read_queue()
            for t in tasks:
                if t["id"] == task_id and t["status"] == "testing":
                    t["status"] = "validated"
                    t["result"] = result_path
                    self._write_queue(tasks)
                    self._write_jsonl({**t, "action": "complete"})
                    return t
            return None
        finally:
            self._release_lock(fd)

    def reject(self, task_id: str, reason_path: str) -> Optional[Dict]:
        """Mark a task as rejected with a reason link.

        Only tasks in "testing" status can be rejected (enforces the
        pending → testing → rejected state machine).

        Returns the updated task dict, or None if not found / invalid state.
        """
        fd = self._acquire_lock()
        try:
            tasks = self._read_queue()
            for t in tasks:
                if t["id"] == task_id and t["status"] == "testing":
                    t["status"] = "rejected"
                    t["result"] = reason_path
                    self._write_queue(tasks)
                    self._write_jsonl({**t, "action": "reject"})
                    return t
            return None
        finally:
            self._release_lock(fd)

    def list_tasks(self, status: Optional[str] = None) -> List[Dict]:
        """List tasks, optionally filtered by status."""
        fd = self._acquire_lock()
        try:
            tasks = self._read_queue()
            if status:
                tasks = [t for t in tasks if t["status"] == status]
            return tasks
        finally:
            self._release_lock(fd)
