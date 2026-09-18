"""
Code history — persistent record of all code executed in the sandbox.

Tracks every execute() and eval() call with auto-incrementing cell IDs
(Jupyter-style). Supports viewing, exporting, and selective export of
individual cells or ranges.

Designed to survive server restarts via file-backed storage.
"""

from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CodeEntry:
    """A single code execution entry."""

    cell_id: int
    timestamp: str  # ISO format
    code: str
    entry_type: str  # "execute" or "eval"
    result_preview: str = ""  # First 200 chars of stdout if available
    success: bool = True


class CodeHistory:
    """Thread-safe, auto-incrementing code history with file persistence.

    Stores every piece of code executed via the sandbox, tagged with
    a monotonically increasing cell ID. Exports to Python files with
    Jupyter-style cell markers.
    """

    def __init__(self, history_file: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._next_id: int = 1
        self._entries: Dict[int, CodeEntry] = {}
        self._history_file = history_file

        # Load from disk if file exists
        if history_file:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load history from a JSONL file."""
        path = pathlib.Path(self._history_file)
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = CodeEntry(**data)
                        self._entries[entry.cell_id] = entry
                        if entry.cell_id >= self._next_id:
                            self._next_id = entry.cell_id + 1
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            pass

    def _append_to_disk(self, entry: CodeEntry) -> None:
        """Append a single entry to the JSONL history file."""
        if not self._history_file:
            return
        try:
            path = pathlib.Path(self._history_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError:
            pass

    def record(
        self,
        code: str,
        entry_type: str = "execute",
        result_preview: str = "",
        success: bool = True,
    ) -> int:
        """Record a code execution and return its cell ID."""
        with self._lock:
            cell_id = self._next_id
            self._next_id += 1

            entry = CodeEntry(
                cell_id=cell_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                code=code,
                entry_type=entry_type,
                result_preview=result_preview[:200] if result_preview else "",
                success=success,
            )

            self._entries[cell_id] = entry
            self._append_to_disk(entry)
            return cell_id

    def list_entries(
        self,
        cell_id: Optional[int] = None,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List code entries, optionally filtered by cell ID or range.

        Args:
            cell_id: Return only this specific cell.
            start_id: Include cells >= this ID.
            end_id: Include cells <= this ID.
        """
        with self._lock:
            if cell_id is not None:
                entry = self._entries.get(cell_id)
                return [asdict(entry)] if entry else []

            ids = sorted(self._entries.keys())
            if start_id is not None:
                ids = [i for i in ids if i >= start_id]
            if end_id is not None:
                ids = [i for i in ids if i <= end_id]
            return [asdict(self._entries[i]) for i in ids]

    def get_source(self, cell_id: int) -> Optional[str]:
        """Get the source code for a specific cell."""
        entry = self._entries.get(cell_id)
        return entry.code if entry else None

    def export_to_file(
        self,
        output_path: str,
        cell_id: Optional[int] = None,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
    ) -> str:
        """Export code history to a Python file.

        Supports exporting all cells, a single cell, or a range.
        Cells are separated by Jupyter-style markers.

        Args:
            output_path: File path to write.
            cell_id: Export only this cell.
            start_id: Export cells >= this ID.
            end_id: Export cells <= this ID.

        Returns:
            The output file path.
        """
        entries = self.list_entries(
            cell_id=cell_id, start_id=start_id, end_id=end_id
        )
        if not entries:
            raise ValueError("No entries to export")

        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            f"# Sandbox code export — {len(entries)} cell(s)",
            f"# Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]

        for entry in entries:
            lines.append(f"# --- Cell {entry['cell_id']} ({entry['timestamp']}) ---")
            lines.append(entry["code"])
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return str(path)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"CodeHistory(entries={len(self._entries)}, next_id={self._next_id})"
