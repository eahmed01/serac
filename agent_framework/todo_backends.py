#!/usr/bin/env python3
"""Todo store protocol and implementations for the agent framework.

Provides a modular, swappable todo store interface. Different implementations
can be loaded via config, enabling easy swapping and comparison with alternative
implementations.

Protocol:
    Any class implementing TodoStoreBackend can be used as a todo store.
    The protocol defines: list_items, add_item, update_item, delete_item,
    reorder, clear_completed, summary.

Implementations:
    - FileTodoStore: JSON file-backed store (default)
    - MemoryTodoStore: In-memory store (for testing)
    - Custom: any class implementing TodoStoreBackend

Config format:
    {
        "todo_store": "agent_framework.todo_backends:FileTodoStore",
        "todo_store_config": {
            "path": "/path/to/.todo.json",
        }
    }

Usage:
    # Direct:
    from agent_framework.todo_backends import FileTodoStore
    store = FileTodoStore(path="/path/to/.todo.json")

    # Config-driven:
    from agent_framework.todo_backends import load_todo_store
    store = load_todo_store(config)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TodoItem data model
# ---------------------------------------------------------------------------

TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
"""Valid todo status values."""

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


@dataclass
class TodoItem:
    """A single todo item."""
    id: str
    content: str
    status: TodoStatus

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TodoItem":
        import uuid

        item_id = data.get("id", "")
        if not item_id:
            item_id = str(uuid.uuid4())[:8]
            logging.getLogger(__name__).warning(
                "Missing todo id, generated %s", item_id
            )

        content = data.get("content", "")
        if content is None:
            content = ""

        status = data.get("status", "pending")
        if status not in VALID_STATUSES:
            logging.getLogger(__name__).warning(
                "Invalid todo status %r, defaulting to 'pending'", status
            )
            status = "pending"
        return cls(
            id=item_id,
            content=content,
            status=status,  # type: ignore[typeddict-item]
        )


# ---------------------------------------------------------------------------
# TodoStoreBackend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TodoStoreBackend(Protocol):
    """Protocol for todo store implementations.

    Any class implementing this protocol can be used as a todo store backend.
    """

    def list_items(self) -> list[TodoItem]:
        """Return all items in order."""
        ...

    def add_item(self, content: str) -> TodoItem:
        """Add a new pending item at the end."""
        ...

    def update_item(self, item_id: str, content: Optional[str] = None,
                     status: Optional[str] = None) -> TodoItem:
        """Update an existing item by id."""
        ...

    def delete_item(self, item_id: str) -> None:
        """Delete an item by id."""
        ...

    def reorder(self, item_ids: list[str]) -> list[TodoItem]:
        """Reorder items to match the given id sequence."""
        ...

    def clear_completed(self) -> int:
        """Remove all completed items. Returns count removed."""
        ...

    def summary(self) -> dict:
        """Get a summary of todo state."""
        ...


# ---------------------------------------------------------------------------
# File-backed implementation
# ---------------------------------------------------------------------------

class FileTodoStore:
    """File-backed todo store with atomic read/write.

    State is stored as a JSON file. Reads parse the file, writes
    atomically replace it (write to tmp, then rename). This means
    any process reading the file sees the latest state.

    Args:
        path: Absolute path to the JSON file. Created if missing.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the file with empty state if it doesn't exist."""
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._write_raw({"items": [], "created_at": time.time()})

    def _read_raw(self) -> dict:
        """Read the raw JSON file."""
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted or missing files gracefully
            logging.getLogger(__name__).warning(
                "Corrupted or missing todo file at %s, recreating", self.path
            )
            return {"items": [], "created_at": time.time()}

    def _write_raw(self, data: dict) -> None:
        """Atomically write the JSON file."""
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def _read(self) -> list[TodoItem]:
        """Read all todo items."""
        data = self._read_raw()
        return [TodoItem.from_dict(item) for item in data.get("items", [])]

    def _write(self, items: list[TodoItem]) -> None:
        """Write all todo items atomically."""
        data = self._read_raw()  # Preserve metadata
        data["items"] = [item.to_dict() for item in items]
        data["updated_at"] = time.time()
        self._write_raw(data)

    def list_items(self) -> list[TodoItem]:
        """Return all items in order."""
        return self._read()

    def add_item(self, content: str) -> TodoItem:
        """Add a new pending item at the end."""
        items = self._read()
        new_item = TodoItem(
            id=str(uuid.uuid4())[:8],
            content=content,
            status="pending",
        )
        items.append(new_item)
        self._write(items)
        return new_item

    def update_item(self, item_id: str, content: Optional[str] = None,
                     status: Optional[str] = None) -> TodoItem:
        """Update an existing item by id."""
        items = self._read()
        updated_item: Optional[TodoItem] = None
        for idx, item in enumerate(items):
            if item.id == item_id:
                if content is not None:
                    item.content = content
                if status is not None:
                    if status not in VALID_STATUSES:
                        raise ValueError(
                            f"Invalid status {status!r}. Must be one of: {VALID_STATUSES}"
                        )
                    item.status = status  # type: ignore[assignment]
                items[idx] = item
                updated_item = item
                self._write(items)
                break
        if updated_item is None:
            raise ValueError(f"Todo item {item_id} not found")
        return updated_item

    def delete_item(self, item_id: str) -> None:
        """Delete an item by id."""
        items = self._read()
        new_items = [item for item in items if item.id != item_id]
        if len(new_items) == len(items):
            raise ValueError(f"Todo item {item_id} not found")
        self._write(new_items)

    def reorder(self, item_ids: list[str]) -> list[TodoItem]:
        """Reorder items to match the given id sequence."""
        items = self._read()
        id_map = {item.id: item for item in items}
        reordered: list[TodoItem] = []
        seen_ids = set()
        for rid in item_ids:
            if rid in id_map:
                reordered.append(id_map[rid])
                seen_ids.add(rid)
        # Append any items not in the reorder list
        for item in items:
            if item.id not in seen_ids:
                reordered.append(item)
        self._write(reordered)
        return reordered

    def clear_completed(self) -> int:
        """Remove all completed items. Returns count removed."""
        items = self._read()
        before = len(items)
        items = [item for item in items if item.status != "completed"]
        if len(items) < before:
            self._write(items)
        return before - len(items)

    def summary(self) -> dict:
        """Get a summary of todo state."""
        items = self._read()
        data = self._read_raw()
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
        for item in items:
            if item.status in counts:
                counts[item.status] += 1
        return {
            "total": len(items),
            "counts": counts,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        }


# ---------------------------------------------------------------------------
# Memory-backed implementation (for testing)
# ---------------------------------------------------------------------------

class MemoryTodoStore:
    """In-memory todo store (no persistence).

    Useful for testing or single-session usage.
    """

    def __init__(self) -> None:
        self._items: list[TodoItem] = []
        self._created_at = time.time()

    def list_items(self) -> list[TodoItem]:
        return list(self._items)

    def add_item(self, content: str) -> TodoItem:
        item = TodoItem(
            id=str(uuid.uuid4())[:8],
            content=content,
            status="pending",
        )
        self._items.append(item)
        return item

    def update_item(self, item_id: str, content: Optional[str] = None,
                     status: Optional[str] = None) -> TodoItem:
        for idx, item in enumerate(self._items):
            if item.id == item_id:
                if content is not None:
                    item.content = content
                if status is not None:
                    if status not in VALID_STATUSES:
                        raise ValueError(
                            f"Invalid status {status!r}. Must be one of: {VALID_STATUSES}"
                        )
                    item.status = status  # type: ignore[assignment]
                self._items[idx] = item
                return item
        raise ValueError(f"Todo item {item_id} not found")

    def delete_item(self, item_id: str) -> None:
        before = len(self._items)
        self._items = [item for item in self._items if item.id != item_id]
        if len(self._items) == before:
            raise ValueError(f"Todo item {item_id} not found")

    def reorder(self, item_ids: list[str]) -> list[TodoItem]:
        id_map = {item.id: item for item in self._items}
        reordered: list[TodoItem] = []
        for rid in item_ids:
            if rid in id_map:
                reordered.append(id_map[rid])
        for item in self._items:
            if item.id not in [r.id for r in reordered]:
                reordered.append(item)
        self._items = reordered
        return reordered

    def clear_completed(self) -> int:
        before = len(self._items)
        self._items = [item for item in self._items if item.status != "completed"]
        return before - len(self._items)

    def summary(self) -> dict:
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
        for item in self._items:
            if item.status in counts:
                counts[item.status] += 1
        return {
            "total": len(self._items),
            "counts": counts,
            "created_at": self._created_at,
            "updated_at": time.time(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_todo_store(
    config: dict[str, Any],
    workspace: Optional[str] = None,
) -> TodoStoreBackend:
    """Load a todo store backend from config.

    Args:
        config: Config dict with keys:
            - "todo_store": "module:ClassName" (optional, defaults to FileTodoStore)
            - "todo_store_config": dict of kwargs for the backend (optional)
        workspace: Workspace path (for default file path)

    Returns:
        TodoStoreBackend instance.
    """
    # Determine store class
    store_spec = config.get("todo_store", "agent_framework.todo_backends:FileTodoStore")
    if ":" in store_spec:
        module_path, class_name = store_spec.rsplit(":", 1)
    else:
        raise ValueError(f"Invalid todo_store spec: {store_spec} (expected 'module:ClassName')")

    # Import dynamically
    import importlib
    module = importlib.import_module(module_path)
    store_class = getattr(module, class_name)

    # Build kwargs
    store_config = config.get("todo_store_config", {})

    # Handle path resolution
    if "path" not in store_config and workspace:
        store_config["path"] = os.path.join(workspace, ".todo.json")
    elif "path" not in store_config:
        store_config["path"] = os.path.join(os.getcwd(), ".todo.json")

    # Filter out path for backends that don't accept it (e.g. MemoryTodoStore)
    import inspect
    sig = inspect.signature(store_class)
    if "path" not in sig.parameters:
        store_config.pop("path", None)

    # Instantiate
    return store_class(**store_config)
