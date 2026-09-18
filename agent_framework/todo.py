#!/usr/bin/env python3
"""Todo list tool for the agent framework.

Persistent task list that can be shared between consultant and orchestrator.
Both can create, edit, rearrange, and delete tasks. Changes are propagated
via shared workspace file.

Architecture:
    - todo_backends.py: TodoStoreBackend protocol + implementations
    - This module: _todo_executor (CRUD operations via backend)

System prompt pattern (models are trained on this):
    "Manage your task list. Use this tool to track work items.
     Create tasks at the start, mark them in_progress as you work,
     completed when done. Only one task in_progress at a time."
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agent_framework.todo_backends import TodoItem, TodoStoreBackend, load_todo_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Todo store registry (per-workspace caching)
# ---------------------------------------------------------------------------

_store_cache: dict[str, TodoStoreBackend] = {}


def _get_store(
    todo_path: Optional[str] = None,
    workspace: Optional[str] = None,
    config: Optional[dict] = None,
) -> TodoStoreBackend:
    """Get or create a todo store for the given workspace."""
    # Config-driven loading (allows swapping backends)
    if config:
        return load_todo_store(config, workspace=workspace)

    # Default: file-backed store
    if todo_path:
        cache_key = todo_path
    elif workspace:
        cache_key = os.path.join(workspace, ".todo.json")
    else:
        cache_key = os.path.join(os.getcwd(), ".todo.json")

    if cache_key not in _store_cache:
        from agent_framework.todo_backends import FileTodoStore
        _store_cache[cache_key] = FileTodoStore(path=cache_key)
    return _store_cache[cache_key]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _todo_executor(
    action: str = "list",
    item_id: str = "",
    content: str = "",
    status: str = "",
    item_ids: list[str] | None = None,
    todos: list[dict] | None = None,
    merge: bool = False,
    **kwargs: Any,
) -> str:
    """Execute a todo list operation.

    Args:
        action: One of: list, add, update, delete, reorder, clear, set, summary
        item_id: Item id for update/delete operations
        content: Task content for add/update
        status: Status for update (pending|in_progress|completed|cancelled)
        item_ids: Ordered list of ids for reorder
        todos: List of todo dicts for 'set' action (full replace)
        merge: If True with 'set', merge into existing instead of replacing
        todo_path: Path to todo file (optional, defaults to workspace/.todo.json)
        workspace: Workspace path (optional, for resolving todo_path)
        todo_config: Config dict for loading backend (optional, enables modular backend)

    Returns:
        Formatted todo list or operation result.
    """
    # Resolve store
    todo_config = kwargs.get("todo_config")
    workspace = kwargs.get("workspace", os.environ.get("AGENT_WORKSPACE", "."))
    store = _get_store(
        todo_path=kwargs.get("todo_path"),
        workspace=workspace,
        config=todo_config,
    )

    # --- list (default) ---
    if action == "list":
        items = store.list_items()
        if not items:
            return "Todo list is empty. Add items with action='add'."
        lines = ["Todo list:", ""]
        for i, item in enumerate(items, 1):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[-]",
                "completed": "[x]",
                "cancelled": "[~]",
            }.get(item.status, "[?]")
            lines.append(f"{i}. {status_icon} {item.content}  (id={item.id}, status={item.status})")
        return "\n".join(lines)

    # --- add ---
    if action == "add":
        if not content:
            return "Error: 'content' is required for action='add'"
        item = store.add_item(content)
        return f"Added: {item.content} (id={item.id})"

    # --- update ---
    if action == "update":
        if not item_id:
            return "Error: 'item_id' is required for action='update'"
        if not content and not status:
            return "Error: at least one of 'content' or 'status' is required for action='update'"
        try:
            item = store.update_item(item_id, content=content, status=status)
        except ValueError as e:
            return f"Error: {e}"
        return f"Updated: {item.content} (id={item.id}, status={item.status})"

    # --- delete ---
    if action == "delete":
        if not item_id:
            return "Error: 'item_id' is required for action='delete'"
        try:
            store.delete_item(item_id)
        except ValueError as e:
            return f"Error: {e}"
        return f"Deleted item {item_id}"

    # --- reorder ---
    if action == "reorder":
        if not item_ids:
            return "Error: 'item_ids' is required for action='reorder'"
        items = store.reorder(item_ids)
        lines = ["Reordered:", ""]
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item.content}")
        return "\n".join(lines)

    # --- clear ---
    if action == "clear":
        removed = store.clear_completed()
        return f"Cleared {removed} completed items."

    # --- set (full replace or merge) ---
    if action == "set":
        if not todos:
            return "Error: 'todos' list is required for action='set'"

        # Check if we have a FileTodoStore with _write method
        has_write = hasattr(store, "_write")

        # Ensure all todos have ids - avoid collisions with existing or other new items
        import uuid
        all_existing_ids = {item.id for item in store.list_items()} if merge else set()
        for td in todos:
            while not td.get("id") or td["id"] in all_existing_ids:
                td["id"] = str(uuid.uuid4())[:8]
            all_existing_ids.add(td["id"])

        if merge:
            # Merge: update existing by id, add new ones
            existing = store.list_items()
            id_map = {item.id: item for item in existing}
            for td in todos:
                tid = td.get("id", "")
                if tid and tid in id_map:
                    id_map[tid].content = td.get("content", id_map[tid].content)
                    id_map[tid].status = td.get("status", id_map[tid].status)
                else:
                    id_map[tid] = TodoItem(
                        id=tid,
                        content=td.get("content", ""),
                        status=td.get("status", "pending"),
                    )
            # Preserve order: existing first (in order), then new
            ordered = []
            seen = set()
            for item in existing:
                if item.id in id_map:
                    ordered.append(id_map[item.id])
                    seen.add(item.id)
            for item in id_map.values():
                if item.id not in seen:
                    ordered.append(item)

            if has_write:
                store._write(ordered)  # type: ignore[attr-defined]
            else:
                # Fallback: delete all and re-add
                for item in existing:
                    try:
                        store.delete_item(item.id)
                    except ValueError:
                        pass
                for item in ordered:
                    store.add_item(item.content)
        else:
            # Full replace
            new_items = [
                TodoItem(
                    id=td.get("id", ""),
                    content=td.get("content", ""),
                    status=td.get("status", "pending"),
                )
                for td in todos
            ]
            if has_write:
                store._write(new_items)  # type: ignore[attr-defined]
            else:
                # Fallback
                existing = store.list_items()
                for item in existing:
                    try:
                        store.delete_item(item.id)
                    except ValueError:
                        pass
                for item in new_items:
                    store.add_item(item.content)
        # Return the current list
        remaining_kwargs = {k: v for k, v in kwargs.items()}
        return _todo_executor(action="list", **remaining_kwargs)

    # --- summary ---
    if action == "summary":
        summary = store.summary()
        lines = [
            "Todo summary:",
            f"  Total: {summary['total']}",
            f"  Pending: {summary['counts']['pending']}",
            f"  In progress: {summary['counts']['in_progress']}",
            f"  Completed: {summary['counts']['completed']}",
            f"  Cancelled: {summary['counts']['cancelled']}",
        ]
        return "\n".join(lines)

    return f"Unknown action: {action}. Use: list, add, update, delete, reorder, clear, set, summary"


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

TODO_SYSTEM_INSTRUCTION = """
TASK MANAGEMENT:
You have a todo list tool. Use it to track your work:

1. At the start of a task, create a todo list with the steps you plan to take.
2. Mark items as "in_progress" when you start working on them.
3. Mark items as "completed" when done. Cancel items that are no longer needed.
4. Only ONE item should be "in_progress" at any time.
5. Reorder items if priorities change.
6. Check the todo list regularly to stay on track.

The todo list is shared — changes are visible to the orchestrator and other
agents. Keep it accurate and up to date.
"""
