#!/usr/bin/env python3
"""Tests for agent_framework.todo — persistent task list."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from agent_framework.todo import (
    TODO_SYSTEM_INSTRUCTION,
    _todo_executor,
)
from agent_framework.todo_backends import (
    FileTodoStore,
    MemoryTodoStore,
    TodoItem,
    TodoStoreBackend,
    load_todo_store,
)


# ---------------------------------------------------------------------------
# TodoItem
# ---------------------------------------------------------------------------


class TestTodoItem:
    """Test TodoItem dataclass."""

    def test_to_dict(self):
        item = TodoItem(id="abc123", content="Test task", status="pending")
        d = item.to_dict()
        assert d == {"id": "abc123", "content": "Test task", "status": "pending"}

    def test_from_dict(self):
        d = {"id": "abc123", "content": "Test task", "status": "pending"}
        item = TodoItem.from_dict(d)
        assert item.id == "abc123"
        assert item.content == "Test task"
        assert item.status == "pending"

    def test_roundtrip(self):
        original = TodoItem(id="abc123", content="Test task", status="pending")
        restored = TodoItem.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.status == original.status


# ---------------------------------------------------------------------------
# TodoStore
# ---------------------------------------------------------------------------


class TestTodoStore:
    """Test FileTodoStore file-backed persistence."""

    def setup_method(self):
        """Create a temporary directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, ".todo.json")

    def test_create_store(self):
        """Test that creating a store creates the file."""
        store = FileTodoStore(self.store_path)
        assert os.path.exists(self.store_path)
        with open(self.store_path, "r") as f:
            data = json.load(f)
        assert "items" in data
        assert "created_at" in data

    def test_add_item(self):
        """Test adding a todo item."""
        store = FileTodoStore(self.store_path)
        item = store.add_item("First task")
        assert item.content == "First task"
        assert item.status == "pending"
        assert item.id  # Has a generated ID

    def test_list_items(self):
        """Test listing all todo items."""
        store = FileTodoStore(self.store_path)
        store.add_item("Task 1")
        store.add_item("Task 2")
        items = store.list_items()
        assert len(items) == 2
        assert items[0].content == "Task 1"
        assert items[1].content == "Task 2"

    def test_update_item_status(self):
        """Test updating item status."""
        store = FileTodoStore(self.store_path)
        item = store.add_item("Task to complete")
        updated = store.update_item(item.id, status="in_progress")
        assert updated.status == "in_progress"

    def test_update_item_content(self):
        """Test updating item content."""
        store = FileTodoStore(self.store_path)
        item = store.add_item("Old content")
        updated = store.update_item(item.id, content="New content")
        assert updated.content == "New content"

    def test_update_item_both(self):
        """Test updating both content and status."""
        store = FileTodoStore(self.store_path)
        item = store.add_item("Old content")
        updated = store.update_item(item.id, content="New content", status="completed")
        assert updated.content == "New content"
        assert updated.status == "completed"

    def test_update_item_not_found(self):
        """Test updating non-existent item raises ValueError."""
        store = FileTodoStore(self.store_path)
        with pytest.raises(ValueError, match="not found"):
            store.update_item("nonexistent", status="pending")

    def test_delete_item(self):
        """Test deleting a todo item."""
        store = FileTodoStore(self.store_path)
        item = store.add_item("Task to delete")
        store.delete_item(item.id)
        items = store.list_items()
        assert len(items) == 0

    def test_delete_item_not_found(self):
        """Test deleting non-existent item raises ValueError."""
        store = FileTodoStore(self.store_path)
        with pytest.raises(ValueError, match="not found"):
            store.delete_item("nonexistent")

    def test_reorder(self):
        """Test reordering todo items."""
        store = FileTodoStore(self.store_path)
        item1 = store.add_item("Task 1")
        item2 = store.add_item("Task 2")
        item3 = store.add_item("Task 3")

        # Reorder: 3, 1, 2
        items = store.reorder([item3.id, item1.id, item2.id])
        assert items[0].content == "Task 3"
        assert items[1].content == "Task 1"
        assert items[2].content == "Task 2"

    def test_clear_completed(self):
        """Test clearing completed items."""
        store = FileTodoStore(self.store_path)
        store.add_item("Task 1")
        item2 = store.add_item("Task 2")
        store.update_item(item2.id, status="completed")
        store.add_item("Task 3")
        item4 = store.add_item("Task 4")
        store.update_item(item4.id, status="completed")

        removed = store.clear_completed()
        assert removed == 2
        items = store.list_items()
        assert len(items) == 2
        assert items[0].content == "Task 1"
        assert items[1].content == "Task 3"

    def test_summary(self):
        """Test summary statistics."""
        store = FileTodoStore(self.store_path)
        item1 = store.add_item("Pending")
        item2 = store.add_item("In progress")
        item3 = store.add_item("Completed")
        store.update_item(item2.id, status="in_progress")
        store.update_item(item3.id, status="completed")

        summary = store.summary()
        assert summary["total"] == 3
        assert summary["counts"]["pending"] == 1
        assert summary["counts"]["in_progress"] == 1
        assert summary["counts"]["completed"] == 1

    def test_persistence(self):
        """Test that data persists across store instances."""
        store1 = FileTodoStore(self.store_path)
        store1.add_item("Persistent task")
        store1.add_item("Another task")

        # Create a new store instance
        store2 = FileTodoStore(self.store_path)
        items = store2.list_items()
        assert len(items) == 2
        assert items[0].content == "Persistent task"


# ---------------------------------------------------------------------------
# MemoryTodoStore
# ---------------------------------------------------------------------------


class TestMemoryTodoStore:
    """Test MemoryTodoStore in-memory implementation."""

    def test_add_and_list(self):
        """Test adding and listing items."""
        store = MemoryTodoStore()
        store.add_item("Task 1")
        store.add_item("Task 2")
        items = store.list_items()
        assert len(items) == 2

    def test_no_persistence(self):
        """Test that data is not persisted (in-memory only)."""
        store = MemoryTodoStore()
        store.add_item("Task 1")
        # Create new store - should be empty
        store2 = MemoryTodoStore()
        assert len(store2.list_items()) == 0

    def test_all_operations(self):
        """Test all CRUD operations."""
        store = MemoryTodoStore()
        item = store.add_item("Task 1")
        store.update_item(item.id, status="in_progress")
        items = store.list_items()
        assert items[0].status == "in_progress"
        store.delete_item(item.id)
        assert len(store.list_items()) == 0


# ---------------------------------------------------------------------------
# load_todo_store
# ---------------------------------------------------------------------------


class TestLoadTodoStore:
    """Test config-driven todo store loading."""

    def test_default_file_store(self):
        """Test loading default file store."""
        config = {}
        store = load_todo_store(config, workspace="/tmp/test_todo")
        assert isinstance(store, FileTodoStore)

    def test_custom_file_store(self):
        """Test loading with custom path."""
        config = {
            "todo_store_config": {
                "path": "/tmp/custom_todo.json",
            }
        }
        store = load_todo_store(config)
        assert store.path == "/tmp/custom_todo.json"

    def test_invalid_spec(self):
        """Test invalid store spec raises error."""
        config = {"todo_store": "invalid_spec"}
        with pytest.raises(ValueError, match="Invalid todo_store spec"):
            load_todo_store(config)

    def test_memory_store(self):
        """Test loading memory store."""
        config = {
            "todo_store": "agent_framework.todo_backends:MemoryTodoStore",
        }
        store = load_todo_store(config)
        assert isinstance(store, MemoryTodoStore)

    def test_custom_backend(self):
        """Test loading custom backend."""
        # Create a simple custom backend
        import importlib.util
        import sys
        from types import ModuleType

        # Create a module with a custom store
        class CustomTodoStore:
            def __init__(self, **kwargs):
                pass
            def list_items(self):
                return []
            def add_item(self, content):
                return TodoItem(id="custom", content=content, status="pending")
            def update_item(self, item_id, content=None, status=None):
                raise NotImplementedError
            def delete_item(self, item_id):
                pass
            def reorder(self, item_ids):
                return []
            def clear_completed(self):
                return 0
            def summary(self):
                return {"total": 0, "counts": {}}

        # Register in a temporary module
        module = ModuleType("custom_todo")
        module.CustomTodoStore = CustomTodoStore
        sys.modules["custom_todo"] = module

        try:
            config = {
                "todo_store": "custom_todo:CustomTodoStore",
            }
            store = load_todo_store(config)
            assert isinstance(store, CustomTodoStore)
        finally:
            del sys.modules["custom_todo"]


# ---------------------------------------------------------------------------
# _todo_executor
# ---------------------------------------------------------------------------


class TestTodoExecutor:
    """Test the todo executor function."""

    def setup_method(self):
        """Set up temporary workspace."""
        self.tmpdir = tempfile.mkdtemp()
        self.todo_path = os.path.join(self.tmpdir, ".todo.json")

    def test_list_empty(self):
        """Test listing empty todo list."""
        result = _todo_executor(action="list", todo_path=self.todo_path)
        assert "empty" in result.lower()

    def test_add(self):
        """Test adding a todo item."""
        result = _todo_executor(action="add", content="New task", todo_path=self.todo_path)
        assert "Added" in result
        assert "New task" in result

    def test_add_missing_content(self):
        """Test adding without content fails."""
        result = _todo_executor(action="add", todo_path=self.todo_path)
        assert "Error" in result

    def test_update_status(self):
        """Test updating item status."""
        _todo_executor(action="add", content="Task to update", todo_path=self.todo_path)
        result = _todo_executor(action="list", todo_path=self.todo_path)
        # Extract ID from the list output
        import re
        match = re.search(r"id=([a-f0-9]+)", result)
        assert match
        item_id = match.group(1)

        result = _todo_executor(
            action="update", item_id=item_id, status="in_progress", todo_path=self.todo_path
        )
        assert "Updated" in result

    def test_delete(self):
        """Test deleting an item."""
        _todo_executor(action="add", content="Task to delete", todo_path=self.todo_path)
        result = _todo_executor(action="list", todo_path=self.todo_path)
        import re
        match = re.search(r"id=([a-f0-9]+)", result)
        assert match
        item_id = match.group(1)

        result = _todo_executor(action="delete", item_id=item_id, todo_path=self.todo_path)
        assert "Deleted" in result

    def test_set_action(self):
        """Test setting the full todo list."""
        todos = [
            {"content": "Task 1", "status": "pending"},
            {"content": "Task 2", "status": "pending"},
        ]
        result = _todo_executor(action="set", todos=todos, todo_path=self.todo_path)
        assert "Task 1" in result
        assert "Task 2" in result

    def test_summary(self):
        """Test summary action."""
        _todo_executor(action="add", content="Pending task", todo_path=self.todo_path)
        result = _todo_executor(action="summary", todo_path=self.todo_path)
        assert "Todo summary" in result
        assert "Pending: 1" in result

    def test_unknown_action(self):
        """Test unknown action returns error."""
        result = _todo_executor(action="unknown", todo_path=self.todo_path)
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestTodoSystemInstruction:
    """Test the todo system instruction."""

    def test_instruction_exists(self):
        """Test that the instruction is non-empty."""
        assert TODO_SYSTEM_INSTRUCTION
        assert len(TODO_SYSTEM_INSTRUCTION) > 50

    def test_mentions_todo(self):
        """Test that the instruction mentions todo."""
        assert "todo" in TODO_SYSTEM_INSTRUCTION.lower()

    def test_mentions_status(self):
        """Test that the instruction mentions status."""
        assert "in_progress" in TODO_SYSTEM_INSTRUCTION or "completed" in TODO_SYSTEM_INSTRUCTION

    def test_mentions_shared(self):
        """Test that the instruction mentions shared state."""
        assert "shared" in TODO_SYSTEM_INSTRUCTION.lower()
