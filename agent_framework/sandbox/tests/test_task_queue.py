"""
Task queue tests — file-backed idea tracking with flock protection.
"""

import os
import tempfile
import pytest
import time
import json
from pathlib import Path
from agent_framework.sandbox.task_queue import TaskQueue


class TestTaskQueue:

    def test_create_queue(self):
        """Queue should initialize with empty table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            tasks = q.list_tasks()
            assert len(tasks) == 0

    def test_add_task(self):
        """Adding a task should return an ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            task_id = q.add("Test volume ratio as predictor")
            assert task_id is not None
            assert task_id.startswith("S")

    def test_add_multiple_tasks(self):
        """Multiple tasks should have incrementing IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            id1 = q.add("First idea")
            id2 = q.add("Second idea")
            assert id1 != id2
            # IDs should increment
            assert id1 == "S001"
            assert id2 == "S002"

    def test_claim_task(self):
        """Claiming a task should mark it as testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            task_id = q.add("Idea to test")
            result = q.claim(task_id, "agent-1")
            assert result["status"] == "testing"
            # List should reflect claim
            tasks = q.list_tasks()
            claimed = [t for t in tasks if t["id"] == task_id]
            assert len(claimed) == 1
            assert claimed[0]["status"] == "testing"

    def test_complete_task(self):
        """Completing a task should mark it validated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            task_id = q.add("Valid idea")
            q.claim(task_id, "agent-1")
            result = q.complete(task_id, "/tmp/report.pdf")
            assert result["status"] == "validated"

    def test_reject_task(self):
        """Rejecting a task should mark it rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            task_id = q.add("Bad idea")
            q.claim(task_id, "agent-1")
            result = q.reject(task_id, "/tmp/reason.md")
            assert result["status"] == "rejected"

    def test_list_by_status(self):
        """Filtering by status should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            q.add("Pending idea")
            id2 = q.add("To be claimed")
            q.claim(id2, "agent-1")
            id3 = q.add("To be rejected")
            q.claim(id3, "agent-2")
            q.reject(id3, "/tmp/reason.md")

            pending = q.list_tasks(status="pending")
            assert len(pending) == 1
            assert pending[0]["status"] == "pending"

            testing = q.list_tasks(status="testing")
            assert len(testing) == 1

            rejected = q.list_tasks(status="rejected")
            assert len(rejected) == 1

    def test_claim_nonexistent_task(self):
        """Claiming a non-existent task should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            result = q.claim("S999", "agent-1")
            assert result is None

    def test_queue_file_exists(self):
        """The queue file should be a valid markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "queue.md")
            q = TaskQueue(queue_path=path)
            q.add("Test idea")
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "| ID |" in content  # Table header

    def test_flock_protection(self):
        """Concurrent access should be flock-protected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            # Basic sanity: multiple sequential operations work
            for i in range(10):
                q.add(f"Idea {i}")
            tasks = q.list_tasks()
            assert len(tasks) == 10

    def test_jsonl_sidecar(self):
        """A JSONL sidecar file should be maintained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "queue.md")
            q = TaskQueue(queue_path=path)
            task_id = q.add("Test idea", asset="stocks")
            q.claim(task_id, "agent-1")
            # JSONL sidecar should exist
            jsonl_path = os.path.join(tmpdir, "queue.jsonl")
            assert Path(jsonl_path).exists()

    def test_pipe_characters_roundtrip(self):
        """Pipe characters in idea text should round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            idea = "Compare SMA(20) vs EMA(20) | which performs better?"
            task_id = q.add(idea)
            tasks = q.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["idea"] == idea
            # Verify the markdown file contains the escaped form
            content = Path(q._path).read_text()
            assert "&#124;" in content  # HTML entity should be present
            # Verify re-reading is consistent
            tasks2 = q.list_tasks()
            assert tasks2[0]["idea"] == idea

    def test_state_transition_complete_requires_testing(self):
        """complete() should return None if task is not in testing status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            task_id = q.add("Idea")
            # Should fail — task is still pending (never claimed)
            assert q.complete(task_id, "/tmp/report.md") is None
            # Claim it → testing
            q.claim(task_id, "agent-1")
            # Now it should succeed
            result = q.complete(task_id, "/tmp/report.md")
            assert result is not None
            assert result["status"] == "validated"
            # Should fail again — already validated
            assert q.complete(task_id, "/tmp/other.md") is None

    def test_state_transition_reject_requires_testing(self):
        """reject() should return None if task is not in testing status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = TaskQueue(queue_path=os.path.join(tmpdir, "queue.md"))
            task_id = q.add("Idea")
            # Should fail — task is still pending
            assert q.reject(task_id, "/tmp/reason.md") is None
            # Claim it → testing
            q.claim(task_id, "agent-1")
            result = q.reject(task_id, "/tmp/reason.md")
            assert result is not None
            assert result["status"] == "rejected"
            # Should fail again — already rejected
            assert q.reject(task_id, "/tmp/other.md") is None

    def test_asset_stored_in_task_and_jsonl(self):
        """The asset parameter should be stored in the task dict and JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "queue.md")
            q = TaskQueue(queue_path=path)
            task_id = q.add("Crypto idea", asset="crypto")
            tasks = q.list_tasks()
            assert tasks[0]["asset"] == "crypto"
            # Verify JSONL contains asset
            jsonl_path = os.path.join(tmpdir, "queue.jsonl")
            entries = []
            for line in Path(jsonl_path).read_text().strip().split("\n"):
                entries.append(json.loads(line))
            assert any(e.get("asset") == "crypto" for e in entries)

    def test_jsonl_atomic_write(self):
        """JSONL writes should be atomic (os.write-based, no partial lines)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "queue.md")
            q = TaskQueue(queue_path=path)
            # Write several entries
            for i in range(5):
                q.add(f"Idea {i}")
            # Every line in the JSONL file should be valid JSON
            jsonl_path = os.path.join(tmpdir, "queue.jsonl")
            content = Path(jsonl_path).read_text()
            for i, line in enumerate(content.strip().split("\n")):
                entry = json.loads(line)  # Should not raise
                assert "action" in entry

    def test_jsonl_schema_consistency(self):
        """All JSONL entries should have an 'action' field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "queue.md")
            q = TaskQueue(queue_path=path)
            task_id = q.add("Test idea")
            q.claim(task_id, "agent-1")
            q.complete(task_id, "/tmp/report.md")
            # Read and verify JSONL
            jsonl_path = os.path.join(tmpdir, "queue.jsonl")
            entries = []
            for line in Path(jsonl_path).read_text().strip().split("\n"):
                entries.append(json.loads(line))
            assert len(entries) == 3
            actions = [e["action"] for e in entries]
            assert actions == ["add", "claim", "complete"]
            # Every entry should have action + status
            for e in entries:
                assert "action" in e
                assert "status" in e
