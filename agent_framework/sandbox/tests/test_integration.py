"""
Cross-component integration tests for the sandbox.

Verifies end-to-end workflows spanning multiple sandbox components:
- Client ↔ Server (real TCP, round-trip, timeout, disconnect)
- Injected sandbox factory / causality checker (generalized server)
- TaskQueue (task lifecycle)
- Experiments (experiment tracking)
"""

import asyncio
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from agent_framework.sandbox.client import SandboxClient
from agent_framework.sandbox.server import SandboxServer, SharedNamespace
from agent_framework.sandbox.transport import ConnectionClosed, Message
from agent_framework.sandbox.experiments import ExperimentTracker
from agent_framework.sandbox.task_queue import TaskQueue
from agent_framework.sandbox.tests.fixtures import make_synthetic_ohlcv


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _find_free_port():
    """Pick a random free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def sandbox_port():
    return _find_free_port()


@pytest.fixture
async def integration_server(sandbox_port):
    """Start a sandbox server on a random port; stop after test."""
    server = SandboxServer("127.0.0.1", sandbox_port)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def tmpdir(tmp_path):
    """Temporary directory path (string)."""
    return str(tmp_path)


# ===================================================================
# 1. Client → Server full round-trip
# ===================================================================

class TestClientServerRoundTrip:
    """Start a real sandbox server, connect with SandboxClient, and verify
    all commands work end-to-end."""

    async def test_execute_code(self, sandbox_port, integration_server):
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            resp = await client.execute("x = 42")
            assert resp.success is True
            assert resp.error == ""
        finally:
            await client.close()

    async def test_execute_print_output(self, sandbox_port, integration_server):
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            resp = await client.execute("print('hello sandbox')")
            assert resp.success is True
            assert "hello sandbox" in resp.output
        finally:
            await client.close()

    async def test_eval_expression(self, sandbox_port, integration_server):
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            resp = await client.eval("2 + 2")
            assert resp.success is True
            assert "4" in resp.output
        finally:
            await client.close()

    async def test_eval_numpy_expression(self, sandbox_port, integration_server):
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            resp = await client.eval("np.array([1,2,3]).sum()")
            assert resp.success is True
            assert "6" in resp.output
        finally:
            await client.close()

    async def test_list_vars_after_execute(self, sandbox_port, integration_server):
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            await client.execute("my_var = [1, 2, 3]")
            resp = await client.list_vars()
            assert resp.success is True
            assert "my_var" in resp.variables
        finally:
            await client.close()

    async def test_get_var_metadata(self, sandbox_port, integration_server):
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            await client.execute("my_arr = np.zeros((3, 4))")
            resp = await client.get_var("my_arr")
            assert resp.success is True
            assert resp.shape == [3, 4]
            assert resp.var_type is not None
        finally:
            await client.close()

    async def test_full_workflow(self, sandbox_port, integration_server):
        """Execute, eval, list_vars, get_var in sequence."""
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            # 1. Execute code to define variables
            resp = await client.execute("a = np.arange(10); b = a ** 2")
            assert resp.success

            # 2. Evaluate an expression using those variables
            resp = await client.eval("a.sum()")
            assert resp.success
            assert "45" in resp.output

            # 3. List variables — should include user-defined ones
            resp = await client.list_vars()
            assert resp.success
            assert "a" in resp.variables
            assert "b" in resp.variables

            # 4. Get metadata for variable 'b'
            resp = await client.get_var("b")
            assert resp.success
            assert resp.shape == [10]
        finally:
            await client.close()

    async def test_context_manager(self, sandbox_port, integration_server):
        """Verify the async context manager pattern works."""
        async with SandboxClient("127.0.0.1", sandbox_port) as client:
            resp = await client.execute("ctx_test = True")
            assert resp.success
            resp = await client.list_vars()
            assert "ctx_test" in resp.variables
        assert client.is_connected is False

    async def test_not_connected_error(self):
        """client.execute() before connect() raises RuntimeError."""
        client = SandboxClient("127.0.0.1", 99999)
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.execute("pass")

    async def test_server_execution_error_propagated(self, sandbox_port, integration_server):
        """Execution errors on the server are returned with success=False."""
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        try:
            resp = await client.execute("1/0")
            assert resp.success is False
            assert "ZeroDivisionError" in resp.error
        finally:
            await client.close()


# ===================================================================
# 2. Client → Server with timeout
# ===================================================================

class TestClientServerTimeout:
    """Verify client timeout enforcement.

    Note: With a real TCP server and Python 3.12+, asyncio.wait_for()
    cannot cancel StreamReader.readexactly(), so timeout tests use
    mock-based approaches (same pattern as test_client.py).
    """

    async def test_timeout_enforcement_raises_on_hung_recv(self):
        """Client raises asyncio.TimeoutError when recv blocks too long."""
        client = SandboxClient(timeout=0.1)
        client._transport = AsyncMock()
        async def slow_recv():
            await asyncio.sleep(60)
        client._transport.recv = slow_recv

        with patch.object(client._transport, 'send', new=AsyncMock()):
            with pytest.raises(asyncio.TimeoutError):
                await client.execute("x = 1")

    async def test_per_call_timeout_override(self):
        """Per-call timeout overrides the client default."""
        client = SandboxClient(timeout=30.0)
        client._transport = AsyncMock()
        async def slow_recv():
            await asyncio.sleep(60)
        client._transport.recv = slow_recv

        with patch.object(client._transport, 'send', new=AsyncMock()):
            # Per-call timeout=0.05 should trigger
            with pytest.raises(asyncio.TimeoutError):
                await client.execute("x = 1", timeout=0.05)

    async def test_negative_timeout_raises_value_error(self):
        """timeout <= 0 raises ValueError immediately."""
        client = SandboxClient(timeout=0)
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv = AsyncMock()

        with pytest.raises(ValueError, match="timeout must be > 0 or None"):
            await client.execute("pass", timeout=0)

    async def test_no_timeout_allows_slow_operation(self):
        """When timeout is None, no wait_for() wraps the call."""
        client = SandboxClient(timeout=None)
        client._transport = AsyncMock()
        async def mock_recv():
            return Message("result", success=True, output="ok")
        client._transport.recv = mock_recv

        resp = await client.execute("pass", timeout=None)
        assert resp.success

    async def test_server_executes_quickly_within_timeout(self, sandbox_port, integration_server):
        """Fast server operations complete well within the client timeout."""
        client = SandboxClient("127.0.0.1", sandbox_port, timeout=5.0)
        await client.connect()
        try:
            resp = await client.execute("quick_result = np.pi")
            assert resp.success
        finally:
            await client.close()


# ===================================================================
# 3. Client → Server error handling (ConnectionClosed)
# ===================================================================

class TestClientServerErrorHandling:
    """Verify ConnectionClosed handling on real and mock server."""

    async def test_connection_closed_propagated(self):
        """When transport raises ConnectionClosed, client propagates it."""
        client = SandboxClient()
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv.side_effect = ConnectionClosed("server gone")

        with pytest.raises(ConnectionClosed, match="server gone"):
            await client.execute("x = 1")

    async def test_connection_reset_normalized(self):
        """ConnectionResetError from transport is normalized to ConnectionClosed."""
        client = SandboxClient()
        client._transport = AsyncMock()
        client._transport.send = AsyncMock()
        client._transport.recv.side_effect = ConnectionResetError("broken pipe")

        with pytest.raises(ConnectionClosed):
            await client.execute("x = 1")

    async def test_connection_refused_on_bad_port(self):
        """Connecting to a port with no server raises OSError."""
        client = SandboxClient("127.0.0.1", 59876)
        with pytest.raises((OSError, ConnectionRefusedError)):
            await client.connect()

    async def test_double_close_is_safe(self, sandbox_port, integration_server):
        """Calling close() twice should not raise."""
        client = SandboxClient("127.0.0.1", sandbox_port)
        await client.connect()
        await client.close()
        await client.close()  # should be safe

    async def test_is_connected_state(self, sandbox_port, integration_server):
        """is_connected property tracks connection state."""
        client = SandboxClient("127.0.0.1", sandbox_port)
        assert client.is_connected is False
        await client.connect()
        assert client.is_connected is True
        await client.close()
        assert client.is_connected is False


# ===================================================================
# 4. Generalized server — injected factory / checker / extra globals
# ===================================================================

class FakeSandbox:
    """Minimal in-memory sandbox stand-in (no data loading)."""

    calls = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        self.ohlcv = pd.DataFrame({"ticker": [], "time": []})


class TestInjectedNamespace:
    """The generalized server accepts injected domain components."""

    def test_default_namespace_has_none_panel(self):
        ns = SharedNamespace()
        assert ns.globals["s"] is None
        assert ns.globals["df"] is None
        assert "check_causality" not in ns.globals
        assert "CausalityViolation" not in ns.globals

    def test_factory_injects_s_and_df(self):
        FakeSandbox.calls.clear()
        panel = pd.DataFrame({"ticker": ["A"], "time": [1]})

        class _Factory:
            def __init__(self, **kwargs):
                self.ohlcv = panel

        ns = SharedNamespace(
            tickers=["A"], start="2020-01-01", end="2021-01-01",
            sandbox_factory=_Factory,
        )
        assert ns.globals["s"].ohlcv is panel
        assert ns.globals["df"] is panel
        names = ns.list_variables()
        assert "s" not in names
        assert "df" not in names

    def test_extra_globals_merged_and_hidden_from_list_vars(self):
        ns = SharedNamespace(extra_globals={"my_helper": len})
        assert ns.globals["my_helper"] is len
        assert "my_helper" not in ns.list_variables()

    def test_checker_injection_registers_names(self):
        class _Violation(Exception):
            pass

        def _checker(func, data, **kwargs):
            return None

        ns = SharedNamespace(causality_checker=_checker,
                             causality_exception=_Violation)
        assert ns.globals["check_causality"] is _checker
        assert ns.globals["CausalityViolation"] is _Violation
        assert "check_causality" not in ns.list_variables()
        assert "CausalityViolation" not in ns.list_variables()

    async def test_check_causality_command_uses_injected_checker(self):
        """With an injected checker, the check_causality command round-trips."""

        class _Violation(Exception):
            pass

        calls = []

        def _checker(func, data, **kwargs):
            calls.append((func.__name__, data))

        panel = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})

        def clean_feat(x):
            return x  # identity — no look-ahead

        ns = SharedNamespace(
            sandbox_factory=lambda **kw: type("S", (), {"ohlcv": panel})(),
            causality_checker=_checker,
            causality_exception=_Violation,
        )
        ns.globals["clean_feat"] = clean_feat

        msg = Message("check_causality", func_name="clean_feat", data_name="df")
        from agent_framework.sandbox.server import _handle_check_causality
        resp = await _handle_check_causality(ns, msg)
        assert resp.payload["success"] is True
        assert calls == [("clean_feat", panel)]

    async def test_check_causality_handler_without_checker(self):
        """The handler reports a clear error when no checker is configured."""
        ns = SharedNamespace()
        from agent_framework.sandbox.server import _handle_check_causality
        msg = Message("check_causality", func_name="f")
        resp = await _handle_check_causality(ns, msg)
        assert resp.payload["success"] is False
        assert "no causality checker configured" in resp.payload["error"]

    async def test_check_causality_unregistered_without_checker(self, sandbox_port):
        """Without a checker, the check_causality command is unknown to the server."""
        server = SandboxServer("127.0.0.1", sandbox_port)
        await server.start()
        try:
            client = SandboxClient("127.0.0.1", sandbox_port)
            await client.connect()
            try:
                resp = await client.check_causality("some_func")
                assert resp.success is False
                assert "unknown command" in (resp.error or "").lower()
            finally:
                await client.close()
        finally:
            await server.stop()


# ===================================================================
# 5. TaskQueue (task lifecycle)
# ===================================================================

class TestTaskQueueFeatures:
    """Add task, claim it, complete."""

    def test_task_queue_lifecycle(self, tmpdir):
        """Full lifecycle: add → claim → complete."""
        queue_path = os.path.join(tmpdir, "queue.md")
        tq = TaskQueue(queue_path)

        # Add a task
        task_id = tq.add("Compute rolling momentum features", asset="stocks")
        assert task_id is not None
        assert task_id.startswith("S")

        # List pending
        pending = tq.list_tasks(status="pending")
        assert len(pending) >= 1
        assert any(t["id"] == task_id for t in pending)

        # Claim the task
        claimed = tq.claim(task_id, agent_id="test_agent")
        assert claimed is not None
        assert claimed["status"] == "testing"
        assert claimed["agent"] == "test_agent"

        # Complete the task
        completed = tq.complete(task_id, result_path="/path/to/result.parquet")
        assert completed is not None
        assert completed["status"] == "validated"
        assert completed["result"] == "/path/to/result.parquet"

        # Verify no more pending tasks with this ID
        pending = tq.list_tasks(status="pending")
        assert not any(t["id"] == task_id for t in pending)

    def test_task_queue_reject(self, tmpdir):
        """Task can be rejected from testing state."""
        queue_path = os.path.join(tmpdir, "queue3.md")
        tq = TaskQueue(queue_path)

        task_id = tq.add("Bad idea feature", asset="stocks")
        claimed = tq.claim(task_id, "reviewer")
        assert claimed["status"] == "testing"

        rejected = tq.reject(task_id, reason_path="Not predictive enough")
        assert rejected is not None
        assert rejected["status"] == "rejected"

    def test_task_queue_invalid_transition(self, tmpdir):
        """Cannot complete a pending task (must claim first)."""
        queue_path = os.path.join(tmpdir, "queue4.md")
        tq = TaskQueue(queue_path)

        task_id = tq.add("Direct completion test")
        # Try to complete without claiming
        result = tq.complete(task_id, result_path="x")
        assert result is None

    def test_task_queue_claim_already_claimed(self, tmpdir):
        """Cannot claim a task that is already in testing state."""
        queue_path = os.path.join(tmpdir, "queue5.md")
        tq = TaskQueue(queue_path)

        task_id = tq.add("Already claimed test")
        tq.claim(task_id, "agent_a")
        # Second claim should return None
        result = tq.claim(task_id, "agent_b")
        assert result is None

    def test_task_queue_multiple_tasks(self, tmpdir):
        """Multiple tasks in queue with sequential operations."""
        queue_path = os.path.join(tmpdir, "queue6.md")
        tq = TaskQueue(queue_path)

        ids = []
        for i in range(5):
            tid = tq.add(f"Feature idea {i}")
            ids.append(tid)

        # All should be pending
        pending = tq.list_tasks(status="pending")
        assert len(pending) >= 5

        # Claim and complete first two
        tq.claim(ids[0], "agent_0")
        tq.claim(ids[1], "agent_1")
        tq.complete(ids[0], "result_0")
        tq.complete(ids[1], "result_1")

        # Check statuses
        validated = tq.list_tasks(status="validated")
        assert len(validated) == 2
        remaining_pending = tq.list_tasks(status="pending")
        assert len(remaining_pending) >= 3

    def test_task_queue_persists_across_instantiation(self, tmpdir):
        """Tasks persist when creating a new TaskQueue instance."""
        queue_path = os.path.join(tmpdir, "queue7.md")
        tq1 = TaskQueue(queue_path)
        task_id = tq1.add("Persistent task")

        tq2 = TaskQueue(queue_path)  # new instance
        tasks = tq2.list_tasks()
        assert any(t["id"] == task_id for t in tasks)

    def test_task_queue_markdown_round_trip(self, tmpdir):
        """Markdown table round-trip preserves data."""
        queue_path = os.path.join(tmpdir, "queue8.md")
        tq = TaskQueue(queue_path)

        task_id = tq.add("Feature with | pipe character", asset="stocks")
        claimed = tq.claim(task_id, "agent")
        assert claimed["idea"] == "Feature with | pipe character"

    def test_task_queue_jsonl_sidecar(self, tmpdir):
        """JSONL sidecar is created and populated."""
        queue_path = os.path.join(tmpdir, "queue9.md")
        tq = TaskQueue(queue_path)

        task_id = tq.add("JSONL test")
        jsonl_path = queue_path.rsplit(".", 1)[0] + ".jsonl"

        assert os.path.exists(jsonl_path)
        with open(jsonl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["id"] == task_id
        assert entry["action"] == "add"


# ===================================================================
# 6. Full cross-component integration (server + experiments + task queue)
# ===================================================================

class TestFullIntegration:
    """End-to-end workflow combining server, experiments, and task queue."""

    async def test_research_workflow(self, sandbox_port, integration_server):
        """Simulate a research workflow:
        1. Connect to sandbox server
        2. Execute computation code
        3. Track results in the experiment tracker
        4. Track progress with the task queue
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            async with SandboxClient("127.0.0.1", sandbox_port) as client:
                # 1. Load and explore data
                resp = await client.execute(
                    "rng = np.random.default_rng(42)\n"
                    "prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 500))\n"
                    "returns = np.diff(prices) / prices[:-1]"
                )
                assert resp.success

                # 2. Compute feature
                resp = await client.execute(
                    "volatility = pd.Series(returns).rolling(20).std().values"
                )
                assert resp.success

                # 3. Evaluate expression
                resp = await client.eval("prices[-1]")
                assert resp.success

                # 4. List variables
                resp = await client.list_vars()
                assert resp.success
                assert "prices" in resp.variables
                assert "returns" in resp.variables
                assert "volatility" in resp.variables

                # 5. Log to experiment tracker
                db_path = os.path.join(tmpdir, "workflow.db")
                tracker = ExperimentTracker(db_path)
                resp = await client.eval("float(prices[-1])")
                final_price = float(resp.output)
                run_id = tracker.log_experiment(
                    experiment_type="train",
                    config_id="research_workflow",
                    params={"seed": 42},
                    results={"final_price": final_price},
                )

                # 6. Task queue
                queue_path = os.path.join(tmpdir, "workflow.md")
                tq = TaskQueue(queue_path)
                tid = tq.add("Research workflow test")
                tq.claim(tid, "workflow_agent")
                tq.complete(tid, "completed")

                # Verify experiment retrieval
                exp = tracker.get_experiment(run_id)
                assert exp is not None
                assert exp["config_id"] == "research_workflow"
                assert abs(exp["results"]["final_price"] - final_price) < 1e-9

    def test_panel_experiment_queue_pipeline(self, tmpdir):
        """Offline pipeline: synthetic panel → stats → experiment → task queue."""
        # Generate data
        df = make_synthetic_ohlcv(tickers=10, days=300, seed=123)

        # Compute feature + label
        df["feat"] = df.groupby("ticker")["raw_close"].transform(
            lambda s: s.pct_change().rolling(20).std()
        )
        df["label"] = df.groupby("ticker")["raw_close"].transform(
            lambda s: s.shift(-5) / s - 1
        )

        valid = df.dropna(subset=["feat", "label"])
        # Spearman IC (rank correlation) of feature vs label
        feat_rank = valid["feat"].rank().to_numpy()
        label_rank = valid["label"].rank().to_numpy()
        ic = float(np.corrcoef(feat_rank, label_rank)[0, 1])

        # Log to experiment tracker
        db_path = os.path.join(tmpdir, "integrated.db")
        tracker = ExperimentTracker(db_path)
        run_id = tracker.log_experiment(
            experiment_type="train",
            config_id="integrated_test",
            params={"tickers": 10, "days": 300, "seed": 123},
            results={"ic": float(ic), "rows": int(len(valid))},
        )

        # Task queue
        queue_path = os.path.join(tmpdir, "integrated.md")
        tq = TaskQueue(queue_path)
        task_id = tq.add("Integrated pipeline feature")
        tq.claim(task_id, "integration_test")
        tq.complete(task_id, result_path=run_id)

        # Verify all artifacts
        assert os.path.exists(db_path)
        completed = tq.list_tasks(status="validated")
        assert len(completed) == 1
        assert completed[0]["result"] == run_id
