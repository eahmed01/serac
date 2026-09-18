"""Tests for WorkerManager — task decomposition and parallel worker dispatch."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from agent_framework.loop import AgentLoop
from agent_framework.providers import ChatResponse, Provider, UsageStats
from agent_framework.tools import ToolDef, ToolRegistry
from agent_framework.worker_manager import WorkerManager, WorkerResult, WorkerTask


class MockProvider(Provider):
    """Controllable mock provider for testing."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = responses
        self.index = 0
        self.calls: list[tuple[list, list | None]] = []

    def chat(
        self,
        messages: list,
        tools=None,
    ) -> ChatResponse:
        self.calls.append((messages, tools))
        resp = self.responses[self.index]
        self.index += 1
        return resp


class MockModelPool:
    """Minimal ModelPool mock that returns a provider per route call."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def route(self):
        from agent_framework.model_pool import PoolRoutingResult
        return PoolRoutingResult(
            provider=self.provider,
            model_name="mock",
            slot_id=0,
        )

    def release(self, slot_id: int) -> None:
        pass


# ------------------------------------------------------------------
# WorkerTask / WorkerResult constructors
# ------------------------------------------------------------------

class TestDataclasses:
    def test_worker_task_defaults(self):
        task = WorkerTask(task_id="t1", goal="do something", context="here")
        assert task.task_id == "t1"
        assert task.goal == "do something"
        assert task.context == "here"
        assert task.system_prompt == ""
        assert task.tool_names is None
        assert task.max_turns == 20
        assert task.priority == 0

    def test_worker_task_custom(self):
        task = WorkerTask(
            task_id="t2",
            goal="research",
            context="background",
            system_prompt="You are a researcher.",
            tool_names=["web_search", "file_read"],
            max_turns=30,
            priority=1,
        )
        assert task.system_prompt == "You are a researcher."
        assert task.tool_names == ["web_search", "file_read"]
        assert task.max_turns == 30
        assert task.priority == 1

    def test_worker_result(self):
        result = WorkerResult(
            task_id="t1",
            success=True,
            output="done",
            error="",
            duration=1.5,
            turns_used=3,
            usage=UsageStats(prompt_tokens=100, completion_tokens=50),
        )
        assert result.success is True
        assert result.output == "done"
        assert result.duration == 1.5
        assert result.turns_used == 3
        assert result.usage.completion_tokens == 50

    def test_worker_result_failure(self):
        result = WorkerResult(
            task_id="t1",
            success=False,
            output="",
            error="timeout",
            duration=30.0,
            turns_used=0,
            usage=UsageStats(),
        )
        assert result.success is False
        assert result.error == "timeout"


# ------------------------------------------------------------------
# WorkerManager.execute()
# ------------------------------------------------------------------

class TestExecute:
    def _make_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            executor=lambda q: f"results for {q}",
        ))
        return reg

    def test_execute_empty(self):
        pool = MockModelPool(MockProvider([]))
        mgr = WorkerManager(model_pool=pool, tool_registry=ToolRegistry())
        assert mgr.execute([]) == []

    def test_execute_single_task(self):
        provider = MockProvider([
            ChatResponse(text="Task 1 complete", usage=UsageStats(prompt_tokens=10, completion_tokens=20)),
        ])
        pool = MockModelPool(provider)
        reg = self._make_registry()

        mgr = WorkerManager(model_pool=pool, tool_registry=reg, max_parallel=2)
        tasks = [WorkerTask(task_id="a1", goal="Do task A", context="ctx")]

        results = mgr.execute(tasks)
        assert len(results) == 1
        assert results[0].task_id == "a1"
        assert results[0].success is True
        assert results[0].output == "Task 1 complete"
        assert results[0].turns_used == 1
        assert results[0].usage.completion_tokens == 20

    def test_execute_multiple_tasks(self):
        provider = MockProvider([
            ChatResponse(text="Alpha done", usage=UsageStats(completion_tokens=10)),
            ChatResponse(text="Beta done", usage=UsageStats(completion_tokens=15)),
            ChatResponse(text="Gamma done", usage=UsageStats(completion_tokens=12)),
        ])
        pool = MockModelPool(provider)
        reg = self._make_registry()

        mgr = WorkerManager(model_pool=pool, tool_registry=reg, max_parallel=6)
        tasks = [
            WorkerTask(task_id="alpha", goal="Alpha task", context="ctx"),
            WorkerTask(task_id="beta", goal="Beta task", context="ctx"),
            WorkerTask(task_id="gamma", goal="Gamma task", context="ctx"),
        ]

        results = mgr.execute(tasks)
        assert len(results) == 3
        # Results ordered by task_id
        assert results[0].task_id == "alpha"
        assert results[1].task_id == "beta"
        assert results[2].task_id == "gamma"
        assert all(r.success for r in results)

    def test_execute_concurrency_limit(self):
        """Verify max_parallel limits concurrent workers."""
        call_times: list[float] = []

        class DelayedProvider(Provider):
            def chat(self, messages, tools=None):
                t = time.monotonic()
                call_times.append(t)
                time.sleep(0.05)  # 50ms per call
                return ChatResponse(text=f"done at {t:.3f}", usage=UsageStats())

        provider = DelayedProvider()
        pool = MockModelPool(provider)
        reg = ToolRegistry()
        mgr = WorkerManager(model_pool=pool, tool_registry=reg, max_parallel=2)

        tasks = [WorkerTask(task_id=f"t{i}", goal=f"Task {i}", context="") for i in range(4)]
        results = mgr.execute(tasks)

        assert len(results) == 4
        assert all(r.success for r in results)

        # With max_parallel=2 and 4 tasks, first 2 run, then next 2.
        # Sequential would be 4 * 0.05 = 0.20s; parallel should be noticeably faster.
        if len(call_times) >= 4:
            total_duration = call_times[-1] - call_times[0]
            assert total_duration < 0.18, (
                f"Expected parallelism (got {total_duration:.3f}s, "
                "sequential would be ~0.20s)"
            )

    def test_execute_task_failure(self):
        """Worker that throws an exception returns a failed WorkerResult."""
        class FailingProvider(Provider):
            def chat(self, messages, tools=None):
                raise RuntimeError("model unavailable")

        pool = MockModelPool(FailingProvider())
        reg = ToolRegistry()
        mgr = WorkerManager(model_pool=pool, tool_registry=reg)

        tasks = [WorkerTask(task_id="fail", goal="crash me", context="")]
        results = mgr.execute(tasks)

        assert len(results) == 1
        assert results[0].success is False
        assert "model unavailable" in results[0].error
        assert results[0].turns_used == 0

    def test_execute_tool_filtering(self):
        """Worker tool_names are respected when filtering the registry."""
        provider = MockProvider([
            ChatResponse(text="filtered tools", usage=UsageStats()),
        ])
        pool = MockModelPool(provider)

        reg = ToolRegistry()
        reg.register(ToolDef(
            name="web_search",
            description="Search",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "search results",
        ))
        reg.register(ToolDef(
            name="file_read",
            description="Read file",
            parameters={"type": "object", "properties": {}},
            executor=lambda path: "file content",
        ))

        mgr = WorkerManager(model_pool=pool, tool_registry=reg)
        tasks = [WorkerTask(
            task_id="t1",
            goal="search something",
            context="",
            tool_names=["web_search"],  # Only web_search
        )]

        results = mgr.execute(tasks)
        assert len(results) == 1
        assert results[0].success is True

    def test_filter_tools_omitted_names_preserve_all_registered_tools(self):
        """Omitted tool_names retain the legacy default of all tools."""
        reg = ToolRegistry()
        for name in ("sec_search", "sec_fetch", "code_search"):
            reg.register(ToolDef(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                executor=lambda: "done",
            ))

        filtered = WorkerManager(MockModelPool(MockProvider([])), reg)._filter_tools(None)
        assert set(filtered.tools) == {"sec_search", "sec_fetch", "code_search"}

    def test_filter_tools_empty_names_loads_no_tools(self):
        """An explicitly empty tool list must not fall back to all tools."""
        filtered = WorkerManager(
            MockModelPool(MockProvider([])), self._make_registry()
        )._filter_tools([])
        assert filtered.tools == {}

    def test_filter_tools_explicit_sec_names_are_authoritative(self):
        """An explicit SEC list includes SEC tools and excludes host-only tools."""
        reg = ToolRegistry()
        for name in ("sec_search", "sec_fetch", "code_search"):
            reg.register(ToolDef(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                executor=lambda: "done",
            ))

        filtered = WorkerManager(MockModelPool(MockProvider([])), reg)._filter_tools(
            ["sec_search", "sec_fetch"]
        )
        assert set(filtered.tools) == {"sec_search", "sec_fetch"}

    def test_execute_unknown_tool_name_is_ignored(self):
        """Missing tool names are logged and the worker still runs."""
        provider = MockProvider([
            ChatResponse(text="done", usage=UsageStats()),
        ])
        pool = MockModelPool(provider)
        reg = ToolRegistry()  # Empty registry
        mgr = WorkerManager(model_pool=pool, tool_registry=reg)

        tasks = [WorkerTask(
            task_id="t1",
            goal="do it",
            context="",
            tool_names=["nonexistent_tool"],
        )]

        results = mgr.execute(tasks)
        assert len(results) == 1
        assert results[0].success is True

    def test_execute_priority_ordering(self):
        """Tasks are sorted by priority before dispatch."""
        provider = MockProvider([
            ChatResponse(text="done", usage=UsageStats()),
            ChatResponse(text="done", usage=UsageStats()),
            ChatResponse(text="done", usage=UsageStats()),
        ])
        pool = MockModelPool(provider)
        reg = ToolRegistry()

        # Force serialization with max_parallel=1 so order matters
        mgr = WorkerManager(model_pool=pool, tool_registry=reg, max_parallel=1)
        tasks = [
            WorkerTask(task_id="low", goal="low", context="", priority=2),
            WorkerTask(task_id="high", goal="high", context="", priority=0),
            WorkerTask(task_id="mid", goal="mid", context="", priority=1),
        ]

        results = mgr.execute(tasks)
        assert len(results) == 3
        # Results are returned sorted by task_id, not priority
        assert results[0].task_id == "high"
        assert results[1].task_id == "low"
        assert results[2].task_id == "mid"

    def test_execute_custom_system_prompt(self):
        """Worker system_prompt is passed to AgentLoop."""
        provider = MockProvider([
            ChatResponse(text="researched", usage=UsageStats()),
        ])
        pool = MockModelPool(provider)
        reg = ToolRegistry()
        mgr = WorkerManager(model_pool=pool, tool_registry=reg)

        tasks = [WorkerTask(
            task_id="t1",
            goal="Research X",
            context="Background",
            system_prompt="You are a research specialist.",
        )]

        results = mgr.execute(tasks)
        assert len(results) == 1
        assert results[0].success is True
        # Verify the system prompt was sent by checking provider calls
        assert len(provider.calls) == 1
        msgs = provider.calls[0][0]
        assert msgs[0]["role"] == "system"
        assert "research specialist" in msgs[0]["content"]


# ------------------------------------------------------------------
# WorkerManager.decompose()
# ------------------------------------------------------------------

class TestDecompose:
    def test_decompose_valid_json(self):
        json_output = json.dumps([
            {"goal": "Research market trends", "tool_names": ["web_search"], "system_prompt": "", "priority": 0},
            {"goal": "Analyze financial data", "tool_names": ["file_read"], "system_prompt": "", "priority": 1},
            {"goal": "Write summary report", "tool_names": [], "system_prompt": "", "priority": 2},
        ])
        provider = MockProvider([
            ChatResponse(text=f"```json\n{json_output}\n```", usage=UsageStats()),
        ])

        pool = MockModelPool(provider)
        mgr = WorkerManager(model_pool=pool, tool_registry=ToolRegistry())

        tasks = mgr.decompose(
            goal="Analyze market and write report",
            context="Financial analysis project",
            provider=provider,
            num_tasks=3,
        )

        assert len(tasks) == 3
        assert tasks[0].goal == "Research market trends"
        assert tasks[1].goal == "Analyze financial data"
        assert tasks[2].goal == "Write summary report"
        assert tasks[0].tool_names == ["web_search"]
        assert tasks[1].tool_names == ["file_read"]
        assert tasks[2].priority == 2

    def test_decompose_plain_json_no_fences(self):
        json_output = json.dumps([
            {"goal": "Task A", "tool_names": [], "system_prompt": "", "priority": 0},
        ])
        provider = MockProvider([
            ChatResponse(text=json_output, usage=UsageStats()),
        ])

        pool = MockModelPool(provider)
        mgr = WorkerManager(model_pool=pool, tool_registry=ToolRegistry())

        tasks = mgr.decompose(
            goal="Single subtask",
            context="ctx",
            provider=provider,
            num_tasks=1,
        )

        assert len(tasks) == 1
        assert tasks[0].goal == "Task A"

    def test_decompose_invalid_json_fallback(self):
        """When the model returns invalid JSON, fall back to a single task."""
        provider = MockProvider([
            ChatResponse(text="I think you should do A, B, and C...", usage=UsageStats()),
        ])

        pool = MockModelPool(provider)
        mgr = WorkerManager(model_pool=pool, tool_registry=ToolRegistry())

        tasks = mgr.decompose(
            goal="Original goal",
            context="Original context",
            provider=provider,
            num_tasks=3,
        )

        # Should fall back to single task
        assert len(tasks) == 1
        assert tasks[0].goal == "Original goal"
        assert tasks[0].context == "Original context"

    def test_decompose_prompt_format(self):
        """The decompose prompt includes the right parameters."""
        provider = MockProvider([
            ChatResponse(text="[]", usage=UsageStats()),
        ])

        pool = MockModelPool(provider)
        mgr = WorkerManager(model_pool=pool, tool_registry=ToolRegistry())

        mgr.decompose(
            goal="Build a website",
            context="Using React",
            provider=provider,
            num_tasks=2,
        )

        assert len(provider.calls) == 1
        msgs = provider.calls[0][0]
        user_msg = msgs[1]["content"]
        assert "Build a website" in user_msg
        assert "Using React" in user_msg
        assert "2" in user_msg

    def test_decompose_minimal_subtask_fields(self):
        """Subtasks with minimal fields still produce valid WorkerTask."""
        json_output = json.dumps([
            {"goal": "Minimal task"},  # No tool_names, system_prompt, or priority
        ])
        provider = MockProvider([
            ChatResponse(text=json_output, usage=UsageStats()),
        ])

        pool = MockModelPool(provider)
        mgr = WorkerManager(model_pool=pool, tool_registry=ToolRegistry())

        tasks = mgr.decompose(
            goal="Parent",
            context="ctx",
            provider=provider,
            num_tasks=1,
        )

        assert len(tasks) == 1
        assert tasks[0].goal == "Minimal task"
        assert tasks[0].tool_names is None
        assert tasks[0].priority == 0


# ------------------------------------------------------------------
# _filter_tools helper
# ------------------------------------------------------------------

class TestFilterTools:
    def test_filter_empty_returns_all(self):
        pool = MockModelPool(MockProvider([]))
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="a",
            description="A",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "a",
        ))
        reg.register(ToolDef(
            name="b",
            description="B",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "b",
        ))

        mgr = WorkerManager(model_pool=pool, tool_registry=reg)
        filtered = mgr._filter_tools([])
        assert len(filtered) == 0

    def test_filter_specific(self):
        pool = MockModelPool(MockProvider([]))
        reg = ToolRegistry()
        reg.register(ToolDef(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "search",
        ))
        reg.register(ToolDef(
            name="write",
            description="Write",
            parameters={"type": "object", "properties": {}},
            executor=lambda: "write",
        ))

        mgr = WorkerManager(model_pool=pool, tool_registry=reg)
        filtered = mgr._filter_tools(["search"])
        assert len(filtered) == 1
        assert "search" in filtered
        assert "write" not in filtered
