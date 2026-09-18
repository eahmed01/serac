"""Worker Manager — Tier 2: task decomposition and parallel worker dispatch.

Receives a batch task from the orchestrator, decomposes it into N worker tasks,
dispatches them concurrently through the model pool, and coalesces results.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from agent_framework.loop import AgentLoop
from agent_framework.providers import Provider, UsageStats
from agent_framework.tools import ToolRegistry
from agent_framework.tracing import Tracer

if TYPE_CHECKING:
    from agent_framework.model_pool import ModelPool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkerTask:
    """A single task to dispatch to a worker agent."""
    task_id: str
    goal: str          # What the worker should do
    context: str       # Background info for the worker
    system_prompt: str = ""  # System prompt (empty = use generic)
    tool_names: Optional[list[str]] = None  # None = default/all; [] = explicitly no tools
    max_turns: int = 20    # Max turns for this worker's agent loop
    priority: int = 0      # Dispatch priority (lower = first)


@dataclass
class WorkerResult:
    """Result from a worker agent."""
    task_id: str
    success: bool
    output: str       # The worker's final text response
    error: str        # Error message if failed
    duration: float   # Seconds elapsed
    turns_used: int   # How many turns the worker took
    usage: UsageStats  # Token usage for this worker
    todo_state: list = field(default_factory=list)  # Final todo list from workspace


# ---------------------------------------------------------------------------
# WorkerManager
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a focused worker agent. Complete the task described below. "
    "Be concise and return your final result clearly."
)

_DECOMPOSE_PROMPT = (
    "You are a task decomposition assistant. Break the following high-level goal\n"
    "into {n} parallel, independent subtasks that together accomplish the goal.\n\n"
    "Goal: {goal}\n\n"
    "Context: {context}\n\n"
    "Return a JSON array with exactly {n} objects. Each object must have:\n"
    "- \"goal\": a clear, actionable subtask description (string)\n"
    "- \"tool_names\": a list of tool names this subtask needs (e.g. [\"web_search\", \"file_read\"])\n"
    "- \"system_prompt\": an optional system prompt for the subtask worker (string, can be empty)\n"
    "- \"priority\": integer dispatch priority (0 = highest)\n\n"
    "Return ONLY the JSON array, no explanation or markdown."
)


class WorkerManager:
    """Tier 2: Task decomposition and parallel worker dispatch.

    Receives a batch task from the orchestrator, decomposes it into N worker tasks,
    dispatches them concurrently through the model pool, and coalesces results.

    Args:
        model_pool: Model pool for routing workers to providers
        tool_registry: Base tool registry (filtered per worker by tool_names)
        max_parallel: Maximum concurrent workers (default 6, matches vLLM capacity)
    """

    def __init__(
        self,
        model_pool: ModelPool,
        tool_registry: ToolRegistry,
        max_parallel: int = 6,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.model_pool = model_pool
        self.tool_registry = tool_registry
        self.max_parallel = max_parallel
        self.tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, tasks: list[WorkerTask]) -> list[WorkerResult]:
        """Execute a batch of worker tasks concurrently.

        Dispatches tasks through the model pool, respecting max_parallel limit.
        Uses a semaphore to control concurrency. Coalesces results on completion.

        Args:
            tasks: List of worker tasks to execute

        Returns:
            List of worker results, ordered by task_id
        """
        if not tasks:
            return []

        # Sort by priority (lower = first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        results: dict[str, WorkerResult] = {}
        semaphore = __import__("threading").Semaphore(self.max_parallel)

        def _run_task(task: WorkerTask) -> WorkerResult:
            return self._execute_worker(task)

        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(sorted_tasks))) as pool:
            futures = {
                pool.submit(_run_task, task): task
                for task in sorted_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results[result.task_id] = result
                except Exception as exc:
                    logger.exception("[%s] worker task failed unexpectedly", task.task_id)
                    results[task.task_id] = WorkerResult(
                        task_id=task.task_id,
                        success=False,
                        output="",
                        error=str(exc),
                        duration=0.0,
                        turns_used=0,
                        usage=UsageStats(),
                    )

        # Return ordered by task_id
        return [results[tid] for tid in sorted(results.keys())]

    def decompose(
        self,
        goal: str,
        context: str,
        provider: Provider,
        num_tasks: int = 3,
    ) -> list[WorkerTask]:
        """Ask a model to decompose a high-level goal into parallel subtasks.

        Args:
            goal: High-level task description
            context: Background information
            provider: Model to ask for decomposition
            num_tasks: Number of subtasks to create

        Returns:
            List of WorkerTask objects
        """
        prompt = _DECOMPOSE_PROMPT.format(n=num_tasks, goal=goal, context=context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a task decomposition assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        response = provider.chat(messages)
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            subtasks = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Decompose: failed to parse JSON from model: %s", exc)
            # Fallback: single task with the original goal
            return [WorkerTask(
                task_id=str(uuid.uuid4())[:8],
                goal=goal,
                context=context,
            )]

        tasks: list[WorkerTask] = []
        for i, sub in enumerate(subtasks):
            tasks.append(WorkerTask(
                task_id=str(uuid.uuid4())[:8],
                goal=sub.get("goal", f"Subtask {i + 1}"),
                context=context,
                system_prompt=sub.get("system_prompt", ""),
                tool_names=sub.get("tool_names"),
                max_turns=20,
                priority=sub.get("priority", i),
            ))

        logger.info("Decomposed into %d subtasks", len(tasks))
        return tasks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_worker(self, task: WorkerTask) -> WorkerResult:
        """Run a single worker task in a thread."""
        start = time.monotonic()
        routing = None

        try:
            # Route to provider via model pool
            routing = self.model_pool.route()
            provider = routing.provider

            # Build filtered tool registry for this worker
            worker_tools = self._filter_tools(task.tool_names)

            # System prompt
            system_prompt = task.system_prompt or _DEFAULT_SYSTEM_PROMPT

            # Combine context + goal for the user message
            user_message = task.goal
            if task.context:
                user_message = (
                    f"[CONTEXT]\n{task.context}\n\n"
                    f"[TASK]\n{task.goal}"
                )

            # Run the agent loop
            loop = AgentLoop(
                provider=provider,
                system_prompt=system_prompt,
                tool_registry=worker_tools,
                max_turns=task.max_turns,
                name=f"worker-{task.task_id}",
            )

            output = loop.run(user_message)
            duration = time.monotonic() - start

            # Read the todo state from the workspace if the todo tool was used
            todo_state = self._read_todo_state(task)

            return WorkerResult(
                task_id=task.task_id,
                success=True,
                output=output,
                error="",
                duration=duration,
                turns_used=loop.turn_count,
                usage=loop.total_usage,
                todo_state=todo_state,
            )

        except Exception as exc:
            duration = time.monotonic() - start
            logger.exception("[%s] worker execution failed", task.task_id)
            return WorkerResult(
                task_id=task.task_id,
                success=False,
                output="",
                error=str(exc),
                duration=duration,
                turns_used=0,
                usage=UsageStats(),
            )

        finally:
            # Always release the pool slot
            if routing is not None:
                self.model_pool.release(routing.slot_id)

    def _filter_tools(self, tool_names: Optional[list[str]]) -> ToolRegistry:
        """Create a ToolRegistry filtered to the specified tool names.

        If tool_names is omitted (None), returns all registered tools for
        compatibility. An explicitly empty list returns no tools.

        Args:
            tool_names: List of tool names to include.

        Returns:
            Filtered ToolRegistry.
        """
        filtered = ToolRegistry()
        if tool_names is None:
            # Include all tools
            for td in self.tool_registry.tools.values():
                filtered.register(td)
        else:
            for name in tool_names:
                if name in self.tool_registry.tools:
                    filtered.register(self.tool_registry.tools[name])
                else:
                    logger.warning("Requested tool '%s' not found in registry", name)
        return filtered

    def _read_todo_state(self, task: WorkerTask) -> list:
        """Read the todo state from the workspace after worker execution.

        Args:
            task: The worker task that was executed.

        Returns:
            List of todo items from the workspace, or empty list if not found.
        """
        try:
            import os
            todo_path = os.path.join(os.getcwd(), ".todo.json")
            if os.path.exists(todo_path):
                with open(todo_path, "r") as f:
                    data = json.load(f)
                return data.get("items", [])
        except Exception as exc:
            logger.debug("Failed to read todo state: %s", exc)
        return []
