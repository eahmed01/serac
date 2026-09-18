"""Cost-aware model routing with capacity tracking.

Routes worker tasks to the first pool entry with available capacity.
Local vLLM (6 slots, $0) fills first, then overflow routes to cheapest API provider.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from agent_framework.providers import Provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProviderSlot:
    """A model pool entry with capacity limits and cost-based priority."""

    provider: Provider
    max_concurrent: int
    priority: int  # Lower = cheaper, dispatched first
    model_name: str  # Display name for logging
    cost_per_token: float = 0.0


@dataclass
class PoolRoutingResult:
    """Result of routing a request to a provider."""

    provider: Provider
    model_name: str
    slot_id: int  # Which slot was used
    cost_per_token: float = 0.0


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------

class ModelPool:
    """Cost-aware model routing with capacity tracking.

    Routes worker tasks to the first pool entry with available capacity.
    Local vLLM (6 slots, $0) fills first, then overflow routes to cheapest API provider.

    Example:
        pool = ModelPool()
        pool.add_slot(VLLMProvider(), max_concurrent=6, priority=0, model_name="vllm-qwen")
        pool.add_slot(AnthropicProvider(model="claude-sonnet-4"), max_concurrent=2, priority=1, model_name="sonnet")

        # Route 8 workers: 6 → vLLM, 2 → Sonnet
        for _ in range(8):
            result = pool.route()  # Thread-safe, respects capacity
            print(f"Routed to {result.model_name}")

    Thread-safe: uses threading.Lock() for concurrent session tracking.
    """

    def __init__(self) -> None:
        self._slots: list[ProviderSlot] = []
        self._active_counts: dict[int, int] = {}  # slot index → active count
        self._lock = threading.Lock()

    def add_slot(
        self,
        provider: Provider,
        max_concurrent: int = 1,
        priority: int = 0,
        model_name: str = "unknown",
        cost_per_token: float = 0.0,
    ) -> None:
        """Add a provider slot to the pool.

        Slots are sorted by priority (lower = dispatched first) so that
        cheaper/local providers are exhausted before overflow providers.
        """
        slot = ProviderSlot(
            provider=provider,
            max_concurrent=max_concurrent,
            priority=priority,
            model_name=model_name,
            cost_per_token=cost_per_token,
        )
        with self._lock:
            # Insert in sorted order by priority
            inserted = False
            for i, existing in enumerate(self._slots):
                if slot.priority < existing.priority:
                    self._slots.insert(i, slot)
                    inserted = True
                    break
            if not inserted:
                self._slots.append(slot)
            # Rebuild active counts to match new indices
            self._active_counts = {i: 0 for i in range(len(self._slots))}

    def route(self) -> PoolRoutingResult:
        """Route a request to the first available slot (lowest priority with capacity).

        Raises ValueError if no slot has available capacity.
        """
        with self._lock:
            for idx, slot in enumerate(self._slots):
                active = self._active_counts.get(idx, 0)
                if active < slot.max_concurrent:
                    self._active_counts[idx] = active + 1
                    logger.debug(
                        "Routed to %s (slot %d, %d/%d active)",
                        slot.model_name,
                        idx,
                        active + 1,
                        slot.max_concurrent,
                    )
                    return PoolRoutingResult(
                        provider=slot.provider,
                        model_name=slot.model_name,
                        slot_id=idx,
                        cost_per_token=slot.cost_per_token,
                    )
        raise ValueError("All provider slots at capacity")

    def release(self, slot_id: int) -> None:
        """Release a slot after worker completes."""
        with self._lock:
            current = self._active_counts.get(slot_id, 0)
            if current > 0:
                self._active_counts[slot_id] = current - 1
                logger.debug(
                    "Released slot %d (%d active remaining)",
                    slot_id,
                    current - 1,
                )
            else:
                logger.warning("Released already-free slot %d", slot_id)

    @property
    def available_capacity(self) -> dict[str, int]:
        """Return available capacity per slot name."""
        with self._lock:
            result: dict[str, int] = {}
            for idx, slot in enumerate(self._slots):
                active = self._active_counts.get(idx, 0)
                result[slot.model_name] = slot.max_concurrent - active
            return result

    @property
    def total_capacity(self) -> int:
        """Total max concurrent across all slots."""
        with self._lock:
            return sum(s.max_concurrent for s in self._slots)
