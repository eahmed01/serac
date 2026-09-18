"""Tests for model_pool.py — ModelPool, ProviderSlot, PoolRoutingResult."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from agent_framework.model_pool import ModelPool, PoolRoutingResult, ProviderSlot
from agent_framework.providers import Provider


class _MockProvider(Provider):
    """Minimal Provider implementation for testing."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name

    def chat(self, messages, tools=None):  # type: ignore[override]
        return MagicMock(text="ok")


# ---------------------------------------------------------------------------
# ProviderSlot
# ---------------------------------------------------------------------------

class TestProviderSlot:
    def test_fields(self):
        p = _MockProvider("test")
        slot = ProviderSlot(
            provider=p,
            max_concurrent=4,
            priority=1,
            model_name="test-model",
            cost_per_token=0.001,
        )
        assert slot.provider is p
        assert slot.max_concurrent == 4
        assert slot.priority == 1
        assert slot.model_name == "test-model"
        assert slot.cost_per_token == 0.001

    def test_default_cost_is_zero(self):
        slot = ProviderSlot(
            provider=_MockProvider(),
            max_concurrent=1,
            priority=0,
            model_name="local",
        )
        assert slot.cost_per_token == 0.0


# ---------------------------------------------------------------------------
# ModelPool — basic routing
# ---------------------------------------------------------------------------

class TestModelPoolRouting:
    def test_empty_pool_raises(self):
        pool = ModelPool()
        with pytest.raises(ValueError, match="All provider slots at capacity"):
            pool.route()

    def test_single_slot_routes(self):
        pool = ModelPool()
        prov = _MockProvider("single")
        pool.add_slot(prov, max_concurrent=1, priority=0, model_name="single")

        result = pool.route()
        assert result.provider is prov
        assert result.model_name == "single"
        assert result.slot_id == 0

    def test_capacity_limit(self):
        pool = ModelPool()
        prov = _MockProvider()
        pool.add_slot(prov, max_concurrent=2, priority=0, model_name="limited")

        r1 = pool.route()
        r2 = pool.route()
        assert r1.slot_id == 0
        assert r2.slot_id == 0

        with pytest.raises(ValueError, match="All provider slots at capacity"):
            pool.route()

    def test_release_restores_capacity(self):
        pool = ModelPool()
        prov = _MockProvider()
        pool.add_slot(prov, max_concurrent=1, priority=0, model_name="one")

        r1 = pool.route()
        with pytest.raises(ValueError):
            pool.route()

        pool.release(r1.slot_id)
        r2 = pool.route()
        assert r2.slot_id == 0


# ---------------------------------------------------------------------------
# ModelPool — priority ordering
# ---------------------------------------------------------------------------

class TestModelPoolPriority:
    def test_lower_priority_dispatched_first(self):
        pool = ModelPool()
        p_cheap = _MockProvider("cheap")
        p_expensive = _MockProvider("expensive")

        # Add expensive first (priority 1), cheap second (priority 0)
        pool.add_slot(p_expensive, max_concurrent=2, priority=1, model_name="expensive")
        pool.add_slot(p_cheap, max_concurrent=2, priority=0, model_name="cheap")

        r1 = pool.route()
        r2 = pool.route()
        # Both should go to cheap first
        assert r1.model_name == "cheap"
        assert r2.model_name == "cheap"

        # Now cheap is full, should overflow to expensive
        r3 = pool.route()
        assert r3.model_name == "expensive"

    def test_local_then_overflow_pattern(self):
        """6 local slots fill first, then API overflow."""
        pool = ModelPool()
        local = _MockProvider("vllm")
        api = _MockProvider("api")

        pool.add_slot(api, max_concurrent=4, priority=1, model_name="api", cost_per_token=0.003)
        pool.add_slot(local, max_concurrent=6, priority=0, model_name="vllm", cost_per_token=0.0)

        results: list[str] = []
        for _ in range(8):
            results.append(pool.route().model_name)

        assert results == ["vllm"] * 6 + ["api"] * 2

    def test_multiple_slots_same_priority(self):
        pool = ModelPool()
        pool.add_slot(_MockProvider(), max_concurrent=1, priority=0, model_name="a")
        pool.add_slot(_MockProvider(), max_concurrent=1, priority=0, model_name="b")

        r1 = pool.route()
        r2 = pool.route()
        # Both priority-0 slots used
        assert r1.model_name in ("a", "b")
        assert r2.model_name != r1.model_name

        with pytest.raises(ValueError):
            pool.route()


# ---------------------------------------------------------------------------
# ModelPool — cost_per_token tracking
# ---------------------------------------------------------------------------

class TestModelPoolCost:
    def test_cost_propagates_to_result(self):
        pool = ModelPool()
        pool.add_slot(_MockProvider(), max_concurrent=1, priority=0, model_name="api", cost_per_token=0.002)

        result = pool.route()
        assert result.cost_per_token == 0.002

    def test_zero_cost_for_local(self):
        pool = ModelPool()
        pool.add_slot(_MockProvider(), max_concurrent=1, priority=0, model_name="local")

        result = pool.route()
        assert result.cost_per_token == 0.0


# ---------------------------------------------------------------------------
# ModelPool — capacity properties
# ---------------------------------------------------------------------------

class TestModelPoolCapacity:
    def test_total_capacity(self):
        pool = ModelPool()
        pool.add_slot(_MockProvider(), max_concurrent=6, priority=0, model_name="vllm")
        pool.add_slot(_MockProvider(), max_concurrent=2, priority=1, model_name="api")
        assert pool.total_capacity == 8

    def test_available_capacity_initial(self):
        pool = ModelPool()
        pool.add_slot(_MockProvider(), max_concurrent=3, priority=0, model_name="a")
        pool.add_slot(_MockProvider(), max_concurrent=2, priority=1, model_name="b")

        cap = pool.available_capacity
        assert cap == {"a": 3, "b": 2}

    def test_available_capacity_decreases(self):
        pool = ModelPool()
        pool.add_slot(_MockProvider(), max_concurrent=2, priority=0, model_name="x")

        r = pool.route()
        cap = pool.available_capacity
        assert cap["x"] == 1

        pool.release(r.slot_id)
        cap = pool.available_capacity
        assert cap["x"] == 2


# ---------------------------------------------------------------------------
# ModelPool — thread safety
# ---------------------------------------------------------------------------

class TestModelPoolThreadSafety:
    def test_concurrent_routing_respects_capacity(self):
        """Multiple threads routing simultaneously should not exceed max_concurrent."""
        pool = ModelPool()
        prov = _MockProvider()
        pool.add_slot(prov, max_concurrent=4, priority=0, model_name="shared")

        results: list[PoolRoutingResult] = []
        errors: list[ValueError] = []
        lock = threading.Lock()

        def route_once():
            try:
                r = pool.route()
                with lock:
                    results.append(r)
            except ValueError as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=route_once) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most 4 should succeed (max_concurrent=4)
        assert len(results) == 4
        assert len(errors) == 6
        assert all(r.slot_id == 0 for r in results)

    def test_concurrent_release_and_route(self):
        """Release from one thread while another routes."""
        pool = ModelPool()
        prov = _MockProvider()
        pool.add_slot(prov, max_concurrent=1, priority=0, model_name="single")

        results: list[PoolRoutingResult] = []
        barrier = threading.Barrier(2)

        def hold_and_release():
            r = pool.route()
            barrier.wait()
            # Hold briefly then release
            import time
            time.sleep(0.05)
            pool.release(r.slot_id)

        def wait_and_route():
            barrier.wait()
            import time
            time.sleep(0.1)
            r = pool.route()
            results.append(r)

        t1 = threading.Thread(target=hold_and_release)
        t2 = threading.Thread(target=wait_and_route)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 1
        assert results[0].slot_id == 0
