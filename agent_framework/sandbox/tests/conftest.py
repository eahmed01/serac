"""
Conftest — shared fixtures for sandbox tests.
"""

import pytest
from agent_framework.sandbox.tests.fixtures import make_synthetic_ohlcv, make_panel_frame


@pytest.fixture
def synthetic_ohlcv():
    """50 tickers, 500 days of synthetic OHLCV."""
    return make_synthetic_ohlcv(tickers=50, days=500, seed=42)


@pytest.fixture
def synthetic_panel():
    """MultiIndex (ticker, time) panel DataFrame."""
    return make_panel_frame(tickers=50, days=500, seed=42)
