"""
Synthetic data fixtures for sandbox tests.

Generates in-memory OHLCV DataFrames with a standard schema,
so tests don't need disk access or real parquet files.
"""

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    tickers: int = 50,
    days: int = 500,
    seed: int = 42,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data.

    Schema:
        entity_id, ticker, time, raw_open, raw_high, raw_low, raw_close,
        raw_volume, unadj_close, dollar_volume, in_universe

    Uses random walk prices with drift and log-normal volumes.

    Args:
        tickers: Number of unique tickers/entities.
        days: Number of trading days per ticker.
        seed: Random seed for reproducibility.
        start_date: Starting date string.

    Returns:
        DataFrame sorted by (ticker, time), reset index.
    """
    rng = np.random.default_rng(seed)

    # Trading dates (skip weekends)
    dates = pd.bdate_range(start_date, periods=days)

    rows = []
    for i in range(tickers):
        ticker = f"TK{ i:04d}"
        entity_id = f"BBG{ i:010d}"

        # Random walk with slight positive drift
        returns = rng.normal(0.0003, 0.015, size=days)
        prices = 100 * np.cumprod(1 + returns)

        # Generate OHLCV from close price
        noise = rng.uniform(0.005, 0.02, size=days)
        close = prices
        open_ = close * (1 + rng.normal(0, 0.005, size=days))
        high = np.maximum(open_, close) * (1 + noise)
        low = np.minimum(open_, close) * (1 - noise)
        volume = rng.lognormal(mean=15, sigma=1, size=days).astype(int)

        for d in range(days):
            rows.append({
                "entity_id": entity_id,
                "ticker": ticker,
                "time": dates[d],
                "raw_open": round(float(open_[d]), 4),
                "raw_high": round(float(high[d]), 4),
                "raw_low": round(float(low[d]), 4),
                "raw_close": round(float(close[d]), 4),
                "raw_volume": int(volume[d]),
                "unadj_close": round(float(close[d]), 4),
                "dollar_volume": int(volume[d] * close[d]),
                "in_universe": True,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "time"]).reset_index(drop=True)
    return df


def make_panel_frame(
    tickers: int = 50,
    days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic panel-style MultiIndex DataFrame.

    Index: (ticker, time) — a long panel reset to a (ticker, time) index.
    Columns: the full synthetic OHLCV columns.

    Args:
        tickers: Number of tickers.
        days: Number of trading days.
        seed: Random seed.

    Returns:
        MultiIndex DataFrame indexed by (ticker, time).
    """
    ohlcv = make_synthetic_ohlcv(tickers=tickers, days=days, seed=seed)
    x = ohlcv.set_index(["ticker", "time"])
    return x
