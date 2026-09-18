"""
Experiment tracker — SQLite-backed store for research pipeline results.

Stores training results, backtest results, and metadata in a single database.
Supports querying, filtering, and comparison across experiments.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    experiment_type TEXT NOT NULL CHECK(experiment_type IN ('train', 'backtest')),
    timestamp TEXT NOT NULL,
    config_id TEXT,
    threshold REAL,
    params TEXT,
    results TEXT,
    notes TEXT,
    source_file TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_type ON experiments(experiment_type);
CREATE INDEX IF NOT EXISTS idx_exp_config ON experiments(config_id);
CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiments(timestamp);
"""


class ExperimentTracker:
    """SQLite-backed experiment tracker.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and schema if needed."""
        db = Path(self._db_path)
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db))
        conn.executescript(SCHEMA)
        conn.close()

    def _get_conn(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _make_run_id(self, prefix: str, config_id: str) -> str:
        """Generate a unique run ID using UUID4 hex (12 chars).

        Scales to ~281 trillion unique IDs (48 bits = 2^48). Each tracker instance
        is independent — no shared class-level counter state.
        """
        return f"{prefix}_{config_id}_{uuid.uuid4().hex[:12]}"

    def log_experiment(
        self,
        experiment_type: str,
        config_id: str,
        params: Dict[str, Any],
        results: Dict[str, Any],
        notes: str = "",
        source_file: str = "",
    ) -> str:
        """Log an experiment result.

        Returns the run_id.
        """
        conn = self._get_conn()
        run_id = self._make_run_id(experiment_type, config_id)
        conn.execute(
            """INSERT OR REPLACE INTO experiments
               (run_id, experiment_type, timestamp, config_id, threshold, params, results, notes, source_file)
               VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)""",
            (
                run_id,
                experiment_type,
                datetime.now(timezone.utc).isoformat(),
                config_id,
                json.dumps(params),
                json.dumps(results),
                notes,
                source_file,
            ),
        )
        conn.commit()
        conn.close()
        return run_id

    def log_backtest(
        self,
        config_id: str,
        threshold: float,
        results: Dict[str, Any],
        notes: str = "",
        source_file: str = "",
    ) -> str:
        """Log a backtest result.

        Returns the run_id.
        """
        conn = self._get_conn()
        run_id = self._make_run_id("bt", config_id)
        conn.execute(
            """INSERT OR REPLACE INTO experiments
               (run_id, experiment_type, timestamp, config_id, threshold, params, results, notes, source_file)
               VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
            (
                run_id,
                "backtest",
                datetime.now(timezone.utc).isoformat(),
                config_id,
                threshold,
                json.dumps(results),
                notes,
                source_file,
            ),
        )
        conn.commit()
        conn.close()
        return run_id

    def list_experiments(
        self,
        experiment_type: Optional[str] = None,
        config_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filters.

        Returns list of experiment dicts ordered by timestamp (newest first).
        """
        conn = self._get_conn()
        query = """SELECT id, run_id, experiment_type, timestamp, config_id,
                   threshold, notes FROM experiments WHERE 1=1"""
        params: list = []

        if experiment_type:
            query += " AND experiment_type = ?"
            params.append(experiment_type)
        if config_id:
            query += " AND config_id = ?"
            params.append(config_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_experiment(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get full experiment details.

        Returns None if not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        d = dict(row)
        for k in ("params", "results"):
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    def _resolve_metric(self, data: Dict[str, Any], metric: str) -> Any:
        """Resolve a metric value from a results dict.

        Checks top-level keys first, then falls back to data["backtest"].
        Returns None if the metric is not found.
        """
        if metric in data:
            return data[metric]
        if "backtest" in data and isinstance(data["backtest"], dict):
            return data["backtest"].get(metric)
        return None

    def best(
        self,
        metric: str,
        experiment_type: Optional[str] = None,
        result_limit: int = 1000,
    ) -> Tuple[Optional[Dict], Optional[float]]:
        """Find the experiment with the best value for a given metric.

        Returns (row, best_value). Metric can be in results dict or in
        results["backtest"] sub-dict.

        Parameters
        ----------
        metric : str
            Metric name to optimize (higher is better).
        experiment_type : str, optional
            Filter by type ('train' or 'backtest').
        result_limit : int, default 1000
            Maximum number of rows to consider. Prevents loading the
            entire table into memory when called without a type filter.

        Uses a single database query (SELECT with optional type filter
        and ORDER BY) to avoid the N+1 query pattern where each row
        required a separate get_experiment() call.

        Tie-breaking: when multiple rows share the same best metric
        value, the most recent row (by timestamp) is returned.
        """
        conn = self._get_conn()

        # Narrow SELECT to only the columns best() actually needs —
        # exclude 'params' which is never used in the result.
        query = """SELECT id, run_id, experiment_type, timestamp, config_id,
                   threshold, results, notes FROM experiments WHERE 1=1"""
        params: list = []

        if experiment_type:
            query += " AND experiment_type = ?"
            params.append(experiment_type)

        # ORDER BY timestamp DESC for deterministic tie-breaking at the
        # SQL level (most recent first), plus LIMIT to bound memory usage
        # for unfiltered calls.
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(result_limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        best_row: Optional[Dict] = None
        best_val: Optional[float] = None

        for row in rows:
            d = dict(row)

            # Parse results JSON inline — no separate query needed
            raw_results = d.get("results")
            if not raw_results:
                continue
            data: Dict[str, Any] = json.loads(raw_results)

            val = self._resolve_metric(data, metric)
            if val is None:
                continue

            # Reject non-numeric types (booleans, strings, lists, etc.).
            # bool is a subclass of int in Python, so check it first.
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue

            # Deterministic tie-breaking: prefer the row with the most
            # recent timestamp. Since rows come ORDER BY timestamp DESC,
            # strict '>' keeps the first (newest) among equals.
            val_float = float(val)
            if best_val is None or val_float > best_val:
                best_val = val_float
                best_row = {
                    "id": d["id"],
                    "run_id": d["run_id"],
                    "experiment_type": d["experiment_type"],
                    "timestamp": d["timestamp"],
                    "config_id": d["config_id"],
                    "threshold": d["threshold"],
                    "notes": d["notes"],
                }

        return best_row, best_val
