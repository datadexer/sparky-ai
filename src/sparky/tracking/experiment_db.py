"""SQLite experiment tracking — prevents duplicate configs, enables querying.

Usage:
    from sparky.tracking.experiment_db import get_db, is_duplicate, log_experiment, get_best

    db = get_db()
    h = config_hash({"model": "xgboost", "lr": 0.05})
    if not is_duplicate(db, h):
        # run experiment...
        log_experiment(db, config_hash=h, model_type="xgboost", ...)
    best = get_best(db, n=5)
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("results/experiments.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL,
    approach_family TEXT NOT NULL DEFAULT '',
    features TEXT NOT NULL DEFAULT '[]',
    hyperparams TEXT NOT NULL DEFAULT '{}',
    sharpe REAL,
    wf_mean_sharpe REAL,
    tier TEXT,
    wall_clock_seconds REAL,
    timestamp TEXT NOT NULL,
    notes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_config_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_sharpe ON experiments(sharpe);
CREATE INDEX IF NOT EXISTS idx_model_type ON experiments(model_type);
"""


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get or create the experiment database connection.

    Args:
        db_path: Path to SQLite DB file. Defaults to results/experiments.db.

    Returns:
        sqlite3.Connection with row_factory set.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of an experiment config.

    Sorts keys recursively to ensure stable hashing.

    Args:
        config: Dictionary of model config (hyperparams, features, etc.)

    Returns:
        SHA-256 hex digest (first 16 chars).
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def is_duplicate(db: sqlite3.Connection, cfg_hash: str) -> bool:
    """Check if an experiment config has already been run.

    Args:
        db: Database connection.
        cfg_hash: Hash from config_hash().

    Returns:
        True if this config was already logged.
    """
    row = db.execute(
        "SELECT 1 FROM experiments WHERE config_hash = ?", (cfg_hash,)
    ).fetchone()
    return row is not None


def log_experiment(
    db: sqlite3.Connection,
    *,
    config_hash: str,
    model_type: str,
    approach_family: str = "",
    features: Optional[list[str]] = None,
    hyperparams: Optional[dict[str, Any]] = None,
    sharpe: Optional[float] = None,
    wf_mean_sharpe: Optional[float] = None,
    tier: Optional[str] = None,
    wall_clock_seconds: Optional[float] = None,
    notes: str = "",
) -> int:
    """Log a completed experiment to the database.

    Args:
        db: Database connection.
        config_hash: Unique hash identifying the config.
        model_type: Model type (xgboost, catboost, lightgbm, etc.)
        approach_family: Approach family (tree_ensemble, regime, etc.)
        features: List of feature names used.
        hyperparams: Dictionary of hyperparameters.
        sharpe: Overall Sharpe ratio.
        wf_mean_sharpe: Walk-forward mean Sharpe.
        tier: Graduated tier (TIER 1-5).
        wall_clock_seconds: Wall-clock time in seconds.
        notes: Free-form notes.

    Returns:
        Row ID of the inserted experiment.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    cursor = db.execute(
        """INSERT OR REPLACE INTO experiments
        (config_hash, model_type, approach_family, features, hyperparams,
         sharpe, wf_mean_sharpe, tier, wall_clock_seconds, timestamp, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            config_hash,
            model_type,
            approach_family,
            json.dumps(features or []),
            json.dumps(hyperparams or {}),
            sharpe,
            wf_mean_sharpe,
            tier,
            wall_clock_seconds,
            timestamp,
            notes,
        ),
    )
    db.commit()
    logger.info(f"[EXP DB] Logged experiment {config_hash}: {model_type} Sharpe={sharpe}")
    return cursor.lastrowid


def get_best(db: sqlite3.Connection, n: int = 5) -> list[dict[str, Any]]:
    """Get the top N experiments by Sharpe ratio.

    Args:
        db: Database connection.
        n: Number of results to return.

    Returns:
        List of experiment dicts sorted by Sharpe descending.
    """
    rows = db.execute(
        "SELECT * FROM experiments WHERE sharpe IS NOT NULL ORDER BY sharpe DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_summary(db: sqlite3.Connection) -> dict[str, Any]:
    """Get a summary of all experiments in the database.

    Returns:
        Dict with total_experiments, by_model_type counts, best_sharpe,
        tier_distribution, and recent experiments.
    """
    total = db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    by_model = db.execute(
        "SELECT model_type, COUNT(*) as cnt FROM experiments GROUP BY model_type"
    ).fetchall()
    by_tier = db.execute(
        "SELECT tier, COUNT(*) as cnt FROM experiments WHERE tier IS NOT NULL GROUP BY tier"
    ).fetchall()
    best = db.execute(
        "SELECT config_hash, model_type, sharpe FROM experiments "
        "WHERE sharpe IS NOT NULL ORDER BY sharpe DESC LIMIT 1"
    ).fetchone()
    recent = db.execute(
        "SELECT config_hash, model_type, sharpe, timestamp FROM experiments "
        "ORDER BY timestamp DESC LIMIT 5"
    ).fetchall()

    return {
        "total_experiments": total,
        "by_model_type": {row["model_type"]: row["cnt"] for row in by_model},
        "by_tier": {row["tier"]: row["cnt"] for row in by_tier},
        "best": dict(best) if best else None,
        "recent": [dict(r) for r in recent],
    }
