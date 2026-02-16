#!/usr/bin/env python3
"""Backfill experiment DB from existing result JSONs in results/validation/.

Parses JSON result files and inserts entries into the experiment database.
Run once to seed the DB with historical results.

Usage:
    PYTHONPATH=. python3 scripts/backfill_experiment_db.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparky.tracking.experiment_db import get_db, config_hash, log_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/validation")


def _extract_sharpe(data: dict) -> float | None:
    """Try to find a Sharpe ratio in various result formats."""
    # Direct sharpe field
    if "sharpe_ratio" in data:
        return data["sharpe_ratio"]
    if "sharpe" in data:
        return data["sharpe"]
    # Nested in strategy results
    if "out_of_sample_periods" in data:
        for period_data in data["out_of_sample_periods"].values():
            if isinstance(period_data, dict) and "strategy" in period_data:
                return period_data["strategy"].get("sharpe_ratio")
    # Walk-forward results
    if "walk_forward" in data:
        wf = data["walk_forward"]
        if "mean_sharpe" in wf:
            return wf["mean_sharpe"]
    # Top-level results list
    if "results" in data and isinstance(data["results"], list):
        for r in data["results"]:
            if isinstance(r, dict) and "sharpe" in r:
                return r["sharpe"]
    return None


def _extract_model_type(data: dict, filename: str) -> str:
    """Infer model type from result data or filename."""
    if "model_type" in data:
        return data["model_type"]
    if "model" in data:
        return data["model"]
    # Infer from filename
    name = filename.lower()
    if "catboost" in name:
        return "catboost"
    if "xgboost" in name or "xgb" in name:
        return "xgboost"
    if "lightgbm" in name or "lgb" in name:
        return "lightgbm"
    if "donchian" in name:
        return "donchian"
    if "regime" in name:
        return "regime"
    if "ensemble" in name:
        return "ensemble"
    return "unknown"


def backfill():
    """Parse all JSON files in results/validation/ and insert into experiment DB."""
    if not RESULTS_DIR.exists():
        logger.warning(f"Results directory {RESULTS_DIR} not found")
        return 0

    db = get_db()
    count = 0

    for json_path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {json_path.name}: {e}")
            continue

        sharpe = _extract_sharpe(data)
        model_type = _extract_model_type(data, json_path.stem)

        # Create a config hash from the filename + key params
        cfg = {"source_file": json_path.name, "model_type": model_type}
        h = config_hash(cfg)

        log_experiment(
            db,
            config_hash=h,
            model_type=model_type,
            approach_family=json_path.stem,
            sharpe=sharpe,
            notes=f"Backfilled from {json_path.name}",
        )
        count += 1
        logger.info(f"  {json_path.name}: {model_type}, Sharpe={sharpe}")

    logger.info(f"\nBackfilled {count} experiments into {db}")
    db.close()
    return count


if __name__ == "__main__":
    n = backfill()
    print(f"\nDone. {n} experiments backfilled.")
