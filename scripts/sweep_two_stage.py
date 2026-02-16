#!/usr/bin/env python3
"""Two-stage hyperparameter sweep with experiment DB integration.

Stage 1 — Screening: Single 80/20 temporal split, ALL configs. ~2 min each.
Stage 2 — Validation: Top 5 configs from Stage 1 get full walk-forward.

Uses:
- sparky.data.loader for holdout-enforced data loading
- sparky.tracking.experiment_db for dedup and logging
- sparky.oversight.timeout for per-config time limits

Usage:
    PYTHONPATH=. python3 scripts/sweep_two_stage.py
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparky.data.loader import load
from sparky.tracking.experiment_db import (
    get_db, config_hash, is_duplicate, log_experiment,
)
from sparky.oversight.timeout import with_timeout, ExperimentTimeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("results/sweep_progress.csv")


def get_sweep_configs() -> list[dict]:
    """Generate hyperparameter configurations to sweep.

    Returns list of config dicts with model_type and hyperparams.
    """
    configs = []

    # XGBoost variants
    for depth in [3, 5, 7]:
        for lr in [0.01, 0.05, 0.1]:
            for n_est in [100, 200]:
                configs.append({
                    "model_type": "xgboost",
                    "hyperparams": {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "tree_method": "hist",
                        "device": "cuda",
                    }
                })

    # CatBoost variants
    for depth in [4, 5, 6]:
        for lr in [0.01, 0.05, 0.1]:
            configs.append({
                "model_type": "catboost",
                "hyperparams": {
                    "depth": depth,
                    "learning_rate": lr,
                    "iterations": 200,
                    "subsample": 0.8,
                    "l2_leaf_reg": 2.0,
                    "task_type": "GPU",
                    "devices": "0",
                    "verbose": 0,
                }
            })

    # LightGBM variants
    for depth in [3, 5, 7]:
        for lr in [0.01, 0.05, 0.1]:
            configs.append({
                "model_type": "lightgbm",
                "hyperparams": {
                    "max_depth": depth,
                    "learning_rate": lr,
                    "n_estimators": 200,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "device": "gpu",
                    "verbose": -1,
                }
            })

    return configs


def create_model(config: dict):
    """Instantiate a model from config dict."""
    model_type = config["model_type"]
    params = config["hyperparams"]

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, random_state=42, eval_metric="logloss")
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, random_seed=42)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@with_timeout(seconds=900)
def run_single_config(config: dict, X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a single config. Timeout after 15 min.

    Returns dict with auc, accuracy, and wall_clock_seconds.
    """
    start = time.time()
    model = create_model(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    accuracy = (y_pred == y_test).mean()
    elapsed = time.time() - start

    return {
        "auc": auc,
        "accuracy": accuracy,
        "wall_clock_seconds": elapsed,
    }


def append_progress(config: dict, result: dict):
    """Append one line to sweep_progress.csv."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PROGRESS_FILE.exists()
    with open(PROGRESS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model_type", "config_hash", "auc", "accuracy", "wall_clock_s", "stage"])
        writer.writerow([
            config["model_type"],
            config.get("_hash", ""),
            f"{result.get('auc', 0):.4f}",
            f"{result.get('accuracy', 0):.4f}",
            f"{result.get('wall_clock_seconds', 0):.1f}",
            result.get("stage", "1"),
        ])


def main():
    """Run two-stage hyperparameter sweep."""
    # Load data ONCE
    logger.info("Loading data via sparky.data.loader...")
    df = load("btc_1h_features", purpose="training")
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns[:10])}...")

    # Separate features and target
    target_col = [c for c in df.columns if "target" in c.lower() or "direction" in c.lower()]
    if not target_col:
        logger.error("No target column found. Expected column with 'target' or 'direction' in name.")
        sys.exit(1)

    target_col = target_col[0]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop non-numeric
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    # 80/20 temporal split for Stage 1
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Stage 1 split: train={len(X_train)}, test={len(X_test)}")

    db = get_db()
    configs = get_sweep_configs()
    logger.info(f"Total configs to sweep: {len(configs)}")

    # Stage 1: Screening
    stage1_results = []
    for i, config in enumerate(configs):
        h = config_hash(config)
        config["_hash"] = h

        if is_duplicate(db, h):
            logger.info(f"  [{i+1}/{len(configs)}] SKIP (duplicate): {config['model_type']} {h}")
            continue

        logger.info(f"  [{i+1}/{len(configs)}] Running: {config['model_type']} {h}")
        try:
            result = run_single_config(config, X_train, y_train, X_test, y_test)
            result["stage"] = "1"
            stage1_results.append((config, result))

            log_experiment(
                db,
                config_hash=h,
                model_type=config["model_type"],
                approach_family="sweep_stage1",
                hyperparams=config["hyperparams"],
                sharpe=None,  # No Sharpe in Stage 1 (just AUC)
                wall_clock_seconds=result["wall_clock_seconds"],
                notes=f"Stage 1 screening: AUC={result['auc']:.4f}",
            )
            append_progress(config, result)
            logger.info(f"    AUC={result['auc']:.4f}, acc={result['accuracy']:.4f}, {result['wall_clock_seconds']:.1f}s")

        except ExperimentTimeout:
            logger.warning(f"    TIMEOUT: {config['model_type']} {h}")
            log_experiment(
                db,
                config_hash=h,
                model_type=config["model_type"],
                approach_family="sweep_stage1",
                notes="TIMEOUT",
            )
        except Exception as e:
            logger.error(f"    ERROR: {e}")

    # Rank by AUC, take top 5 for Stage 2
    stage1_results.sort(key=lambda x: x[1]["auc"], reverse=True)
    top5 = stage1_results[:5]
    logger.info(f"\nStage 1 complete. Top 5 configs for Stage 2:")
    for config, result in top5:
        logger.info(f"  {config['model_type']} {config['_hash']}: AUC={result['auc']:.4f}")

    # Stage 2: Walk-forward validation for top 5
    # TODO: Integrate with WalkForwardBacktester when CEO wires up the pipeline
    logger.info("\nStage 2: Walk-forward validation (scaffold — CEO to implement)")
    logger.info("Top 5 configs saved. Run walk-forward manually or extend this script.")

    # Save summary
    summary = {
        "total_configs": len(configs),
        "stage1_completed": len(stage1_results),
        "top5": [
            {
                "model_type": c["model_type"],
                "config_hash": c["_hash"],
                "auc": r["auc"],
                "hyperparams": c["hyperparams"],
            }
            for c, r in top5
        ],
    }
    summary_path = Path("results/sweep_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    db.close()


if __name__ == "__main__":
    main()
