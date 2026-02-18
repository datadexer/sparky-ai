#!/usr/bin/env python3
"""Regime-aware model training scaffold.

Loads data via sparky.data.loader, classifies market regimes, trains
regime-conditional models, and evaluates with walk-forward backtesting.

All infrastructure (loader, experiment DB, GPU, timeout) is wired in.
Research agent fills in the regime classification and trading rule TODOs.

Usage:
    PYTHONPATH=. python3 scripts/train_regime_aware.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker, config_hash
from sparky.oversight.timeout import with_timeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """Classify each row into a market regime.

    TODO (Research Agent): Implement regime classification. Options:
    - Volatility-based: realized_vol > threshold → "high_vol" else "low_vol"
    - Trend-based: SMA crossover → "bull" / "bear" / "sideways"
    - HMM: Hidden Markov Model with 2-3 states

    Args:
        df: Feature DataFrame with DatetimeIndex.

    Returns:
        Series of regime labels aligned to df.index.
    """
    # Placeholder: volatility-based regime using rolling std of returns
    if "close" in df.columns:
        returns = df["close"].pct_change()
    elif "log_return" in df.columns:
        returns = df["log_return"]
    else:
        # Fallback: use first numeric column
        returns = df.select_dtypes(include=[np.number]).iloc[:, 0].pct_change()

    vol_20d = returns.rolling(20, min_periods=5).std()
    median_vol = vol_20d.median()

    regime = pd.Series("low_vol", index=df.index)
    regime[vol_20d > median_vol * 1.5] = "high_vol"
    regime[vol_20d > median_vol * 2.5] = "crisis"

    return regime


@with_timeout(seconds=900)
def train_regime_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    regime_train: pd.Series,
    regime_label: str,
) -> object:
    """Train a model on data from a specific regime.

    TODO (Research Agent): Choose model type and hyperparams per regime.

    Args:
        X_train: Features for this regime's training data.
        y_train: Labels for this regime's training data.
        regime_train: Regime labels (for filtering).
        regime_label: Which regime to train on.

    Returns:
        Trained model.
    """
    from catboost import CatBoostClassifier

    mask = regime_train == regime_label
    X_regime = X_train[mask]
    y_regime = y_train[mask]

    if len(X_regime) < 50:
        logger.warning(f"Regime '{regime_label}' has only {len(X_regime)} samples — skipping")
        return None

    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=200,
        verbose=0,
        task_type="GPU",
        devices="0",
        random_seed=42,
    )
    model.fit(X_regime, y_regime)
    return model


def main():
    """Run regime-aware training pipeline."""
    start = time.time()

    # Load data via enforced loader
    logger.info("Loading data...")
    df = load("btc_1h_features", purpose="training")
    logger.info(f"Loaded {len(df)} rows")

    # Separate features and target
    target_col = [c for c in df.columns if "target" in c.lower() or "direction" in c.lower()]
    if not target_col:
        logger.error("No target column found")
        sys.exit(1)

    target_col = target_col[0]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    # Classify regimes
    logger.info("Classifying regimes...")
    regimes = classify_regime(df)
    regime_counts = regimes.value_counts()
    logger.info(f"Regime distribution:\n{regime_counts}")

    # Check MLflow for duplicates
    tracker = ExperimentTracker(experiment_name="regime_aware")
    cfg = {"approach": "regime_aware", "regime_method": "volatility_threshold"}
    h = config_hash(cfg)

    if tracker.is_duplicate(h):
        logger.info("This exact config already ran — check MLflow")
        return

    # Train per-regime models
    models = {}
    for regime_label in regimes.unique():
        logger.info(f"Training model for regime: {regime_label}")
        model = train_regime_model(X, y, regimes, regime_label)
        if model is not None:
            models[regime_label] = model

    logger.info(f"Trained {len(models)} regime-specific models")

    # TODO (Research Agent): Walk-forward evaluation
    # Use WalkForwardBacktester with regime-conditional prediction:
    # At each test point, check which regime we're in, then use that regime's model.

    elapsed = time.time() - start

    # Log to MLflow
    tracker.log_experiment(
        name=f"regime_aware_{h}",
        config={**cfg, "model_type": "catboost"},
        metrics={"wall_clock_seconds": elapsed},
    )

    logger.info(f"Done in {elapsed:.1f}s. Walk-forward evaluation pending.")


if __name__ == "__main__":
    main()
