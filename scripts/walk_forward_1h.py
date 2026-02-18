#!/usr/bin/env python3
"""
Walk-forward validation for 1h-ahead BTC prediction model.

This script performs expanding-window walk-forward cross-validation on hourly BTC data
to test temporal stability and detect overfitting to specific train/val splits.

Design:
- Expanding window (train on all historical data)
- 6-month test folds
- 9 folds from 2019-07 to 2023-12
- CatBoost classifier with same hyperparameters as single-split validation
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def load_data():
    """Load and align features and targets."""
    logger.info("Loading hourly features and targets...")

    features = pd.read_parquet(project_root / "data/processed/features_hourly_full.parquet")
    targets = pd.read_parquet(project_root / "data/processed/targets_hourly_1h.parquet")

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Targets shape: {targets.shape}")

    # Drop NaN
    features = features.dropna()
    targets = targets.loc[features.index].dropna()

    # Align
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    logger.info(f"After alignment: {len(common_idx)} samples")
    logger.info(f"Date range: {common_idx.min()} to {common_idx.max()}")

    return features, targets


def define_walk_forward_folds():
    """
    Define walk-forward folds (expanding window, 6-month test periods).

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    folds = [
        # Fold 1: Train 2017-01 to 2019-06 → Test 2019-07 to 2019-12
        (None, "2019-07-01", "2019-07-01", "2020-01-01"),
        # Fold 2: Train 2017-01 to 2019-12 → Test 2020-01 to 2020-06
        (None, "2020-01-01", "2020-01-01", "2020-07-01"),
        # Fold 3: Train 2017-01 to 2020-06 → Test 2020-07 to 2020-12
        (None, "2020-07-01", "2020-07-01", "2021-01-01"),
        # Fold 4: Train 2017-01 to 2020-12 → Test 2021-01 to 2021-06
        (None, "2021-01-01", "2021-01-01", "2021-07-01"),
        # Fold 5: Train 2017-01 to 2021-06 → Test 2021-07 to 2021-12
        (None, "2021-07-01", "2021-07-01", "2022-01-01"),
        # Fold 6: Train 2017-01 to 2021-12 → Test 2022-01 to 2022-06
        (None, "2022-01-01", "2022-01-01", "2022-07-01"),
        # Fold 7: Train 2017-01 to 2022-06 → Test 2022-07 to 2022-12
        (None, "2022-07-01", "2022-07-01", "2023-01-01"),
        # Fold 8: Train 2017-01 to 2022-12 → Test 2023-01 to 2023-06
        (None, "2023-01-01", "2023-01-01", "2023-07-01"),
        # Fold 9: Train 2017-01 to 2023-06 → Test 2023-07 to 2023-12
        (None, "2023-07-01", "2023-07-01", "2024-01-01"),
    ]

    return folds


def create_fold_splits(features, targets, train_start, train_end, test_start, test_end):
    """
    Create train/test splits for a single fold.

    Args:
        features: Feature DataFrame with DatetimeIndex (UTC)
        targets: Target DataFrame with DatetimeIndex (UTC)
        train_start: Training start date (None = beginning of data)
        train_end: Training end date (exclusive)
        test_start: Test start date (inclusive)
        test_end: Test end date (exclusive)

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Convert dates to UTC timestamps
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts = pd.Timestamp(test_end, tz="UTC")

    # Create masks
    if train_start is None:
        train_mask = features.index < train_end_ts
    else:
        train_start_ts = pd.Timestamp(train_start, tz="UTC")
        train_mask = (features.index >= train_start_ts) & (features.index < train_end_ts)

    test_mask = (features.index >= test_start_ts) & (features.index < test_end_ts)

    # Split data
    X_train = features.loc[train_mask]
    y_train = targets.loc[train_mask, "target"]
    X_test = features.loc[test_mask]
    y_test = targets.loc[test_mask, "target"]

    return X_train, y_train, X_test, y_test


def train_catboost(X_train, y_train):
    """
    Train CatBoost classifier with fixed hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained CatBoost model
    """
    model = CatBoostClassifier(
        depth=5, learning_rate=0.05, iterations=200, l2_leaf_reg=3.0, random_seed=42, verbose=0, subsample=0.8, rsm=0.8
    )

    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    """Run walk-forward validation."""
    logger.info("=" * 80)
    logger.info("Walk-Forward Validation for 1h-ahead BTC Prediction")
    logger.info("=" * 80)

    # Load data
    features, targets = load_data()

    # Define folds
    folds = define_walk_forward_folds()
    logger.info(f"\nDefined {len(folds)} walk-forward folds")

    # Run walk-forward validation
    results = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Fold {fold_idx}/{len(folds)}")
        logger.info(f"Train: {train_start or 'start'} to {train_end}")
        logger.info(f"Test:  {test_start} to {test_end}")
        logger.info(f"{'=' * 80}")

        # Create splits
        X_train, y_train, X_test, y_test = create_fold_splits(
            features, targets, train_start, train_end, test_start, test_end
        )

        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Test samples:  {len(X_test)}")
        logger.info(f"Train balance: {y_train.mean():.3f}")
        logger.info(f"Test balance:  {y_test.mean():.3f}")

        if len(X_test) == 0:
            logger.warning(f"Fold {fold_idx} has no test samples, skipping...")
            continue

        # Train model
        logger.info("Training CatBoost...")
        model = train_catboost(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)

        logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {metrics['recall']:.4f}")
        logger.info(f"Test F1:        {metrics['f1']:.4f}")

        # Store results
        results.append(
            {
                "fold": fold_idx,
                "train_start": train_start or "start",
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_balance": float(y_train.mean()),
                "test_balance": float(y_test.mean()),
                "metrics": metrics,
            }
        )

    # Aggregate statistics
    logger.info(f"\n{'=' * 80}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'=' * 80}")

    # Extract AUC scores
    auc_scores = [r["metrics"]["roc_auc"] for r in results]

    logger.info("\nROC-AUC by fold:")
    for r in results:
        logger.info(f"  Fold {r['fold']} ({r['test_start']} to {r['test_end']}): {r['metrics']['roc_auc']:.4f}")

    logger.info("\nAUC Statistics:")
    logger.info(f"  Mean:   {np.mean(auc_scores):.4f}")
    logger.info(f"  Std:    {np.std(auc_scores):.4f}")
    logger.info(f"  Min:    {np.min(auc_scores):.4f}")
    logger.info(f"  Max:    {np.max(auc_scores):.4f}")
    logger.info(f"  Median: {np.median(auc_scores):.4f}")

    # Check for temporal degradation
    logger.info("\nTemporal stability check:")
    early_folds = auc_scores[:3]
    late_folds = auc_scores[-3:]
    logger.info(f"  Early folds (1-3) mean AUC: {np.mean(early_folds):.4f}")
    logger.info(f"  Late folds (7-9) mean AUC:  {np.mean(late_folds):.4f}")
    logger.info(f"  Difference:                 {np.mean(late_folds) - np.mean(early_folds):.4f}")

    # Validation check
    logger.info("\nValidation check:")
    reference_auc = 0.557
    mean_auc = np.mean(auc_scores)
    diff = abs(mean_auc - reference_auc)
    logger.info(f"  Reference single-split AUC: {reference_auc:.4f}")
    logger.info(f"  Walk-forward mean AUC:      {mean_auc:.4f}")
    logger.info(f"  Absolute difference:        {diff:.4f}")

    if diff > 0.02:
        logger.warning("  ⚠️  Walk-forward AUC differs by >2% — potential overfitting!")
    else:
        logger.info("  ✓ Walk-forward AUC within 2% — model is stable")

    # Save results
    output_dir = project_root / "results/walk_forward"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "walk_forward_results.json"

    summary = {
        "model": "CatBoost",
        "target": "1h-ahead",
        "folds": results,
        "aggregate_statistics": {
            "mean_auc": float(np.mean(auc_scores)),
            "std_auc": float(np.std(auc_scores)),
            "min_auc": float(np.min(auc_scores)),
            "max_auc": float(np.max(auc_scores)),
            "median_auc": float(np.median(auc_scores)),
        },
        "temporal_stability": {
            "early_folds_mean": float(np.mean(early_folds)),
            "late_folds_mean": float(np.mean(late_folds)),
            "difference": float(np.mean(late_folds) - np.mean(early_folds)),
        },
        "validation_check": {
            "reference_auc": reference_auc,
            "mean_auc": float(mean_auc),
            "difference": float(diff),
            "passes": bool(diff <= 0.02),
        },
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Print summary table
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Fold':<6} {'Test Period':<30} {'Train N':<10} {'Test N':<10} {'AUC':<10}")
    logger.info("-" * 80)
    for r in results:
        period = f"{r['test_start']} to {r['test_end']}"
        logger.info(
            f"{r['fold']:<6} {period:<30} {r['train_samples']:<10} "
            f"{r['test_samples']:<10} {r['metrics']['roc_auc']:<10.4f}"
        )
    logger.info("-" * 80)
    logger.info(f"{'Mean':<6} {'':<30} {'':<10} {'':<10} {np.mean(auc_scores):<10.4f}")
    logger.info(f"{'Std':<6} {'':<30} {'':<10} {'':<10} {np.std(auc_scores):<10.4f}")

    logger.info(f"\n{'=' * 80}")
    logger.info("Walk-forward validation complete!")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
