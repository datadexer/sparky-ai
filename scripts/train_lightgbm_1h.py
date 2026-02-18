#!/usr/bin/env python3
"""Train LightGBM on 1-hour ahead BTC prediction

Compare LightGBM vs XGBoost on the same hourly feature set.

Task: Predict close(t+1h) > close(t) using 1h-resolution features.

Data:
- Features: data/processed/features_hourly_full.parquet (~114K rows × 23 features)
- Target: data/processed/targets_hourly_1h.parquet (1h-ahead binary)

Splits (by timestamp):
- Train: 2017-2020
- Validation: 2021-2022
- Test: 2023

Baseline (XGBoost):
- Val: 0.5549 ROC-AUC, 0.5416 accuracy
- Test: 0.5521 ROC-AUC, 0.5357 accuracy
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data():
    """Load hourly features and 1h-ahead targets."""
    features_path = Path("data/processed/features_hourly_full.parquet")
    targets_path = Path("data/processed/targets_hourly_1h.parquet")

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}\nRun scripts/prepare_hourly_features.py first")

    if not targets_path.exists():
        raise FileNotFoundError(f"Targets not found: {targets_path}\nRun scripts/prepare_hourly_features.py first")

    logger.info(f"Loading features from {features_path}")
    X = pd.read_parquet(features_path)

    logger.info(f"Loading targets from {targets_path}")
    y = pd.read_parquet(targets_path)["target"]

    # Align by index
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    logger.info(f"Loaded {len(X):,} samples with {X.shape[1]} features")
    logger.info(f"Date range: {X.index.min()} to {X.index.max()}")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    dropped = len(X) - len(X_clean)
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} rows with NaN/inf ({dropped / len(X) * 100:.1f}%)")

    return X_clean, y_clean


def create_splits(X: pd.DataFrame, y: pd.Series):
    """Create train/val/test splits by timestamp.

    Splits:
    - Train: 2017-2020
    - Validation: 2021-2022
    - Test: 2023
    """
    logger.info("Creating time-based splits...")

    splits = {}

    # Train: 2017-2020
    train_mask = (X.index >= "2017-01-01") & (X.index < "2021-01-01")
    splits["X_train"] = X[train_mask]
    splits["y_train"] = y[train_mask]
    logger.info(
        f"Train: {splits['X_train'].index.min().date()} to "
        f"{splits['X_train'].index.max().date()} ({len(splits['X_train']):,} samples)"
    )

    # Validation: 2021-2022
    val_mask = (X.index >= "2021-01-01") & (X.index < "2023-01-01")
    splits["X_val"] = X[val_mask]
    splits["y_val"] = y[val_mask]
    logger.info(
        f"Validation: {splits['X_val'].index.min().date()} to "
        f"{splits['X_val'].index.max().date()} ({len(splits['X_val']):,} samples)"
    )

    # Test: 2023
    test_mask = (X.index >= "2023-01-01") & (X.index < "2024-01-01")
    splits["X_test"] = X[test_mask]
    splits["y_test"] = y[test_mask]
    logger.info(
        f"Test: {splits['X_test'].index.min().date()} to "
        f"{splits['X_test'].index.max().date()} ({len(splits['X_test']):,} samples)"
    )

    return splits


def train_lightgbm(X_train, y_train):
    """Train LightGBM classifier.

    Hyperparameters chosen to match XGBoost complexity:
    - max_depth=5 (XGBoost used 3, but LightGBM can handle slightly deeper)
    - learning_rate=0.05 (same as XGBoost)
    - n_estimators=200 (XGBoost used 100, we use more with regularization)
    - L1/L2 regularization to prevent overfitting
    """
    logger.info("Training LightGBM...")

    params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.5,  # L1 regularization
        "reg_lambda": 2.0,  # L2 regularization
        "random_state": 42,
        "verbose": -1,
        "device": "gpu",
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Training complete")
    return model, params


def evaluate_model(model, X, y, split_name: str):
    """Compute classification metrics.

    Returns:
        dict with accuracy, ROC-AUC, precision, recall, F1
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        f"{split_name}_accuracy": accuracy_score(y, y_pred),
        f"{split_name}_roc_auc": roc_auc_score(y, y_prob),
        f"{split_name}_precision": precision_score(y, y_pred, zero_division=0),
        f"{split_name}_recall": recall_score(y, y_pred, zero_division=0),
        f"{split_name}_f1": f1_score(y, y_pred, zero_division=0),
    }

    logger.info(
        f"{split_name} - Accuracy: {metrics[f'{split_name}_accuracy']:.4f}, "
        f"ROC-AUC: {metrics[f'{split_name}_roc_auc']:.4f}"
    )

    return metrics


def get_feature_importances(model, feature_names, top_n=10):
    """Extract top N feature importances."""
    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    logger.info(f"\nTop {top_n} features:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df.to_dict("records")


def save_results(results: dict, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def print_comparison_table(lgb_metrics):
    """Print comparison table between XGBoost and LightGBM."""
    # XGBoost baseline from the task description
    xgb_metrics = {
        "val_accuracy": 0.5416,
        "val_roc_auc": 0.5549,
        "test_accuracy": 0.5357,
        "test_roc_auc": 0.5521,
    }

    print("\n" + "=" * 80)
    print("MODEL COMPARISON: XGBoost vs LightGBM")
    print("=" * 80)
    print(f"{'Metric':<20} {'XGBoost':<12} {'LightGBM':<12} {'Delta':<10}")
    print("-" * 80)

    metrics_to_compare = [
        ("Val Accuracy", "val_accuracy"),
        ("Val ROC-AUC", "val_roc_auc"),
        ("Test Accuracy", "test_accuracy"),
        ("Test ROC-AUC", "test_roc_auc"),
    ]

    for label, key in metrics_to_compare:
        xgb_val = xgb_metrics[key]
        lgb_val = lgb_metrics[key]
        delta = lgb_val - xgb_val
        delta_str = f"{delta:+.4f}"

        print(f"{label:<20} {xgb_val:<12.4f} {lgb_val:<12.4f} {delta_str:<10}")

    print("=" * 80)


def main():
    """Main execution: train LightGBM and compare to XGBoost."""

    logger.info("=" * 80)
    logger.info("TRAIN LIGHTGBM ON 1-HOUR AHEAD BTC PREDICTION")
    logger.info("=" * 80)

    # Load data
    X, y = load_data()

    # Create splits
    splits = create_splits(X, y)

    # Train model
    model, params = train_lightgbm(splits["X_train"], splits["y_train"])

    # Evaluate on all splits
    train_metrics = evaluate_model(model, splits["X_train"], splits["y_train"], "train")
    val_metrics = evaluate_model(model, splits["X_val"], splits["y_val"], "val")
    test_metrics = evaluate_model(model, splits["X_test"], splits["y_test"], "test")

    # Feature importances
    feature_importances = get_feature_importances(model, list(X.columns), top_n=10)

    # Compile results
    results = {
        "model": "LightGBM",
        "task": "1h-ahead binary classification (close(t+1h) > close(t))",
        "train_samples": len(splits["X_train"]),
        "val_samples": len(splits["X_val"]),
        "test_samples": len(splits["X_test"]),
        "num_features": X.shape[1],
        "hyperparameters": params,
        "metrics": {
            **train_metrics,
            **val_metrics,
            **test_metrics,
        },
        "feature_importances": feature_importances,
        "baseline_xgboost": {
            "val_accuracy": 0.5416,
            "val_roc_auc": 0.5549,
            "test_accuracy": 0.5357,
            "test_roc_auc": 0.5521,
        },
    }

    # Save results
    output_path = Path("results/model_comparison/lightgbm_1h_results.json")
    save_results(results, output_path)

    # Print comparison table
    print_comparison_table(results["metrics"])

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"LightGBM Val ROC-AUC: {val_metrics['val_roc_auc']:.4f}")
    logger.info("XGBoost Val ROC-AUC:  0.5549")
    logger.info(f"Delta: {val_metrics['val_roc_auc'] - 0.5549:+.4f}")
    logger.info("")
    logger.info(f"LightGBM Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
    logger.info("XGBoost Test ROC-AUC:  0.5521")
    logger.info(f"Delta: {test_metrics['test_roc_auc'] - 0.5521:+.4f}")


if __name__ == "__main__":
    main()
