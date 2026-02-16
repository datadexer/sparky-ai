#!/usr/bin/env python3
"""Train CatBoost on Hourly BTC Data (1h-ahead prediction)

Compare CatBoost to XGBoost baseline:
- XGBoost: Val AUC 0.5549, Test AUC 0.5521
- CatBoost: Uses gradient boosting with categorical features support

Data:
- Features: ~114K hourly samples, 23 features
- Target: 1h-ahead binary (close(t+1) > close(t))

Splits (by timestamp):
- Train: 2017-2020
- Validation: 2021-2022
- Test: 2023
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data():
    """Load hourly feature matrix and 1h-ahead targets."""
    features_path = Path("data/processed/features_hourly_full.parquet")
    targets_path = Path("data/processed/targets_hourly_1h.parquet")

    if not features_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {features_path}")

    if not targets_path.exists():
        raise FileNotFoundError(f"Targets not found: {targets_path}")

    logger.info(f"Loading features from {features_path}")
    X = pd.read_parquet(features_path)

    logger.info(f"Loading targets from {targets_path}")
    y = pd.read_parquet(targets_path)["target"]

    # Align by index
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    logger.info(f"Loaded {len(X):,} samples with {X.shape[1]} features")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Date range: {X.index.min()} to {X.index.max()}")
    logger.info(f"Target balance: {y.mean():.2%} positive")

    return X, y


def clean_data(X: pd.DataFrame):
    """Replace inf values with NaN and drop NaN rows.

    Args:
        X: Feature DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data (replacing inf with NaN)...")

    # Replace inf with NaN
    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # Count NaNs
    nan_counts = X_clean.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN counts after inf replacement:\n{nan_counts[nan_counts > 0]}")

    # Drop rows with NaN
    before_count = len(X_clean)
    X_clean = X_clean.dropna()
    after_count = len(X_clean)

    if before_count > after_count:
        logger.info(f"Dropped {before_count - after_count:,} NaN rows ({(before_count - after_count) / before_count:.1%})")

    return X_clean


def create_splits(X: pd.DataFrame, y: pd.Series):
    """Create train/val/test splits by timestamp.

    Splits:
    - Train: 2017-01-01 to 2020-12-31
    - Validation: 2021-01-01 to 2022-12-31
    - Test: 2023-01-01 to 2023-12-31

    Returns:
        dict with train/val/test DataFrames
    """
    logger.info("Creating train/val/test splits...")

    splits = {}

    # Train: 2017-2020
    train_mask = (X.index >= "2017-01-01") & (X.index < "2021-01-01")
    splits["X_train"] = X[train_mask]
    splits["y_train"] = y[train_mask]
    logger.info(
        f"Train: {splits['X_train'].index.min().date()} to {splits['X_train'].index.max().date()} "
        f"({len(splits['X_train']):,} samples)"
    )

    # Validation: 2021-2022
    val_mask = (X.index >= "2021-01-01") & (X.index < "2023-01-01")
    splits["X_val"] = X[val_mask]
    splits["y_val"] = y[val_mask]
    logger.info(
        f"Validation: {splits['X_val'].index.min().date()} to {splits['X_val'].index.max().date()} "
        f"({len(splits['X_val']):,} samples)"
    )

    # Test: 2023
    test_mask = (X.index >= "2023-01-01") & (X.index < "2024-01-01")
    splits["X_test"] = X[test_mask]
    splits["y_test"] = y[test_mask]
    logger.info(
        f"Test: {splits['X_test'].index.min().date()} to {splits['X_test'].index.max().date()} "
        f"({len(splits['X_test']):,} samples)"
    )

    return splits


def train_catboost(X_train, y_train):
    """Train CatBoost model.

    Hyperparameters chosen to match XGBoost complexity:
    - depth=5 (XGBoost max_depth=3, but CatBoost uses symmetric trees)
    - learning_rate=0.05 (same as XGBoost)
    - iterations=200 (XGBoost n_estimators=100, but CatBoost is more aggressive)
    - subsample=0.8 (same as XGBoost)
    - l2_leaf_reg=2.0 (similar to XGBoost reg_lambda)

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained CatBoostClassifier
    """
    logger.info("Training CatBoost model...")

    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=200,
        subsample=0.8,
        l2_leaf_reg=2.0,
        random_seed=42,
        verbose=0,
        task_type="GPU",
        devices="0",
    )

    model.fit(X_train, y_train)
    logger.info("Model training complete")

    return model


def evaluate_model(model, X, y, split_name: str):
    """Evaluate model on a dataset.

    Args:
        model: Trained CatBoost model
        X: Features
        y: Targets
        split_name: Name of split (for logging)

    Returns:
        dict with evaluation metrics
    """
    logger.info(f"Evaluating on {split_name}...")

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    metrics = {
        f"{split_name}_accuracy": accuracy,
        f"{split_name}_roc_auc": roc_auc,
        f"{split_name}_precision": precision,
        f"{split_name}_recall": recall,
        f"{split_name}_f1": f1,
    }

    logger.info(
        f"{split_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return metrics


def get_feature_importances(model, feature_names):
    """Extract feature importances from CatBoost model.

    Args:
        model: Trained CatBoost model
        feature_names: List of feature names

    Returns:
        dict mapping feature names to importance scores (top 10)
    """
    importances = model.get_feature_importance()

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Get top 10
    top_10 = importance_df.head(10)

    logger.info("\nTop 10 Feature Importances:")
    for idx, row in top_10.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return dict(zip(top_10['feature'].tolist(), top_10['importance'].tolist()))


def save_results(metrics: dict, feature_importances: dict, n_train_samples: int):
    """Save results to JSON file.

    Args:
        metrics: Dictionary of all metrics
        feature_importances: Dictionary of feature importances
        n_train_samples: Number of training samples
    """
    results_dir = Path("results/model_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "catboost_1h_results.json"

    results = {
        "model": "CatBoost",
        "target": "1h-ahead",
        "training_samples": n_train_samples,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "feature_importances": feature_importances,
        "xgboost_baseline": {
            "val_accuracy": 0.5416,
            "val_roc_auc": 0.5549,
            "test_accuracy": 0.5357,
            "test_roc_auc": 0.5521,
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")


def print_comparison_table(metrics: dict):
    """Print comparison table between XGBoost and CatBoost.

    Args:
        metrics: Dictionary of CatBoost metrics
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: XGBoost vs CatBoost")
    print("=" * 80)
    print("\n| Metric        | XGBoost | CatBoost |")
    print("|---------------|---------|----------|")
    print(f"| Val Accuracy  | 0.5416  | {metrics['validation_accuracy']:.4f}   |")
    print(f"| Val ROC-AUC   | 0.5549  | {metrics['validation_roc_auc']:.4f}   |")
    print(f"| Test Accuracy | 0.5357  | {metrics['test_accuracy']:.4f}   |")
    print(f"| Test ROC-AUC  | 0.5521  | {metrics['test_roc_auc']:.4f}   |")
    print("\n" + "=" * 80)

    # Compute deltas
    val_auc_delta = metrics['validation_roc_auc'] - 0.5549
    test_auc_delta = metrics['test_roc_auc'] - 0.5521

    print("\nDELTA FROM XGBOOST:")
    print(f"  Validation AUC: {val_auc_delta:+.4f} ({val_auc_delta / 0.5549 * 100:+.2f}%)")
    print(f"  Test AUC: {test_auc_delta:+.4f} ({test_auc_delta / 0.5521 * 100:+.2f}%)")
    print("=" * 80 + "\n")


def main():
    """Main execution: train CatBoost and compare to XGBoost."""

    logger.info("=" * 80)
    logger.info("TRAIN CATBOOST ON HOURLY BTC DATA (1h-ahead prediction)")
    logger.info("=" * 80)

    # Load data
    X, y = load_data()

    # Clean data (replace inf with NaN, drop NaN rows)
    X = clean_data(X)
    y = y.loc[X.index]  # Align after cleaning

    # Create splits
    splits = create_splits(X, y)

    # Train model
    model = train_catboost(splits["X_train"], splits["y_train"])

    # Evaluate on all splits
    train_metrics = evaluate_model(model, splits["X_train"], splits["y_train"], "train")
    val_metrics = evaluate_model(model, splits["X_val"], splits["y_val"], "validation")
    test_metrics = evaluate_model(model, splits["X_test"], splits["y_test"], "test")

    # Combine all metrics
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    # Get feature importances
    feature_importances = get_feature_importances(model, list(X.columns))

    # Save results
    save_results(all_metrics, feature_importances, len(splits["X_train"]))

    # Print comparison table
    print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()
