#!/usr/bin/env python3
"""Train XGBoost on Hourly BTC Data

Test APPROACH 1: Higher frequency data (52K hourly samples).

Strategy:
- Train on hourly-derived features (RSI-14h, Momentum-30h, etc.)
- Predict daily direction
- Use proper validation split (no data snooping)

Splits:
- Train: 2017-2020 (hourly features)
- Validation: 2021-2022 (for model selection)
- Test: 2023 (for hyperparameter tuning)
- Holdout: 2024-2025 (FINAL test, ONE run only)

Success criteria:
- Holdout Sharpe >= 0.5 (positive alpha)
- Train-holdout gap < 0.3 (good generalization)
- Leakage detector passes
"""

import logging
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data():
    """Load hourly feature matrix and targets."""
    features_path = Path("data/processed/feature_matrix_btc_hourly.parquet")
    targets_path = Path("data/processed/targets_btc_hourly_1d.parquet")

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {features_path}\nRun scripts/prepare_hourly_features.py first"
        )

    if not targets_path.exists():
        raise FileNotFoundError(f"Targets not found: {targets_path}\nRun scripts/prepare_hourly_features.py first")

    logger.info(f"Loading features from {features_path}")
    X = pd.read_parquet(features_path)

    logger.info(f"Loading targets from {targets_path}")
    y = pd.read_parquet(targets_path)["target"]

    logger.info(f"Loaded {len(X):,} samples with {X.shape[1]} features")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Date range: {X.index.min()} to {X.index.max()}")

    return X, y


def create_splits(X: pd.DataFrame, y: pd.Series):
    """Create train/val/test/holdout splits (no data snooping).

    Splits:
    - Train: 2017-2020 (3 years)
    - Validation: 2021-2022 (2 years) - for model selection
    - Test: 2023 (1 year) - for hyperparameter tuning
    - Holdout: 2024-2025 (2 years) - FINAL test, ONE run only

    Returns:
        dict with train/val/test/holdout DataFrames
    """
    logger.info("Creating train/val/test/holdout splits...")

    splits = {}

    # Train: 2017-2020
    train_mask = (X.index >= "2017-01-01") & (X.index < "2021-01-01")
    splits["X_train"] = X[train_mask]
    splits["y_train"] = y[train_mask]
    logger.info(
        f"Train: {splits['X_train'].index.min().date()} to {splits['X_train'].index.max().date()} ({len(splits['X_train']):,} samples)"
    )

    # Validation: 2021-2022
    val_mask = (X.index >= "2021-01-01") & (X.index < "2023-01-01")
    splits["X_val"] = X[val_mask]
    splits["y_val"] = y[val_mask]
    logger.info(
        f"Validation: {splits['X_val'].index.min().date()} to {splits['X_val'].index.max().date()} ({len(splits['X_val']):,} samples)"
    )

    # Test: 2023
    test_mask = (X.index >= "2023-01-01") & (X.index < "2024-01-01")
    splits["X_test"] = X[test_mask]
    splits["y_test"] = y[test_mask]
    logger.info(
        f"Test: {splits['X_test'].index.min().date()} to {splits['X_test'].index.max().date()} ({len(splits['X_test']):,} samples)"
    )

    # Holdout: 2024-2025 (NEVER touch until final validation)
    holdout_mask = (X.index >= "2024-01-01") & (X.index < "2026-01-01")
    splits["X_holdout"] = X[holdout_mask]
    splits["y_holdout"] = y[holdout_mask]
    logger.info(
        f"Holdout: {splits['X_holdout'].index.min().date()} to {splits['X_holdout'].index.max().date()} ({len(splits['X_holdout']):,} samples)"
    )

    return splits


def train_model(X_train, y_train):
    """Train XGBoost model on training set.

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained XGBoost model
    """
    from sparky.models.xgboost_model import XGBoostModel

    logger.info("Training XGBoost model...")

    # Use conservative hyperparameters (prevent overfitting)
    model = XGBoostModel(
        max_depth=3,  # Shallow trees
        learning_rate=0.05,  # Slow learning
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=2.0,  # L2 regularization
        random_state=42,
    )

    model.fit(X_train, y_train)
    logger.info("Model training complete")

    return model


def evaluate_model(model, X, y, split_name: str):
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        X: Features
        y: Targets
        split_name: Name of split (for logging)

    Returns:
        dict with evaluation metrics
    """
    logger.info(f"Evaluating on {split_name}...")

    predictions = model.predict(X)

    # Basic metrics
    accuracy = (predictions == y).mean()
    positive_rate = predictions.mean()
    target_balance = y.mean()

    metrics = {
        f"{split_name}_accuracy": accuracy,
        f"{split_name}_positive_rate": positive_rate,
        f"{split_name}_target_balance": target_balance,
    }

    logger.info(
        f"{split_name} - Accuracy: {accuracy:.3f}, Positive rate: {positive_rate:.1%}, Target balance: {target_balance:.1%}"
    )

    return metrics, predictions


def run_leakage_detection(model, X_train, y_train, X_test, y_test):
    """Run leakage detector to verify no data snooping.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets

    Returns:
        bool: True if all checks passed, False otherwise
    """
    from sparky.backtest.leakage_detector import LeakageDetector

    logger.info("Running leakage detection (n_trials=10)...")

    detector = LeakageDetector(n_shuffle_trials=10)
    report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

    logger.info(f"Leakage detection: {'PASSED' if report.passed else 'FAILED'}")

    if not report.passed:
        logger.error("LEAKAGE DETECTED - Results are INVALID")
        for check in report.checks:
            if not check.passed:
                logger.error(f"  Failed: {check.check_name} - {check.message}")

    return report.passed


def main():
    """Main execution: train on hourly data and validate."""

    logger.info("=" * 80)
    logger.info("TRAIN ON HOURLY BTC DATA (APPROACH 1)")
    logger.info("=" * 80)

    # Load data
    X, y = load_data()

    # Create splits
    splits = create_splits(X, y)

    # Remove NaNs from training set
    train_valid_mask = ~splits["X_train"].isna().any(axis=1)
    X_train_clean = splits["X_train"][train_valid_mask]
    y_train_clean = splits["y_train"][train_valid_mask]
    logger.info(f"Removed {(~train_valid_mask).sum()} NaN rows from training, {len(X_train_clean):,} remain")

    # Train model
    model = train_model(X_train_clean, y_train_clean)

    # Evaluate on train
    train_metrics, _ = evaluate_model(model, X_train_clean, y_train_clean, "train")

    # Evaluate on validation (can use for model selection)
    val_valid_mask = ~splits["X_val"].isna().any(axis=1)
    X_val_clean = splits["X_val"][val_valid_mask]
    y_val_clean = splits["y_val"][val_valid_mask]
    val_metrics, _ = evaluate_model(model, X_val_clean, y_val_clean, "validation")

    # Evaluate on test (can use for hyperparameter tuning)
    test_valid_mask = ~splits["X_test"].isna().any(axis=1)
    X_test_clean = splits["X_test"][test_valid_mask]
    y_test_clean = splits["y_test"][test_valid_mask]
    test_metrics, _ = evaluate_model(model, X_test_clean, y_test_clean, "test")

    # Leakage detection (use train vs test)
    leakage_passed = run_leakage_detection(model, X_train_clean, y_train_clean, X_test_clean, y_test_clean)

    # Log to MLflow
    mlflow.set_experiment("phase3_hourly_data")
    with mlflow.start_run(run_name="xgboost_hourly_btc"):
        # Log parameters
        mlflow.log_param("approach", "APPROACH_1_hourly_data")
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("data_frequency", "hourly")
        mlflow.log_param("train_samples", len(X_train_clean))
        mlflow.log_param("features", list(X.columns))

        # Log metrics
        for key, value in {**train_metrics, **val_metrics, **test_metrics}.items():
            mlflow.log_metric(key, value)

        mlflow.log_metric("leakage_passed", 1.0 if leakage_passed else 0.0)

        # Log feature importance
        importance_df = model.get_feature_importances()
        for _, row in importance_df.iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", float(row["importance"]))

        logger.info(f"Logged to MLflow run: {mlflow.active_run().info.run_id}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Train accuracy: {train_metrics['train_accuracy']:.3f}")
    logger.info(f"Validation accuracy: {val_metrics['validation_accuracy']:.3f}")
    logger.info(f"Test accuracy: {test_metrics['test_accuracy']:.3f}")
    logger.info(f"Leakage detection: {'PASSED ✓' if leakage_passed else 'FAILED ✗'}")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    if leakage_passed:
        if test_metrics["test_accuracy"] > 0.52:
            logger.info("✓ Test accuracy > 52% (better than random)")
            logger.info("  Next: Run holdout validation (ONE test only)")
            logger.info("  Script: scripts/validate_holdout_hourly.py")
        else:
            logger.info("✗ Test accuracy ≈ random (no predictive power)")
            logger.info("  Recommendation: Try APPROACH 2 (cross-asset training)")
    else:
        logger.info("✗ LEAKAGE DETECTED - Results invalid")
        logger.info("  Action: Debug feature generation, check for look-ahead bias")


if __name__ == "__main__":
    main()
