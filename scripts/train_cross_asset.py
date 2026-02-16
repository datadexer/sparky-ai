#!/usr/bin/env python3
"""Train XGBoost on Cross-Asset Pooled Data

Test APPROACH 2: Cross-asset training (490K pooled hourly samples).

Strategy:
- Train on 7 assets pooled together (BTC, ETH, SOL, ADA, DOT, MATIC, AVAX)
- Use asset_id as categorical feature
- Test ONLY on BTC 2024-2025 holdout
- Model learns universal crypto dynamics

Splits:
- Train: All assets 2017-2023 (~350K samples)
- Validation: All assets 2024 (for hyperparameter tuning)
- Holdout: BTC 2025 ONLY (FINAL test, ONE run only)

Success criteria:
- BTC holdout Sharpe >= 0.5
- Cross-asset model >= BTC-only model
"""

import logging
from pathlib import Path

import pandas as pd
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_cross_asset_data():
    """Load cross-asset pooled feature matrix."""
    features_path = Path("data/processed/feature_matrix_cross_asset_hourly.parquet")
    targets_path = Path("data/processed/targets_cross_asset_hourly_1d.parquet")

    if not features_path.exists():
        raise FileNotFoundError(
            f"Cross-asset features not found: {features_path}\n"
            "Run scripts/prepare_cross_asset_features.py first"
        )

    logger.info(f"Loading cross-asset features from {features_path}")
    X = pd.read_parquet(features_path)

    logger.info(f"Loading cross-asset targets from {targets_path}")
    y = pd.read_parquet(targets_path)["target"]

    logger.info(f"Loaded {len(X):,} pooled samples with {X.shape[1]} features")
    logger.info(f"Assets: {X['asset_name'].unique()}")
    logger.info(f"Samples per asset:\n{X['asset_name'].value_counts()}")

    return X, y


def create_cross_asset_splits(X: pd.DataFrame, y: pd.Series):
    """Create train/val/holdout splits for cross-asset training.

    Splits:
    - Train: All assets 2017-2023
    - Validation: All assets 2024 (for hyperparameter tuning)
    - Holdout: BTC 2025 ONLY (final test)

    Returns:
        dict with train/val/holdout DataFrames
    """
    logger.info("Creating cross-asset splits...")

    splits = {}

    # Train: All assets 2017-2023
    train_mask = (X.index >= "2017-01-01") & (X.index < "2024-01-01")
    splits["X_train"] = X[train_mask]
    splits["y_train"] = y[train_mask]
    logger.info(f"Train (all assets): {splits['X_train'].index.min().date()} to {splits['X_train'].index.max().date()} ({len(splits['X_train']):,} samples)")

    # Validation: All assets 2024
    val_mask = (X.index >= "2024-01-01") & (X.index < "2025-01-01")
    splits["X_val"] = X[val_mask]
    splits["y_val"] = y[val_mask]
    logger.info(f"Validation (all assets): {splits['X_val'].index.min().date()} to {splits['X_val'].index.max().date()} ({len(splits['X_val']):,} samples)")

    # Holdout: BTC 2025 ONLY
    holdout_mask = (X.index >= "2025-01-01") & (X.index < "2026-01-01") & (X["asset_name"] == "btc")
    splits["X_holdout"] = X[holdout_mask]
    splits["y_holdout"] = y[holdout_mask]
    logger.info(f"Holdout (BTC only): {splits['X_holdout'].index.min().date()} to {splits['X_holdout'].index.max().date()} ({len(splits['X_holdout']):,} samples)")

    # Asset distribution in each split
    logger.info(f"\nTrain asset distribution:\n{splits['X_train']['asset_name'].value_counts()}")
    logger.info(f"\nValidation asset distribution:\n{splits['X_val']['asset_name'].value_counts()}")

    return splits


def train_cross_asset_model(X_train, y_train):
    """Train XGBoost on pooled cross-asset data.

    Args:
        X_train: Pooled training features (all assets)
        y_train: Pooled training targets

    Returns:
        Trained XGBoost model
    """
    from sparky.models.xgboost_model import XGBoostModel

    logger.info("Training XGBoost on pooled cross-asset data...")

    # Prepare features: one-hot encode asset_id
    X_train_encoded = X_train.copy()

    # One-hot encode asset_id (categorical feature)
    asset_dummies = pd.get_dummies(X_train_encoded["asset_id"], prefix="asset")
    X_train_encoded = pd.concat([X_train_encoded.drop(["asset_id", "asset_name"], axis=1), asset_dummies], axis=1)

    logger.info(f"Features after encoding: {X_train_encoded.shape[1]}")
    logger.info(f"Feature columns: {list(X_train_encoded.columns)}")

    # Train model with GPU acceleration
    model = XGBoostModel(
        max_depth=4,  # Slightly deeper for cross-asset complexity
        learning_rate=0.05,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        tree_method="hist",  # GPU-compatible method
        device="cuda",       # Use GPU
        random_state=42
    )

    model.fit(X_train_encoded, y_train)
    logger.info("Cross-asset model training complete")

    return model, X_train_encoded.columns


def evaluate_cross_asset(model, X, y, feature_cols, split_name: str):
    """Evaluate cross-asset model.

    Args:
        model: Trained model
        X: Features (raw, before encoding)
        y: Targets
        feature_cols: Feature column names (after encoding)
        split_name: Name of split

    Returns:
        dict with metrics
    """
    logger.info(f"Evaluating on {split_name}...")

    # Encode features
    X_encoded = X.copy()
    asset_dummies = pd.get_dummies(X_encoded["asset_id"], prefix="asset")
    X_encoded = pd.concat([X_encoded.drop(["asset_id", "asset_name"], axis=1), asset_dummies], axis=1)

    # Align columns with training (handle missing asset dummies)
    for col in feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[feature_cols]

    # Predict
    predictions = model.predict(X_encoded)

    # Metrics
    accuracy = (predictions == y).mean()
    positive_rate = predictions.mean()
    target_balance = y.mean()

    metrics = {
        f"{split_name}_accuracy": accuracy,
        f"{split_name}_positive_rate": positive_rate,
        f"{split_name}_target_balance": target_balance,
    }

    logger.info(f"{split_name} - Accuracy: {accuracy:.3f}, Positive rate: {positive_rate:.1%}, Target balance: {target_balance:.1%}")

    return metrics


def main():
    """Main execution: train cross-asset model."""

    logger.info("=" * 80)
    logger.info("TRAIN CROSS-ASSET MODEL (APPROACH 2)")
    logger.info("=" * 80)

    # Load pooled data
    X, y = load_cross_asset_data()

    # Create splits
    splits = create_cross_asset_splits(X, y)

    # Remove NaNs
    train_valid_mask = ~splits["X_train"].drop(["asset_id", "asset_name"], axis=1).isna().any(axis=1)
    X_train_clean = splits["X_train"][train_valid_mask]
    y_train_clean = splits["y_train"][train_valid_mask]
    logger.info(f"Removed {(~train_valid_mask).sum()} NaN rows, {len(X_train_clean):,} remain")

    # Train cross-asset model
    model, feature_cols = train_cross_asset_model(X_train_clean, y_train_clean)

    # Evaluate on train
    train_metrics = evaluate_cross_asset(model, X_train_clean, y_train_clean, feature_cols, "train")

    # Evaluate on validation (all assets 2024)
    val_valid_mask = ~splits["X_val"].drop(["asset_id", "asset_name"], axis=1).isna().any(axis=1)
    X_val_clean = splits["X_val"][val_valid_mask]
    y_val_clean = splits["y_val"][val_valid_mask]
    val_metrics = evaluate_cross_asset(model, X_val_clean, y_val_clean, feature_cols, "validation")

    # Evaluate on BTC 2025 holdout
    holdout_valid_mask = ~splits["X_holdout"].drop(["asset_id", "asset_name"], axis=1).isna().any(axis=1)
    X_holdout_clean = splits["X_holdout"][holdout_valid_mask]
    y_holdout_clean = splits["y_holdout"][holdout_valid_mask]
    holdout_metrics = evaluate_cross_asset(model, X_holdout_clean, y_holdout_clean, feature_cols, "btc_holdout")

    # Log to MLflow
    mlflow.set_experiment("phase3_cross_asset")
    with mlflow.start_run(run_name="xgboost_cross_asset_7_cryptos"):
        # Log parameters
        mlflow.log_param("approach", "APPROACH_2_cross_asset")
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("num_assets", 7)
        mlflow.log_param("train_samples", len(X_train_clean))
        mlflow.log_param("assets", "BTC,ETH,SOL,ADA,DOT,MATIC,AVAX")

        # Log metrics
        for key, value in {**train_metrics, **val_metrics, **holdout_metrics}.items():
            mlflow.log_metric(key, value)

        # Feature importance
        importance = model.get_feature_importances()
        top_10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 features:")
        for feat, imp in top_10:
            logger.info(f"  {feat}: {imp:.4f}")
            mlflow.log_metric(f"importance_{feat}", imp)

        logger.info(f"Logged to MLflow run: {mlflow.active_run().info.run_id}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-ASSET RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Train accuracy (all assets 2017-2023): {train_metrics['train_accuracy']:.3f}")
    logger.info(f"Validation accuracy (all assets 2024): {val_metrics['validation_accuracy']:.3f}")
    logger.info(f"BTC holdout accuracy (2025): {holdout_metrics['btc_holdout_accuracy']:.3f}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO BTC-ONLY")
    logger.info("=" * 80)
    logger.info("Run scripts/train_on_hourly.py to compare:")
    logger.info("  - BTC-only model (trained on BTC hourly data)")
    logger.info("  - Cross-asset model (trained on 7 assets)")
    logger.info("Expected: Cross-asset should generalize better (more data, universal patterns)")


if __name__ == "__main__":
    main()
