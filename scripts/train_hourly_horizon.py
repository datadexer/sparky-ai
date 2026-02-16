#!/usr/bin/env python3
"""Train XGBoost on Hourly BTC Data — Single Horizon

Parameterized by --horizon {1h, 4h, 24h, exec24h}.
Trains on hourly features, evaluates with both standard (all samples) and
non-overlapping (honest, independent) evaluation modes.

Usage:
    python scripts/train_hourly_horizon.py --horizon 1h
    python scripts/train_hourly_horizon.py --horizon 4h
    python scripts/train_hourly_horizon.py --horizon 24h
    python scripts/train_hourly_horizon.py --horizon exec24h

Outputs:
    results/hourly_horizon_experiments/{horizon}_results.json
    MLflow run in experiment "hourly_multi_horizon"
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_HORIZONS = ["1h", "4h", "24h", "exec24h"]
HORIZON_STRIDE = {"1h": 1, "4h": 4, "24h": 24, "exec24h": 24}

# Splits (hourly timestamps)
SPLIT_BOUNDS = {
    "train": ("2017-01-01", "2021-01-01"),
    "val": ("2021-01-01", "2023-01-01"),
    "test": ("2023-01-01", "2024-01-01"),
    "holdout": ("2024-01-01", "2026-01-01"),
}


def load_data(horizon: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load hourly features and target for the given horizon."""
    features_path = Path("data/processed/features_hourly_full.parquet")
    target_path = Path(f"data/processed/targets_hourly_{horizon}.parquet")

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found: {features_path}\n"
            "Run scripts/prepare_hourly_training_data.py first"
        )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Target not found: {target_path}\n"
            "Run scripts/prepare_hourly_training_data.py first"
        )

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(target_path)["target"]

    # Align on common index
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Replace inf with NaN (XGBoost handles NaN natively but crashes on inf)
    n_inf = np.isinf(X.values).sum()
    if n_inf > 0:
        logger.warning(f"Replacing {n_inf} inf values with NaN")
        X = X.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Loaded {len(X):,} samples × {X.shape[1]} features for horizon={horizon}")
    return X, y


def create_splits(
    X: pd.DataFrame, y: pd.Series
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Create temporal train/val/test/holdout splits."""
    splits = {}
    for name, (start, end) in SPLIT_BOUNDS.items():
        mask = (X.index >= start) & (X.index < end)
        X_split = X[mask]
        y_split = y[mask]
        if len(X_split) > 0:
            logger.info(
                f"{name:8s}: {X_split.index.min()} to {X_split.index.max()} "
                f"({len(X_split):,} samples, balance={y_split.mean():.3f})"
            )
        else:
            logger.warning(f"{name}: empty split")
        splits[name] = (X_split, y_split)
    return splits


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "n_samples": len(y_true),
        "class_balance": float(y_true.mean()),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def evaluate_standard_and_nonoverlap(
    model, X: pd.DataFrame, y: pd.Series, stride: int, split_name: str
) -> dict:
    """Evaluate model in both standard and non-overlapping modes.

    Args:
        model: Trained model with predict() and predict_proba().
        X: Feature matrix.
        y: Binary targets.
        stride: Non-overlapping stride (1 for 1h, 4 for 4h, 24 for 24h/exec).
        split_name: Name for logging.

    Returns:
        Dict with 'standard' and 'nonoverlap' metric dicts.
    """
    results = {}

    # Standard evaluation (all samples)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    results["standard"] = compute_metrics(y, y_pred, y_proba)
    logger.info(
        f"{split_name} [standard]: acc={results['standard']['accuracy']:.4f}, "
        f"auc={results['standard']['roc_auc']:.4f}, n={len(y):,}"
    )

    # Non-overlapping evaluation (every Nth sample for honest assessment)
    X_no = X.iloc[::stride]
    y_no = y.iloc[::stride]
    y_pred_no = model.predict(X_no)
    y_proba_no = model.predict_proba(X_no)[:, 1]
    results["nonoverlap"] = compute_metrics(y_no, y_pred_no, y_proba_no)
    logger.info(
        f"{split_name} [nonoverlap stride={stride}]: acc={results['nonoverlap']['accuracy']:.4f}, "
        f"auc={results['nonoverlap']['roc_auc']:.4f}, n={len(y_no):,}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost for a single horizon")
    parser.add_argument(
        "--horizon",
        type=str,
        required=True,
        choices=VALID_HORIZONS,
        help="Target horizon: 1h, 4h, 24h, or exec24h",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    args = parser.parse_args()

    horizon = args.horizon
    stride = HORIZON_STRIDE[horizon]

    logger.info("=" * 80)
    logger.info(f"TRAIN HOURLY HORIZON: {horizon} (stride={stride})")
    logger.info("=" * 80)

    # Load data
    X, y = load_data(horizon)

    # Create splits
    splits = create_splits(X, y)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Train XGBoost
    from sparky.models.xgboost_model import XGBoostModel

    logger.info("Training XGBoost (max_depth=5, n_estimators=200, lr=0.05)...")
    model = XGBoostModel(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on all splits
    results = {"horizon": horizon, "stride": stride}

    train_eval = evaluate_standard_and_nonoverlap(model, X_train, y_train, stride, "train")
    results["train"] = train_eval

    val_eval = evaluate_standard_and_nonoverlap(model, X_val, y_val, stride, "val")
    results["val"] = val_eval

    test_eval = evaluate_standard_and_nonoverlap(model, X_test, y_test, stride, "test")
    results["test"] = test_eval

    # Feature importance
    importance_df = model.get_feature_importances()
    results["feature_importance"] = importance_df.to_dict(orient="records")
    logger.info(f"\nTop 10 features ({horizon}):")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    # Leakage detection
    from sparky.backtest.leakage_detector import LeakageDetector

    logger.info("\nRunning leakage detection (n_trials=10)...")
    detector = LeakageDetector(n_shuffle_trials=10)
    report = detector.run_all_checks(model, X_train, y_train, X_val, y_val)
    results["leakage_passed"] = report.passed
    results["leakage_checks"] = []
    for check in report.checks:
        results["leakage_checks"].append({
            "name": check.check_name,
            "passed": check.passed,
            "detail": check.detail,
            "metric_value": check.metric_value,
        })
    logger.info(f"Leakage detection: {'PASSED' if report.passed else 'FAILED'}")
    if not report.passed:
        for check in report.failed_checks:
            logger.error(f"  FAILED: {check.check_name} — {check.detail}")

    # Val-test consistency
    val_auc = val_eval["nonoverlap"]["roc_auc"]
    test_auc = test_eval["nonoverlap"]["roc_auc"]
    gap = abs(val_auc - test_auc)
    results["val_test_auc_gap"] = gap
    results["val_test_consistent"] = gap < 0.05
    logger.info(f"\nVal-test AUC gap: {gap:.4f} ({'<5% PASS' if gap < 0.05 else '>=5% WARN'})")

    # Save results JSON
    output_dir = Path("results/hourly_horizon_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{horizon}_results.json"

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_path.write_text(json.dumps(results, indent=2, default=convert_types))
    logger.info(f"\nSaved results: {results_path}")

    # MLflow logging
    if not args.skip_mlflow:
        try:
            import mlflow

            mlflow.set_experiment("hourly_multi_horizon")
            with mlflow.start_run(run_name=f"xgboost_{horizon}"):
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("stride", stride)
                mlflow.log_param("model_type", "XGBoost")
                mlflow.log_param("max_depth", 5)
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("learning_rate", 0.05)
                mlflow.log_param("train_samples", len(X_train))

                # Log non-overlapping metrics (honest assessment)
                for split_name in ["train", "val", "test"]:
                    for metric_name, value in results[split_name]["nonoverlap"].items():
                        if isinstance(value, (int, float)) and metric_name != "confusion_matrix":
                            mlflow.log_metric(f"{split_name}_nonoverlap_{metric_name}", value)

                # Log standard metrics
                for split_name in ["train", "val", "test"]:
                    for metric_name, value in results[split_name]["standard"].items():
                        if isinstance(value, (int, float)) and metric_name != "confusion_matrix":
                            mlflow.log_metric(f"{split_name}_standard_{metric_name}", value)

                mlflow.log_metric("leakage_passed", 1.0 if report.passed else 0.0)
                mlflow.log_metric("val_test_auc_gap", gap)

                # Log feature importance
                for item in results["feature_importance"][:10]:
                    mlflow.log_metric(
                        f"importance_{item['feature']}", float(item["importance"])
                    )

                mlflow.log_artifact(str(results_path))
                logger.info(f"Logged to MLflow: {mlflow.active_run().info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-blocking): {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS SUMMARY — {horizon}")
    logger.info("=" * 80)
    logger.info(f"Leakage:              {'PASSED' if report.passed else 'FAILED'}")
    logger.info(f"Val AUC (nonoverlap): {val_auc:.4f}")
    logger.info(f"Test AUC (nonoverlap):{test_auc:.4f}")
    logger.info(f"Val-Test gap:         {gap:.4f}")
    logger.info(f"Val acc (nonoverlap): {val_eval['nonoverlap']['accuracy']:.4f}")
    logger.info(f"Test acc (nonoverlap):{test_eval['nonoverlap']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
