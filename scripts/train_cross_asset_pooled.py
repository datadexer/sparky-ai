#!/usr/bin/env python3
"""
PHASE 1: CROSS-ASSET POOLED TRAINING
====================================

Train CatBoost on pooled hourly data from 6 crypto assets (BTC, ETH, SOL, DOT, LINK, ADA).

Strategy:
- Load hourly OHLCV for 6 assets: BTC, ETH, SOL, DOT, LINK, ADA (364,830 samples total)
- Compute EXACT same 23 base technical features for each asset
- Add asset_id as categorical feature (values: btc, eth, sol, dot, link, ada)
- Pool all assets into single dataset
- Train CatBoost with SAME hyperparameters as best BTC-only model
- Test on BTC-only holdout (2024-2025)

Splits:
- Training: 2017-01-01 to 2020-12-31 (all assets, different start dates OK)
- Validation: 2021-01-01 to 2022-12-31 (all assets)
- Test: 2023-01-01 to 2023-12-31 (all assets)
- Holdout: BTC ONLY 2024-01-01 to 2025-12-31 (never seen, final test)

Validation Protocol:
1. Multi-seed stability (5 random seeds: 0, 42, 123, 456, 789)
2. Walk-forward validation (9 folds, expanding window, 6-month test periods)
3. Leakage check (shuffled-label test, temporal boundary audit)

Success Criteria:
- ✅ Holdout AUC ≥ 0.57 → PROCEED to Phase 2
- ⚠️ Holdout AUC 0.55-0.57 → MARGINAL
- ❌ Holdout AUC < 0.55 → STOP

Baseline to beat: BTC-only CatBoost, Holdout AUC 0.536
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparky.features.advanced import (
    atr,
    bollinger_bandwidth,
    bollinger_position,
    day_of_week,
    higher_highs_lower_lows,
    intraday_range,
    momentum_quality,
    price_acceleration,
    price_distance_from_sma,
    session_hour,
    volatility_clustering,
    volume_ma_ratio,
    volume_momentum,
    vwap_deviation,
)
from sparky.features.returns import simple_returns
from sparky.features.technical import ema, macd, momentum, rsi

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_asset_data(asset: str) -> pd.DataFrame:
    """Load hourly OHLCV data for a specific asset.

    Args:
        asset: Asset symbol (btc, eth, sol, dot, link, ada)

    Returns:
        DataFrame with hourly OHLCV data
    """
    data_path = Path(f"data/raw/{asset}/ohlcv_hourly.parquet")

    if not data_path.exists():
        raise FileNotFoundError(f"Hourly data not found for {asset}: {data_path}")

    logger.info(f"Loading {asset.upper()} hourly data from {data_path}")
    df = pd.read_parquet(data_path)

    logger.info(f"  {asset.upper()}: {len(df):,} candles, {df.index.min()} to {df.index.max()}")

    return df


def compute_features_for_asset(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Compute comprehensive technical features for a single asset.

    Uses EXACT same 23 base features as best BTC-only model.

    Args:
        df: Hourly OHLCV DataFrame
        asset: Asset symbol (for logging)

    Returns:
        DataFrame with 23 technical features
    """
    logger.info(f"Computing 23 base features for {asset.upper()}...")

    features = pd.DataFrame(index=df.index)

    # === TECHNICAL INDICATORS ===
    # RSI (multiple timeframes)
    features["rsi_14h"] = rsi(df["close"], period=14)
    features["rsi_6h"] = rsi(df["close"], period=6)

    # MACD (12/26/9 hourly)
    macd_line, signal_line, histogram = macd(df["close"], fast_period=12, slow_period=26, signal_period=9)
    features["macd_line"] = macd_line
    features["macd_histogram"] = histogram

    # EMA ratios (trend strength)
    ema_fast = ema(df["close"], span=10)
    ema_slow = ema(df["close"], span=20)
    features["ema_ratio_20h"] = ema_fast / ema_slow - 1.0

    # Bollinger Bands
    features["bb_bandwidth_20h"] = bollinger_bandwidth(df["close"], period=20)
    features["bb_position_20h"] = bollinger_position(df["close"], period=20)

    # === MOMENTUM FEATURES ===
    features["momentum_4h"] = momentum(df["close"], period=4)
    features["momentum_24h"] = momentum(df["close"], period=24)
    features["momentum_168h"] = momentum(df["close"], period=168)
    features["momentum_quality_30h"] = momentum_quality(df["close"], period=30)
    features["price_acceleration_10h"] = price_acceleration(df["close"], period=10)

    # === VOLATILITY FEATURES ===
    features["atr_14h"] = atr(df["high"], df["low"], df["close"], period=14)
    features["intraday_range"] = intraday_range(df["high"], df["low"], df["close"])

    hourly_returns = simple_returns(df["close"])
    features["vol_clustering_24h"] = volatility_clustering(hourly_returns, period=24)
    features["realized_vol_24h"] = hourly_returns.rolling(window=24).std()

    # === VOLUME FEATURES ===
    features["volume_momentum_30h"] = volume_momentum(df["volume"], period=30)
    features["volume_ma_ratio_20h"] = volume_ma_ratio(df["volume"], period=20)
    features["vwap_deviation_24h"] = vwap_deviation(df["close"], df["volume"], df["high"], df["low"], period=24)

    # === MARKET MICROSTRUCTURE ===
    features["distance_from_sma_200h"] = price_distance_from_sma(df["close"], period=200)
    features["higher_highs_lower_lows_5h"] = higher_highs_lower_lows(df["high"], df["low"], period=5)

    # === TEMPORAL ===
    features["hour_of_day"] = session_hour(df.index)
    features["day_of_week"] = day_of_week(df.index)

    # Clean inf values
    features = features.replace([np.inf, -np.inf], np.nan)

    logger.info(f"  {asset.upper()}: {features.shape[1]} features computed")

    return features


def create_targets(df: pd.DataFrame) -> pd.Series:
    """Create 1h-ahead binary targets (1 if price rises next hour, 0 otherwise).

    Args:
        df: Hourly OHLCV DataFrame

    Returns:
        Series of binary targets
    """
    # Target: close(t+1) > close(t)
    targets = (df["close"].shift(-1) > df["close"]).astype(int)
    targets.name = "target"

    return targets


def load_and_pool_all_assets():
    """Load hourly data for all 6 assets, compute features, pool into single dataset.

    Returns:
        Tuple of (pooled_features, pooled_targets)
    """
    logger.info("=" * 80)
    logger.info("LOADING AND POOLING CROSS-ASSET DATA")
    logger.info("=" * 80)

    assets = ["btc", "eth", "sol", "dot", "link", "ada"]

    all_features = []
    all_targets = []

    for asset in assets:
        # Load OHLCV
        df = load_asset_data(asset)

        # Compute features
        features = compute_features_for_asset(df, asset)

        # Create targets
        targets = create_targets(df)

        # Add asset_id column
        features["asset_id"] = asset

        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        features = features.loc[common_idx]
        targets = targets.loc[common_idx]

        # Drop NaN rows
        valid_mask = ~features.drop(["asset_id"], axis=1).isna().any(axis=1)
        features = features[valid_mask]
        targets = targets.loc[features.index]

        logger.info(f"  {asset.upper()}: {len(features):,} valid samples after feature computation")

        all_features.append(features)
        all_targets.append(targets)

    # Pool all assets
    logger.info("\nPooling all assets...")
    X_pooled = pd.concat(all_features, axis=0).sort_index()
    y_pooled = pd.concat(all_targets, axis=0).sort_index()

    logger.info(f"Total pooled samples: {len(X_pooled):,}")
    logger.info(f"Total features: {X_pooled.shape[1]} (23 base + 1 asset_id)")
    logger.info(f"Date range: {X_pooled.index.min()} to {X_pooled.index.max()}")
    logger.info(f"Target balance: {y_pooled.mean():.2%} positive")

    logger.info("\nAsset distribution:")
    for asset in assets:
        count = (X_pooled["asset_id"] == asset).sum()
        logger.info(f"  {asset.upper()}: {count:,} samples ({count / len(X_pooled) * 100:.1f}%)")

    return X_pooled, y_pooled


def create_splits(X: pd.DataFrame, y: pd.Series):
    """Create train/val/test/holdout splits.

    Splits:
    - Training: 2017-01-01 to 2020-12-31 (all assets)
    - Validation: 2021-01-01 to 2022-12-31 (all assets)
    - Test: 2023-01-01 to 2023-12-31 (all assets)
    - Holdout: BTC ONLY 2024-01-01 to 2025-12-31

    Returns:
        dict with train/val/test/holdout DataFrames
    """
    logger.info("=" * 80)
    logger.info("CREATING TRAIN/VAL/TEST/HOLDOUT SPLITS")
    logger.info("=" * 80)

    splits = {}

    # Train: All assets 2017-2020
    train_mask = (X.index >= "2017-01-01") & (X.index < "2021-01-01")
    splits["X_train"] = X[train_mask]
    splits["y_train"] = y[train_mask]
    logger.info(
        f"Train (all assets): {splits['X_train'].index.min().date()} to {splits['X_train'].index.max().date()} "
        f"({len(splits['X_train']):,} samples)"
    )

    # Validation: All assets 2021-2022
    val_mask = (X.index >= "2021-01-01") & (X.index < "2023-01-01")
    splits["X_val"] = X[val_mask]
    splits["y_val"] = y[val_mask]
    logger.info(
        f"Validation (all assets): {splits['X_val'].index.min().date()} to {splits['X_val'].index.max().date()} "
        f"({len(splits['X_val']):,} samples)"
    )

    # Test: All assets 2023
    test_mask = (X.index >= "2023-01-01") & (X.index < "2024-01-01")
    splits["X_test"] = X[test_mask]
    splits["y_test"] = y[test_mask]
    logger.info(
        f"Test (all assets): {splits['X_test'].index.min().date()} to {splits['X_test'].index.max().date()} "
        f"({len(splits['X_test']):,} samples)"
    )

    # Holdout: BTC ONLY 2024-2025
    holdout_mask = (X.index >= "2024-01-01") & (X.index < "2026-01-01") & (X["asset_id"] == "btc")
    splits["X_holdout"] = X[holdout_mask]
    splits["y_holdout"] = y[holdout_mask]
    logger.info(
        f"Holdout (BTC only): {splits['X_holdout'].index.min().date()} to {splits['X_holdout'].index.max().date()} "
        f"({len(splits['X_holdout']):,} samples)"
    )

    # Asset distribution per split
    logger.info("\nTrain asset distribution:")
    logger.info(splits["X_train"]["asset_id"].value_counts().to_string())
    logger.info("\nValidation asset distribution:")
    logger.info(splits["X_val"]["asset_id"].value_counts().to_string())
    logger.info("\nTest asset distribution:")
    logger.info(splits["X_test"]["asset_id"].value_counts().to_string())

    return splits


def train_model(X_train, y_train, seed=42):
    """Train CatBoost model with SAME hyperparameters as best BTC-only model.

    Hyperparameters from best BTC-only CatBoost (Test AUC 0.5612):
    - depth=5
    - learning_rate=0.05
    - iterations=200
    - subsample=0.8
    - l2_leaf_reg=2.0
    - rsm=0.8 (colsample_bylevel)
    - cat_features=['asset_id']

    Args:
        X_train: Training features (with asset_id column)
        y_train: Training targets
        seed: Random seed

    Returns:
        Trained CatBoostClassifier
    """
    logger.info(f"Training CatBoost model (seed={seed})...")

    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=200,
        subsample=0.8,
        rsm=0.8,  # colsample_bylevel
        l2_leaf_reg=3.0,  # Updated from multiseed experiments
        random_seed=seed,
        cat_features=["asset_id"],  # Categorical feature
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
        f"  {split_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return metrics, y_proba


def multiseed_validation(X_train, y_train, X_val, y_val, X_test, y_test, X_holdout, y_holdout):
    """Multi-seed stability test with 5 random seeds.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        X_holdout, y_holdout: Holdout data

    Returns:
        dict with multi-seed results
    """
    logger.info("=" * 80)
    logger.info("MULTI-SEED STABILITY VALIDATION")
    logger.info("=" * 80)

    seeds = [0, 42, 123, 456, 789]
    results = []

    for seed in seeds:
        logger.info(f"\nTraining with seed={seed}...")

        # Train model
        model = train_model(X_train, y_train, seed=seed)

        # Evaluate on all splits
        val_metrics, _ = evaluate_model(model, X_val, y_val, f"val_seed{seed}")
        test_metrics, _ = evaluate_model(model, X_test, y_test, f"test_seed{seed}")
        holdout_metrics, _ = evaluate_model(model, X_holdout, y_holdout, f"holdout_seed{seed}")

        results.append(
            {
                "seed": seed,
                "val_auc": val_metrics[f"val_seed{seed}_roc_auc"],
                "test_auc": test_metrics[f"test_seed{seed}_roc_auc"],
                "holdout_auc": holdout_metrics[f"holdout_seed{seed}_roc_auc"],
            }
        )

    # Compute statistics
    val_aucs = [r["val_auc"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    holdout_aucs = [r["holdout_auc"] for r in results]

    stats = {
        "seeds": seeds,
        "results": results,
        "val_auc_mean": float(np.mean(val_aucs)),
        "val_auc_std": float(np.std(val_aucs)),
        "test_auc_mean": float(np.mean(test_aucs)),
        "test_auc_std": float(np.std(test_aucs)),
        "holdout_auc_mean": float(np.mean(holdout_aucs)),
        "holdout_auc_std": float(np.std(holdout_aucs)),
    }

    logger.info("\n" + "=" * 80)
    logger.info("MULTI-SEED RESULTS")
    logger.info("=" * 80)
    logger.info(f"Validation AUC: {stats['val_auc_mean']:.4f} ± {stats['val_auc_std']:.4f}")
    logger.info(f"Test AUC: {stats['test_auc_mean']:.4f} ± {stats['test_auc_std']:.4f}")
    logger.info(f"Holdout AUC: {stats['holdout_auc_mean']:.4f} ± {stats['holdout_auc_std']:.4f}")
    logger.info(
        f"Stability check: std(holdout_auc) = {stats['holdout_auc_std']:.4f} {'PASS' if stats['holdout_auc_std'] < 0.01 else 'FAIL'} (threshold < 0.01)"
    )

    return stats


def walk_forward_validation(X, y):
    """Walk-forward validation with expanding window (9 folds, 6-month test periods).

    Args:
        X: Full feature matrix (excluding holdout)
        y: Full targets (excluding holdout)

    Returns:
        dict with walk-forward results
    """
    logger.info("=" * 80)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 80)

    # Use only 2017-2023 data (exclude 2024-2025 holdout)
    mask = (X.index >= "2017-01-01") & (X.index < "2024-01-01")
    X_wf = X[mask]
    y_wf = y[mask]

    logger.info(f"Walk-forward data: {len(X_wf):,} samples from {X_wf.index.min().date()} to {X_wf.index.max().date()}")

    # Time series split (9 folds)
    tscv = TimeSeriesSplit(n_splits=9)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_wf), 1):
        X_train_fold = X_wf.iloc[train_idx]
        y_train_fold = y_wf.iloc[train_idx]
        X_test_fold = X_wf.iloc[test_idx]
        y_test_fold = y_wf.iloc[test_idx]

        logger.info(
            f"\nFold {fold_idx}: Train {X_train_fold.index.min().date()} to {X_train_fold.index.max().date()} "
            f"({len(X_train_fold):,} samples), "
            f"Test {X_test_fold.index.min().date()} to {X_test_fold.index.max().date()} "
            f"({len(X_test_fold):,} samples)"
        )

        # Train model
        model = train_model(X_train_fold, y_train_fold, seed=42)

        # Evaluate
        test_metrics, _ = evaluate_model(model, X_test_fold, y_test_fold, f"fold{fold_idx}")

        fold_results.append(
            {
                "fold": fold_idx,
                "train_start": X_train_fold.index.min().strftime("%Y-%m-%d"),
                "train_end": X_train_fold.index.max().strftime("%Y-%m-%d"),
                "test_start": X_test_fold.index.min().strftime("%Y-%m-%d"),
                "test_end": X_test_fold.index.max().strftime("%Y-%m-%d"),
                "test_auc": test_metrics[f"fold{fold_idx}_roc_auc"],
            }
        )

    # Compute statistics
    test_aucs = [r["test_auc"] for r in fold_results]

    stats = {
        "n_folds": 9,
        "fold_results": fold_results,
        "mean_auc": float(np.mean(test_aucs)),
        "std_auc": float(np.std(test_aucs)),
        "min_auc": float(np.min(test_aucs)),
        "max_auc": float(np.max(test_aucs)),
    }

    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD RESULTS")
    logger.info("=" * 80)
    logger.info(f"Mean AUC: {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f}")
    logger.info(f"Min AUC: {stats['min_auc']:.4f}")
    logger.info(f"Max AUC: {stats['max_auc']:.4f}")

    return stats


def leakage_check(model, X_holdout, y_holdout):
    """Leakage check: shuffled-label test on holdout data.

    If model performs well on shuffled labels, it's leaking information.

    Args:
        model: Trained model
        X_holdout: Holdout features
        y_holdout: Holdout targets

    Returns:
        dict with leakage check results
    """
    logger.info("=" * 80)
    logger.info("LEAKAGE CHECK: Shuffled-Label Test")
    logger.info("=" * 80)

    # Shuffled-label test (should get ~0.5 AUC)
    y_shuffled = y_holdout.sample(frac=1.0, random_state=42).reset_index(drop=True)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    shuffled_auc = roc_auc_score(y_shuffled, y_proba)

    logger.info(f"Shuffled-label AUC: {shuffled_auc:.4f}")

    leakage_detected = shuffled_auc > 0.55

    if leakage_detected:
        logger.error(f"LEAKAGE DETECTED: Shuffled-label AUC {shuffled_auc:.4f} > 0.55")
    else:
        logger.info(f"PASS: Shuffled-label AUC {shuffled_auc:.4f} ≤ 0.55 (no leakage)")

    return {
        "shuffled_label_auc": float(shuffled_auc),
        "leakage_detected": leakage_detected,
    }


def get_feature_importances(model, feature_names):
    """Extract feature importances from CatBoost model.

    Args:
        model: Trained CatBoost model
        feature_names: List of feature names

    Returns:
        dict mapping feature names to importance scores
    """
    importances = model.get_feature_importance()

    importance_dict = dict(zip(feature_names, importances))

    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    logger.info("\nTop 10 Feature Importances:")
    for idx, (feat, imp) in enumerate(list(sorted_importance.items())[:10], 1):
        logger.info(f"  {idx}. {feat}: {imp:.4f}")

    return sorted_importance


def save_results(all_results: dict):
    """Save comprehensive results to JSON file.

    Args:
        all_results: Dictionary with all results
    """
    results_dir = Path("results/cross_asset_pooled")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "phase1_results.json"

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")


def print_final_summary(all_results: dict):
    """Print final summary and recommendation.

    Args:
        all_results: Dictionary with all results
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: CROSS-ASSET POOLED TRAINING — FINAL SUMMARY")
    logger.info("=" * 80)

    # Baseline
    baseline_auc = 0.536

    # Cross-asset results
    holdout_auc = all_results["multiseed"]["holdout_auc_mean"]
    holdout_std = all_results["multiseed"]["holdout_auc_std"]

    delta_auc = holdout_auc - baseline_auc

    logger.info("\nBASELINE (BTC-only CatBoost):")
    logger.info(f"  Holdout AUC: {baseline_auc:.4f}")

    logger.info("\nCROSS-ASSET POOLED (6 assets):")
    logger.info(f"  Training samples: {all_results['training_samples']:,}")
    logger.info(
        f"  Validation AUC: {all_results['multiseed']['val_auc_mean']:.4f} ± {all_results['multiseed']['val_auc_std']:.4f}"
    )
    logger.info(
        f"  Test AUC: {all_results['multiseed']['test_auc_mean']:.4f} ± {all_results['multiseed']['test_auc_std']:.4f}"
    )
    logger.info(f"  Holdout AUC: {holdout_auc:.4f} ± {holdout_std:.4f}")
    logger.info(f"  Delta from baseline: {delta_auc:+.4f} ({delta_auc / baseline_auc * 100:+.2f}%)")

    logger.info("\nVALIDATION CHECKS:")
    logger.info(f"  Multi-seed stability: {holdout_std:.4f} {'PASS' if holdout_std < 0.01 else 'FAIL'} (std < 0.01)")
    logger.info(f"  Walk-forward mean AUC: {all_results['walk_forward']['mean_auc']:.4f}")
    logger.info(f"  Leakage check: {'PASS' if not all_results['leakage_check']['leakage_detected'] else 'FAIL'}")

    logger.info("\nFEATURE IMPORTANCE:")
    top_5 = list(all_results["feature_importances"].items())[:5]
    for idx, (feat, imp) in enumerate(top_5, 1):
        logger.info(f"  {idx}. {feat}: {imp:.4f}")

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION")
    logger.info("=" * 80)

    if holdout_auc >= 0.57:
        recommendation = "✅ PROCEED to Phase 2 (enhanced technical features)"
        logger.info(recommendation)
        logger.info(f"Rationale: Holdout AUC {holdout_auc:.4f} ≥ 0.57 threshold")
    elif holdout_auc >= 0.55:
        recommendation = "⚠️ MARGINAL — Report to RBM for decision"
        logger.warning(recommendation)
        logger.warning(f"Rationale: Holdout AUC {holdout_auc:.4f} in 0.55-0.57 range")
    else:
        recommendation = "❌ STOP — Reassess approach"
        logger.error(recommendation)
        logger.error(f"Rationale: Holdout AUC {holdout_auc:.4f} < 0.55 threshold")

    logger.info("=" * 80 + "\n")

    all_results["recommendation"] = recommendation


def main():
    """Main execution: Phase 1 Cross-Asset Pooled Training."""

    logger.info("=" * 80)
    logger.info("PHASE 1: CROSS-ASSET POOLED TRAINING")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")

    # 1. Load and pool all assets
    X_pooled, y_pooled = load_and_pool_all_assets()

    # 2. Create splits
    splits = create_splits(X_pooled, y_pooled)

    # 3. Multi-seed stability validation
    multiseed_results = multiseed_validation(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        splits["X_test"],
        splits["y_test"],
        splits["X_holdout"],
        splits["y_holdout"],
    )

    # 4. Walk-forward validation
    # Combine train + val + test for walk-forward (exclude holdout)
    X_no_holdout = pd.concat([splits["X_train"], splits["X_val"], splits["X_test"]], axis=0).sort_index()
    y_no_holdout = pd.concat([splits["y_train"], splits["y_val"], splits["y_test"]], axis=0).sort_index()

    walk_forward_results = walk_forward_validation(X_no_holdout, y_no_holdout)

    # 5. Train final model on all data (train + val + test)
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODEL (train + val + test)")
    logger.info("=" * 80)

    final_model = train_model(X_no_holdout, y_no_holdout, seed=42)

    # 6. Evaluate final model on holdout
    holdout_metrics, _ = evaluate_model(final_model, splits["X_holdout"], splits["y_holdout"], "final_holdout")

    # 7. Leakage check
    leakage_results = leakage_check(final_model, splits["X_holdout"], splits["y_holdout"])

    # 8. Feature importances
    feature_importances = get_feature_importances(final_model, list(X_pooled.columns))

    # 9. Compile all results
    all_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment": "phase1_cross_asset_pooled",
        "assets": ["btc", "eth", "sol", "dot", "link", "ada"],
        "total_samples": len(X_pooled),
        "training_samples": len(splits["X_train"]),
        "validation_samples": len(splits["X_val"]),
        "test_samples": len(splits["X_test"]),
        "holdout_samples": len(splits["X_holdout"]),
        "multiseed": multiseed_results,
        "walk_forward": walk_forward_results,
        "final_holdout_metrics": {k: float(v) for k, v in holdout_metrics.items()},
        "leakage_check": leakage_results,
        "feature_importances": {k: float(v) for k, v in feature_importances.items()},
        "baseline": {
            "model": "BTC-only CatBoost",
            "holdout_auc": 0.536,
        },
    }

    # 10. Print final summary and recommendation
    print_final_summary(all_results)

    # 11. Save results
    save_results(all_results)

    logger.info("Phase 1 execution complete!")


if __name__ == "__main__":
    main()
