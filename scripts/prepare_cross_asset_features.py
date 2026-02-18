#!/usr/bin/env python3
"""Prepare Cross-Asset Feature Matrix

Pool 7 assets into single training set with asset_id as categorical feature.

Strategy:
1. Load hourly OHLCV for all 7 assets
2. Compute identical features for each asset
3. Add asset_id column (categorical: 0-6)
4. Pool all assets into single DataFrame
5. Generate daily targets per asset
6. Save pooled feature matrix: ~490,000 hourly samples

Result:
- Training: 490K pooled hourly observations (all assets 2017-2023)
- Validation: Multi-asset 2024 (for hyperparameter tuning)
- Holdout: BTC 2025 ONLY (final test)

Model learns universal crypto dynamics, not BTC-specific noise.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Asset configuration (must match fetch_cross_asset_hourly.py)
ASSETS = [
    {"name": "btc", "asset_id": 0},
    {"name": "eth", "asset_id": 1},
    {"name": "sol", "asset_id": 2},
    {"name": "ada", "asset_id": 3},
    {"name": "dot", "asset_id": 4},
    {"name": "matic", "asset_id": 5},
    {"name": "avax", "asset_id": 6},
]


def load_asset_hourly(asset_name: str) -> pd.DataFrame:
    """Load hourly OHLCV for a single asset."""
    input_path = Path(f"data/raw/{asset_name}/ohlcv_hourly.parquet")

    if not input_path.exists():
        logger.warning(f"Asset {asset_name} hourly data not found: {input_path}")
        return pd.DataFrame()

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {asset_name}: {len(df):,} hourly candles")
    return df


def compute_features(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Compute comprehensive technical features on hourly data for a single asset.

    Uses the same 58-feature set as BTC hourly features for consistency.

    Args:
        df: Hourly OHLCV DataFrame
        asset_name: Asset identifier (for logging)

    Returns:
        DataFrame with 58 hourly features
    """
    import sys

    sys.path.insert(0, "scripts")
    from prepare_hourly_features import compute_hourly_features

    logger.info(f"  {asset_name}: Computing 58 features...")

    # Use the same feature computation as BTC
    features = compute_hourly_features(df)

    logger.info(f"  {asset_name}: Computed {features.shape[1]} features")
    return features


def generate_targets(prices: pd.DataFrame, asset_name: str, horizon_days: int = 1) -> pd.Series:
    """Generate daily targets from hourly prices.

    Args:
        prices: Hourly OHLCV DataFrame
        asset_name: Asset identifier
        horizon_days: Prediction horizon in days

    Returns:
        Series of binary targets (1 = long, 0 = flat)
    """
    # Resample to daily close
    prices_daily = prices.resample("D").last()
    prices_daily.index = prices_daily.index.tz_localize(None)

    # Target: close at T+horizon > close at T
    future_close = prices_daily["close"].shift(-horizon_days)
    current_close = prices_daily["close"]

    targets = (future_close > current_close).astype(int)
    targets = targets.dropna()

    logger.info(f"  {asset_name}: Generated {len(targets)} daily targets ({targets.mean():.1%} positive)")

    return targets


def resample_features_to_daily(features_hourly: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Resample hourly features to daily close."""
    features_daily = features_hourly.resample("D").last()
    features_daily.index = features_daily.index.tz_localize(None)

    logger.info(f"  {asset_name}: Resampled to {len(features_daily)} daily rows")
    return features_daily


def main():
    """Main execution: prepare cross-asset feature matrix."""

    logger.info("=" * 80)
    logger.info("PREPARE CROSS-ASSET FEATURE MATRIX")
    logger.info("=" * 80)
    logger.info(f"Assets: {len(ASSETS)}")
    logger.info("Strategy: Pool all assets with asset_id as categorical feature")
    logger.info("=" * 80)

    all_features = []
    all_targets = []

    for asset in ASSETS:
        asset_name = asset["name"]
        asset_id = asset["asset_id"]

        logger.info(f"\nProcessing {asset_name} (asset_id={asset_id})...")

        # Load hourly data
        df_hourly = load_asset_hourly(asset_name)

        if df_hourly.empty:
            logger.warning(f"Skipping {asset_name} (no data)")
            continue

        # Compute features
        features_hourly = compute_features(df_hourly, asset_name)

        # Generate targets (daily)
        targets_daily = generate_targets(df_hourly, asset_name, horizon_days=1)

        # Resample features to daily
        features_daily = resample_features_to_daily(features_hourly, asset_name)

        # Align features and targets
        common_dates = features_daily.index.intersection(targets_daily.index)
        features_aligned = features_daily.loc[common_dates].copy()
        targets_aligned = targets_daily.loc[common_dates].copy()

        # Add asset_id column
        features_aligned["asset_id"] = asset_id

        # Add asset_name for reference
        features_aligned["asset_name"] = asset_name

        logger.info(f"  {asset_name}: Aligned {len(features_aligned)} samples (features + targets)")

        all_features.append(features_aligned)
        all_targets.append(targets_aligned)

    # Pool all assets
    logger.info("\n" + "=" * 80)
    logger.info("POOLING ASSETS")
    logger.info("=" * 80)

    features_pooled = pd.concat(all_features, axis=0)
    targets_pooled = pd.concat(all_targets, axis=0)

    # Targets should already align with features (both built from common_dates)
    # Verify alignment
    if len(targets_pooled) != len(features_pooled):
        raise ValueError(f"Features ({len(features_pooled)}) and targets ({len(targets_pooled)}) length mismatch!")
    if not (targets_pooled.index == features_pooled.index).all():
        raise ValueError("Features and targets index mismatch!")

    logger.info(f"Pooled feature matrix: {features_pooled.shape[0]:,} rows × {features_pooled.shape[1]} columns")
    logger.info(f"Pooled targets: {len(targets_pooled):,} daily labels")

    # Asset distribution
    asset_counts = features_pooled["asset_name"].value_counts()
    logger.info(f"\nSamples per asset:\n{asset_counts}")

    # Save pooled feature matrix
    output_path = Path("data/processed/feature_matrix_cross_asset_hourly.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_pooled.to_parquet(output_path)
    logger.info(f"\nSaved pooled features to {output_path}")

    # Save pooled targets
    targets_path = Path("data/processed/targets_cross_asset_hourly_1d.parquet")
    targets_df = pd.DataFrame({"target": targets_pooled})
    targets_df.to_parquet(targets_path)
    logger.info(f"Saved pooled targets to {targets_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS")
    logger.info("=" * 80)
    logger.info(f"Pooled samples: {len(features_pooled):,}")
    logger.info(f"Assets: {features_pooled['asset_name'].nunique()}")
    logger.info(f"Features: {features_pooled.shape[1] - 2} (excluding asset_id, asset_name)")
    logger.info(f"Date range: {features_pooled.index.min().date()} to {features_pooled.index.max().date()}")
    logger.info(f"Target balance: {targets_pooled.mean():.1%} positive (long)")
    logger.info("\nNext step: Run scripts/train_cross_asset.py")


if __name__ == "__main__":
    main()
