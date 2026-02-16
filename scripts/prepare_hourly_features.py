#!/usr/bin/env python3
"""Prepare Hourly Feature Matrix for BTC

Compute features on hourly data, resample to daily close for target alignment.

Strategy:
1. Load hourly OHLCV (~70K hourly candles)
2. Compute features on hourly frequency (RSI-14h, Momentum-30h, EMA-ratio-20h)
3. Resample features to daily close (last hourly value of each day)
4. Generate daily targets (close T+1 > close T)
5. Save feature matrix: ~2,900 daily rows with hourly-derived features

Result:
- Training samples: 52K-70K hourly observations
- Prediction granularity: Daily direction
- 24x more data than daily-only approach
"""

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_hourly_data() -> pd.DataFrame:
    """Load hourly OHLCV data."""
    input_path = Path("data/raw/btc/ohlcv_hourly.parquet")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Hourly data not found: {input_path}\n"
            "Run scripts/fetch_hourly_btc.py first"
        )

    logger.info(f"Loading hourly data from {input_path}")
    df = pd.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} hourly candles")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def compute_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features on hourly data.

    Args:
        df: Hourly OHLCV DataFrame

    Returns:
        DataFrame with hourly features
    """
    from sparky.features.technical import compute_rsi, compute_ema, simple_returns

    logger.info("Computing hourly features...")

    features = pd.DataFrame(index=df.index)

    # RSI-14 (hourly)
    logger.info("Computing RSI-14 (hourly)...")
    features["rsi_14h"] = compute_rsi(df["close"], period=14)

    # Momentum-30 hours (≈ 1.25 days)
    logger.info("Computing Momentum-30h...")
    features["momentum_30h"] = simple_returns(df["close"], periods=30)

    # EMA-ratio-20 hours
    logger.info("Computing EMA-ratio-20h...")
    ema_fast = compute_ema(df["close"], span=10)  # 10-hour EMA
    ema_slow = compute_ema(df["close"], span=20)  # 20-hour EMA
    features["ema_ratio_20h"] = ema_fast / ema_slow - 1.0

    # 1-hour returns
    logger.info("Computing returns-1h...")
    features["returns_1h"] = simple_returns(df["close"], periods=1)

    # Intraday volatility (24-hour rolling std of hourly returns)
    logger.info("Computing intraday volatility (24h rolling)...")
    hourly_returns = df["close"].pct_change()
    features["volatility_24h"] = hourly_returns.rolling(window=24).std()

    # Volume momentum (30-hour)
    logger.info("Computing volume momentum (30h)...")
    features["volume_momentum_30h"] = df["volume"].pct_change(periods=30)

    logger.info(f"Computed {features.shape[1]} hourly features")
    logger.info(f"Feature columns: {list(features.columns)}")

    # Check for NaNs
    nan_counts = features.isna().sum()
    logger.info(f"NaN counts per feature:\n{nan_counts}")

    return features


def resample_to_daily(features_hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly features to daily close.

    Takes the last hourly value of each day (the value at market close).

    Args:
        features_hourly: DataFrame with hourly features

    Returns:
        DataFrame with daily features (resampled from hourly)
    """
    logger.info("Resampling hourly features to daily close...")

    # Resample to daily, taking last value of each day
    features_daily = features_hourly.resample("D").last()

    # Remove timezone info to match existing daily data format
    features_daily.index = features_daily.index.tz_localize(None)

    logger.info(f"Resampled to {len(features_daily)} daily rows")
    logger.info(f"Date range: {features_daily.index.min()} to {features_daily.index.max()}")

    # Check for NaNs after resampling
    nan_counts = features_daily.isna().sum()
    logger.info(f"NaN counts after resampling:\n{nan_counts}")

    return features_daily


def generate_daily_targets(
    prices_hourly: pd.DataFrame,
    horizon_days: int = 1
) -> pd.Series:
    """Generate daily targets from hourly price data.

    Target: Close at T+horizon > Close at T (binary classification)

    Args:
        prices_hourly: Hourly OHLCV DataFrame
        horizon_days: Prediction horizon in days (default: 1 day)

    Returns:
        Series of binary targets (1 = long, 0 = flat) indexed by day
    """
    logger.info(f"Generating daily targets (horizon={horizon_days}d)...")

    # Resample hourly prices to daily close
    prices_daily = prices_hourly.resample("D").last()
    prices_daily.index = prices_daily.index.tz_localize(None)

    # Target: close at T+horizon > close at T
    future_close = prices_daily["close"].shift(-horizon_days)
    current_close = prices_daily["close"]

    targets = (future_close > current_close).astype(int)
    targets = targets.dropna()

    logger.info(f"Generated {len(targets)} daily targets")
    logger.info(f"Target distribution: {targets.value_counts().to_dict()}")
    logger.info(f"Balance: {targets.mean():.1%} positive (long)")

    return targets


def main():
    """Main execution: prepare hourly feature matrix."""

    logger.info("=" * 80)
    logger.info("PREPARE HOURLY FEATURE MATRIX")
    logger.info("=" * 80)

    # Load hourly data
    df_hourly = load_hourly_data()

    # Compute hourly features
    features_hourly = compute_hourly_features(df_hourly)

    # Resample to daily close
    features_daily = resample_to_daily(features_hourly)

    # Generate daily targets (1-day horizon)
    targets_1d = generate_daily_targets(df_hourly, horizon_days=1)

    # Align features and targets
    common_dates = features_daily.index.intersection(targets_1d.index)
    features_aligned = features_daily.loc[common_dates]
    targets_aligned = targets_1d.loc[common_dates]

    logger.info(f"Aligned {len(features_aligned)} samples (features + targets)")

    # Save feature matrix
    output_path = Path("data/processed/feature_matrix_btc_hourly.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_aligned.to_parquet(output_path)
    logger.info(f"Saved feature matrix to {output_path}")

    # Save targets
    targets_path = Path("data/processed/targets_btc_hourly_1d.parquet")
    targets_df = pd.DataFrame({"target": targets_aligned})
    targets_df.to_parquet(targets_path)
    logger.info(f"Saved targets to {targets_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS")
    logger.info("=" * 80)
    logger.info(f"Feature matrix: {features_aligned.shape[0]:,} rows × {features_aligned.shape[1]} features")
    logger.info(f"Hourly samples: {len(df_hourly):,} (underlying data)")
    logger.info(f"Daily samples: {len(features_aligned):,} (for model training)")
    logger.info(f"Sample increase: {len(df_hourly) / len(features_aligned):.1f}x more hourly data")
    logger.info(f"Feature columns: {list(features_aligned.columns)}")
    logger.info(f"Target balance: {targets_aligned.mean():.1%} positive (long)")
    logger.info(f"\nNext step: Run scripts/train_on_hourly.py")


if __name__ == "__main__":
    main()
