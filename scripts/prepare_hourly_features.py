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
    # Use max coverage file (115K candles, 2013-2026) if available
    max_coverage_path = Path("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    standard_path = Path("data/raw/btc/ohlcv_hourly.parquet")

    if max_coverage_path.exists():
        input_path = max_coverage_path
        logger.info("Using maximum coverage dataset (multi-exchange)")
    else:
        input_path = standard_path
        logger.info("Using standard dataset (single exchange)")

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
    """Compute comprehensive technical features on hourly data.

    Feature categories:
    1. Technical indicators (RSI, MACD, Bollinger Bands)
    2. Volatility measures (ATR, BB bandwidth, vol clustering)
    3. Volume features (vol momentum, VWAP deviation, vol ratio)
    4. Market microstructure (intraday range, price patterns)
    5. Multi-timeframe (4h, 24h, 200h trends)
    6. Temporal (hour of day, day of week for session effects)

    Total: ~25 high-quality features designed for hourly crypto data

    Args:
        df: Hourly OHLCV DataFrame

    Returns:
        DataFrame with hourly features
    """
    from sparky.features.technical import rsi, ema, macd, momentum
    from sparky.features.returns import simple_returns
    from sparky.features.advanced import (
        bollinger_bandwidth,
        bollinger_position,
        atr,
        intraday_range,
        volume_momentum,
        volume_ma_ratio,
        vwap_deviation,
        higher_highs_lower_lows,
        volatility_clustering,
        price_distance_from_sma,
        momentum_quality,
        session_hour,
        day_of_week,
        price_acceleration,
    )

    logger.info("Computing comprehensive hourly features (25+ features)...")

    features = pd.DataFrame(index=df.index)

    # === TECHNICAL INDICATORS ===
    logger.info("Computing technical indicators...")

    # RSI (multiple timeframes)
    features["rsi_14h"] = rsi(df["close"], period=14)  # Standard RSI
    features["rsi_6h"] = rsi(df["close"], period=6)    # Fast RSI (oversold/overbought)

    # MACD (12/26/9 hourly)
    macd_line, signal_line, histogram = macd(df["close"], fast_period=12, slow_period=26, signal_period=9)
    features["macd_line"] = macd_line
    features["macd_histogram"] = histogram

    # EMA ratios (trend strength)
    ema_fast = ema(df["close"], span=10)   # 10-hour EMA
    ema_slow = ema(df["close"], span=20)   # 20-hour EMA
    features["ema_ratio_20h"] = ema_fast / ema_slow - 1.0

    # Bollinger Bands
    features["bb_bandwidth_20h"] = bollinger_bandwidth(df["close"], period=20)
    features["bb_position_20h"] = bollinger_position(df["close"], period=20)

    # === MOMENTUM FEATURES ===
    logger.info("Computing momentum features...")

    features["momentum_4h"] = momentum(df["close"], period=4)    # 4-hour momentum (short-term)
    features["momentum_24h"] = momentum(df["close"], period=24)  # Daily momentum
    features["momentum_168h"] = momentum(df["close"], period=168)  # Weekly momentum (7 days)

    # Momentum quality (consistency)
    features["momentum_quality_30h"] = momentum_quality(df["close"], period=30)

    # Price acceleration (momentum of momentum)
    features["price_acceleration_10h"] = price_acceleration(df["close"], period=10)

    # === VOLATILITY FEATURES ===
    logger.info("Computing volatility features...")

    features["atr_14h"] = atr(df["high"], df["low"], df["close"], period=14)
    features["intraday_range"] = intraday_range(df["high"], df["low"], df["close"])

    # Volatility clustering (ARCH effect)
    hourly_returns = simple_returns(df["close"])
    features["vol_clustering_24h"] = volatility_clustering(hourly_returns, period=24)

    # Realized volatility (rolling std)
    features["realized_vol_24h"] = hourly_returns.rolling(window=24).std()

    # === VOLUME FEATURES ===
    logger.info("Computing volume features...")

    features["volume_momentum_30h"] = volume_momentum(df["volume"], period=30)
    features["volume_ma_ratio_20h"] = volume_ma_ratio(df["volume"], period=20)
    features["vwap_deviation_24h"] = vwap_deviation(df["high"], df["low"], df["close"], df["volume"], period=24)

    # === MARKET MICROSTRUCTURE ===
    logger.info("Computing market microstructure features...")

    features["higher_highs_lower_lows_5h"] = higher_highs_lower_lows(df["high"], df["low"], period=5)

    # Price distance from long-term trend
    features["distance_from_sma_200h"] = price_distance_from_sma(df["close"], period=200)  # ~8-day MA

    # === TEMPORAL FEATURES ===
    logger.info("Computing temporal features...")

    features["hour_of_day"] = session_hour(df.index)
    features["day_of_week"] = day_of_week(df.index)

    logger.info(f"Computed {features.shape[1]} hourly features")
    logger.info(f"Feature columns:\n{list(features.columns)}")

    # Check for NaNs
    nan_counts = features.isna().sum()
    logger.info(f"\nNaN counts per feature:")
    for col in features.columns:
        if nan_counts[col] > 0:
            logger.info(f"  {col}: {nan_counts[col]} ({100*nan_counts[col]/len(features):.1f}%)")

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

    # Ensure timezone-aware (UTC) — do NOT strip timezone
    if features_daily.index.tz is None:
        features_daily.index = features_daily.index.tz_localize("UTC")

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
    if prices_daily.index.tz is None:
        prices_daily.index = prices_daily.index.tz_localize("UTC")

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
