#!/usr/bin/env python3
"""Expand feature set from 58 to ~80 features.

New features targeting:
1. Order book proxies (spread estimation, depth imbalance)
2. Longer momentum horizons (72h, 336h)
3. Cross-feature interactions (RSI*volume, volatility*momentum)
4. Mean reversion signals (z-score, deviation from MA)
5. Volatility breakout features
"""
import sys
sys.path.insert(0, "src")

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.features.technical import rsi, ema
from sparky.features.returns import simple_returns

def sma(series, period):
    """Simple moving average."""
    return series.rolling(window=period).mean()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_additional_features(df: pd.DataFrame, features_base: pd.DataFrame) -> pd.DataFrame:
    """Compute 20-30 additional features."""
    logger.info("Computing 20-30 additional features...")
    
    features = pd.DataFrame(index=df.index)
    hourly_returns = simple_returns(df["close"])
    
    # === ORDER BOOK PROXIES (5 features) ===
    logger.info("Order book proxies...")
    features["hl_spread_pct"] = (df["high"] - df["low"]) / df["close"]
    features["hl_spread_ma_ratio"] = features["hl_spread_pct"] / features["hl_spread_pct"].rolling(20).mean()
    features["price_range_expansion"] = (
        (df["high"] - df["low"]) / (df["high"].rolling(20).mean() - df["low"].rolling(20).mean())
    )
    features["mid_vs_close"] = ((df["high"] + df["low"]) / 2 - df["close"]) / df["close"]
    features["volume_at_high"] = (df["close"] == df["high"]).astype(int)
    
    # === LONGER MOMENTUM HORIZONS (4 features) ===
    logger.info("Longer momentum horizons...")
    features["momentum_72h"] = (df["close"] / df["close"].shift(72) - 1.0)  # 3-day
    features["momentum_336h"] = (df["close"] / df["close"].shift(336) - 1.0)  # 14-day (2 weeks)
    features["momentum_720h"] = (df["close"] / df["close"].shift(720) - 1.0)  # 30-day
    features["momentum_divergence_72h_336h"] = features["momentum_72h"] - features["momentum_336h"]
    
    # === CROSS-FEATURE INTERACTIONS (6 features) ===
    logger.info("Cross-feature interactions...")
    rsi_14 = features_base["rsi_14h"]
    volume_ma_20 = df["volume"].rolling(20).mean()
    features["rsi_volume_interaction"] = rsi_14 * (df["volume"] / volume_ma_20)
    features["volatility_momentum_interaction"] = (
        features_base["realized_vol_24h"] * features_base["momentum_24h"]
    )
    features["high_vol_low_momentum"] = (
        (features_base["realized_vol_24h"] > features_base["realized_vol_24h"].rolling(50).quantile(0.75))
        & (features_base["momentum_24h"] < 0)
    ).astype(int)
    features["low_vol_high_momentum"] = (
        (features_base["realized_vol_24h"] < features_base["realized_vol_24h"].rolling(50).quantile(0.25))
        & (features_base["momentum_24h"] > 0)
    ).astype(int)
    features["rsi_divergence_volume_surge"] = (
        (features_base["rsi_14h"] < 30) & (df["volume"] > 2 * volume_ma_20)
    ).astype(int)
    features["rsi_extreme_high_volume"] = (
        (features_base["rsi_14h"] > 70) & (df["volume"] > 2 * volume_ma_20)
    ).astype(int)
    
    # === MEAN REVERSION SIGNALS (5 features) ===
    logger.info("Mean reversion signals...")
    sma_50 = sma(df["close"], period=50)
    sma_200 = sma(df["close"], period=200)
    features["zscore_vs_sma50"] = (df["close"] - sma_50) / df["close"].rolling(50).std()
    features["zscore_vs_sma200"] = (df["close"] - sma_200) / df["close"].rolling(200).std()
    features["sma_crossover_50_200"] = ((sma_50 > sma_200).astype(int).diff() == 1).astype(int)
    features["sma_crossunder_50_200"] = ((sma_50 < sma_200).astype(int).diff() == 1).astype(int)
    features["deviation_from_sma50_pct"] = (df["close"] - sma_50) / sma_50
    
    # === VOLATILITY BREAKOUT FEATURES (4 features) ===
    logger.info("Volatility breakout features...")
    vol_24h = hourly_returns.rolling(24).std()
    vol_168h = hourly_returns.rolling(168).std()
    features["vol_breakout_vs_24h"] = (vol_24h > 1.5 * vol_24h.rolling(50).mean()).astype(int)
    features["vol_contraction_vs_168h"] = (vol_168h < 0.5 * vol_168h.rolling(50).mean()).astype(int)
    features["vol_ratio_24h_168h"] = vol_24h / vol_168h
    features["vol_acceleration"] = vol_24h.diff()
    
    # === PRICE PATTERN FEATURES (3 features) ===
    logger.info("Price pattern features...")
    features["close_above_open_ratio_20h"] = (
        (df["close"] > df["open"]).rolling(20).sum() / 20
    )
    features["higher_close_streak"] = (
        (df["close"] > df["close"].shift(1)).astype(int).rolling(10).sum()
    )
    features["lower_close_streak"] = (
        (df["close"] < df["close"].shift(1)).astype(int).rolling(10).sum()
    )
    
    # === VOLUME PATTERN FEATURES (3 features) ===
    logger.info("Volume pattern features...")
    features["volume_trend_24h"] = (
        df["volume"].rolling(24).mean() / df["volume"].rolling(168).mean()
    )
    features["volume_spike_count_20h"] = (
        (df["volume"] > 2 * volume_ma_20).rolling(20).sum()
    )
    features["volume_dry_up"] = (
        df["volume"] < 0.5 * volume_ma_20
    ).astype(int)
    
    logger.info(f"Computed {features.shape[1]} additional features")
    
    return features


def main():
    """Load hourly data, compute base + additional features, save."""
    # Load hourly data
    hourly_path = Path("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    if not hourly_path.exists():
        raise FileNotFoundError(f"{hourly_path} not found. Run fetch script first.")
    
    logger.info(f"Loading {hourly_path}")
    df = pd.read_parquet(hourly_path)
    logger.info(f"Loaded {len(df):,} hourly candles ({df.index.min()} to {df.index.max()})")
    
    # Load existing base features (58 features)
    base_path = Path("data/processed/feature_matrix_btc_hourly.parquet")
    if not base_path.exists():
        logger.error(f"{base_path} not found. Run prepare_hourly_features.py first.")
        sys.exit(1)
    
    logger.info(f"Loading base features from {base_path}")
    features_base = pd.read_parquet(base_path)
    logger.info(f"Base features: {features_base.shape} ({features_base.shape[1]} features)")
    
    # Compute additional features
    features_new = compute_additional_features(df, features_base)
    
    # Merge base + new
    features_combined = pd.concat([features_base, features_new], axis=1)
    features_combined = features_combined.loc[features_base.index]  # Align to base index
    
    logger.info(f"Combined features: {features_combined.shape} ({features_combined.shape[1]} features)")
    
    # Drop rows with NaN
    features_combined = features_combined.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(f"After cleaning: {features_combined.shape}")
    
    # Save
    output_path = Path("data/processed/feature_matrix_btc_hourly_expanded.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_combined.to_parquet(output_path)
    logger.info(f"Saved to {output_path}")
    
    # Print summary
    logger.info(f"\nFinal feature count: {features_combined.shape[1]}")
    logger.info(f"Sample count: {features_combined.shape[0]}")
    logger.info(f"Date range: {features_combined.index.min()} to {features_combined.index.max()}")


if __name__ == "__main__":
    main()
