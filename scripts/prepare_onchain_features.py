#!/usr/bin/env python3
"""
On-chain feature engineering for BTC.

Combines CoinMetrics and Blockchain.com daily on-chain data, computes
derived features, and aligns to hourly frequency with proper look-ahead
bias prevention.

Features:
- mvrv_ratio: Market value to realized value ratio (already computed by CoinMetrics)
- mvrv_zscore: Z-score of MVRV over rolling 365-day window
- active_addresses_change_7d: 7-day percent change in active addresses
- hash_rate_change_30d: 30-day percent change in hash rate
- exchange_net_flow_7d: 7-day rolling sum of net exchange flow (in - out)
- fee_ratio_change_7d: 7-day percent change in total fees
- tx_count_change_7d: 7-day percent change in transaction count
- nvt_ratio: Network Value to Transactions ratio (market cap / transaction volume)

Look-ahead bias prevention:
- All daily features are shifted by 1 day before forward-filling to hourly
- This ensures that on-chain data published on day T is only used for predictions
  starting on day T+1, reflecting real-world publication delays
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [ONCHAIN_FEATURES] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def load_data():
    """Load CoinMetrics and BTC hourly data."""
    logging.info("Loading CoinMetrics daily data...")
    cm = pd.read_parquet("data/raw/onchain/coinmetrics_btc_daily.parquet")

    logging.info("Loading BTC hourly OHLCV data...")
    btc_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")

    logging.info(f"CoinMetrics: {len(cm)} days ({cm.index.min()} to {cm.index.max()})")
    logging.info(f"BTC hourly: {len(btc_hourly)} hours ({btc_hourly.index.min()} to {btc_hourly.index.max()})")

    return cm, btc_hourly


def compute_daily_features(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute on-chain features from daily CoinMetrics data.

    Args:
        cm: CoinMetrics daily data with columns:
            - CapMVRVCur: MVRV ratio (market cap / realized cap)
            - AdrActCnt: Active addresses count
            - HashRate: Network hash rate
            - FlowInExNtv: Exchange inflow (BTC)
            - FlowOutExNtv: Exchange outflow (BTC)
            - FeeTotNtv: Total fees (BTC)
            - TxCnt: Transaction count
            - CapMrktCurUSD: Market capitalization (USD)
            - PriceUSD: BTC price (USD)

    Returns:
        DataFrame with 8 on-chain features, same index as input
    """
    logging.info("Computing on-chain features...")

    features = pd.DataFrame(index=cm.index)

    # 1. MVRV ratio (already computed by CoinMetrics)
    features["mvrv_ratio"] = cm["CapMVRVCur"]

    # 2. MVRV Z-score (365-day rolling window)
    # Z = (MVRV - rolling_mean) / rolling_std
    mvrv_mean = cm["CapMVRVCur"].rolling(window=365, min_periods=30).mean()
    mvrv_std = cm["CapMVRVCur"].rolling(window=365, min_periods=30).std()
    features["mvrv_zscore"] = (cm["CapMVRVCur"] - mvrv_mean) / mvrv_std

    # 3. Active addresses 7-day percent change
    # pct_change = (current - previous) / previous
    features["active_addresses_change_7d"] = cm["AdrActCnt"].pct_change(periods=7)

    # 4. Hash rate 30-day percent change
    features["hash_rate_change_30d"] = cm["HashRate"].pct_change(periods=30)

    # 5. Exchange net flow 7-day rolling sum
    # Net flow = inflow - outflow (positive = net inflow to exchanges)
    net_flow = cm["FlowInExNtv"] - cm["FlowOutExNtv"]
    features["exchange_net_flow_7d"] = net_flow.rolling(window=7).sum()

    # 6. Fee ratio 7-day percent change
    features["fee_ratio_change_7d"] = cm["FeeTotNtv"].pct_change(periods=7)

    # 7. Transaction count 7-day percent change
    features["tx_count_change_7d"] = cm["TxCnt"].pct_change(periods=7)

    # 8. NVT ratio (Network Value to Transactions)
    # NVT = Market Cap / Transaction Volume
    # Transaction Volume ≈ TxCnt * PriceUSD (approximation)
    tx_volume = cm["TxCnt"] * cm["PriceUSD"]
    features["nvt_ratio"] = cm["CapMrktCurUSD"] / tx_volume
    # Replace inf/NaN from division by zero
    features["nvt_ratio"] = features["nvt_ratio"].replace([np.inf, -np.inf], np.nan)

    logging.info(f"Computed {len(features.columns)} features")
    logging.info("NaN counts per feature:")
    for col in features.columns:
        nan_count = features[col].isna().sum()
        logging.info(f"  {col}: {nan_count} ({100 * nan_count / len(features):.1f}%)")

    return features


def align_to_hourly(daily_features: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align daily features to hourly frequency with look-ahead bias prevention.

    Strategy:
    1. Shift daily features by 1 day (on-chain data published with delay)
    2. Reindex to hourly frequency using forward-fill
    3. Each hour gets the most recent available daily feature value

    Args:
        daily_features: Daily feature DataFrame
        hourly_index: Target hourly DatetimeIndex

    Returns:
        Hourly feature DataFrame
    """
    logging.info("Aligning daily features to hourly frequency...")

    # CRITICAL: Shift by 1 day to prevent look-ahead bias
    # On-chain data for day T is only available at the end of day T,
    # so we can only use it for predictions starting on day T+1
    logging.info("Shifting daily features by 1 day to prevent look-ahead bias...")
    daily_shifted = daily_features.shift(1)

    # Reindex to hourly frequency with forward-fill
    # Each hour in a day gets the previous day's feature values
    logging.info(f"Reindexing from {len(daily_shifted)} daily rows to {len(hourly_index)} hourly rows...")
    hourly_features = daily_shifted.reindex(hourly_index, method="ffill")

    logging.info(f"Hourly features: {hourly_features.shape}")
    logging.info(f"Date range: {hourly_features.index.min()} to {hourly_features.index.max()}")

    # Report NaN counts after alignment
    logging.info("NaN counts after hourly alignment:")
    for col in hourly_features.columns:
        nan_count = hourly_features[col].isna().sum()
        logging.info(f"  {col}: {nan_count} ({100 * nan_count / len(hourly_features):.1f}%)")

    return hourly_features


def main():
    """Main pipeline: load data, compute features, align to hourly, save."""
    # Load data
    cm, btc_hourly = load_data()

    # Compute daily features
    daily_features = compute_daily_features(cm)

    # Align to hourly frequency
    hourly_features = align_to_hourly(daily_features, btc_hourly.index)

    # Save to processed directory
    output_path = Path("data/processed/onchain_features_hourly.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving to {output_path}...")
    hourly_features.to_parquet(output_path)

    # Final summary
    logging.info("=" * 60)
    logging.info("On-chain feature engineering complete!")
    logging.info(f"Output: {output_path}")
    logging.info(f"Shape: {hourly_features.shape}")
    logging.info(f"Features: {list(hourly_features.columns)}")
    logging.info(f"Date range: {hourly_features.index.min()} to {hourly_features.index.max()}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
