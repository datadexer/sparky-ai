#!/usr/bin/env python3
"""
Prepare macro features from daily DXY/Gold/SPX/VIX data and align to hourly frequency.

Features computed:
- DXY: return_1d, return_5d, sma_ratio_20d
- Gold: return_1d, return_5d, sma_ratio_20d
- SPX: return_1d, return_5d, vol_5d
- VIX: level, change_1d, sma_ratio_10d
- Cross-asset: btc_gold_corr_30d

Output: data/processed/macro_features_hourly.parquet
"""

from pathlib import Path

import pandas as pd


def compute_macro_features() -> pd.DataFrame:
    """
    Load daily macro data, compute features, align to hourly frequency.

    Returns:
        DataFrame with hourly index and macro features (forward-filled from daily).
    """
    # Load daily macro data
    dxy = pd.read_parquet("data/raw/macro/dxy_daily.parquet")
    gold = pd.read_parquet("data/raw/macro/gold_daily.parquet")
    spx = pd.read_parquet("data/raw/macro/spx_daily.parquet")
    vix = pd.read_parquet("data/raw/macro/vix_daily.parquet")

    # Load BTC hourly data for correlation and hourly index
    btc = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")

    # Compute BTC daily returns for cross-asset correlation
    # Normalize to date (no time component) for alignment with daily macro data
    btc_daily = btc["close"].resample("D").last()
    btc_daily.index = btc_daily.index.normalize()  # Set to midnight UTC
    btc_daily_return = btc_daily.pct_change()

    # Initialize feature dataframe with daily frequency
    # Use the longest common date range across all assets
    start_date = max(dxy.index.min(), gold.index.min(), spx.index.min(), vix.index.min())
    end_date = min(dxy.index.max(), gold.index.max(), spx.index.max(), vix.index.max())

    # Filter all dataframes to common date range
    dxy = dxy.loc[start_date:end_date]
    gold = gold.loc[start_date:end_date]
    spx = spx.loc[start_date:end_date]
    vix = vix.loc[start_date:end_date]

    # Normalize all daily indices to midnight UTC for alignment
    dxy.index = dxy.index.normalize()
    gold.index = gold.index.normalize()
    spx.index = spx.index.normalize()
    vix.index = vix.index.normalize()

    # Create daily feature dataframe
    daily_features = pd.DataFrame(index=dxy.index)

    # DXY features
    daily_features["dxy_return_1d"] = dxy["close"].pct_change(1)
    daily_features["dxy_return_5d"] = dxy["close"].pct_change(5)
    daily_features["dxy_sma_ratio_20d"] = dxy["close"] / dxy["close"].rolling(20).mean() - 1

    # Gold features
    daily_features["gold_return_1d"] = gold["close"].pct_change(1)
    daily_features["gold_return_5d"] = gold["close"].pct_change(5)
    daily_features["gold_sma_ratio_20d"] = gold["close"] / gold["close"].rolling(20).mean() - 1

    # SPX features
    daily_features["spx_return_1d"] = spx["close"].pct_change(1)
    daily_features["spx_return_5d"] = spx["close"].pct_change(5)
    spx_returns = spx["close"].pct_change(1)
    daily_features["spx_vol_5d"] = spx_returns.rolling(5).std()

    # VIX features
    daily_features["vix_level"] = vix["close"]
    daily_features["vix_change_1d"] = vix["close"].diff(1)
    daily_features["vix_sma_ratio_10d"] = vix["close"] / vix["close"].rolling(10).mean() - 1

    # Cross-asset: BTC-Gold correlation (30-day rolling)
    gold_daily_return = gold["close"].pct_change(1)
    # Align BTC and Gold returns on common dates
    aligned_btc = btc_daily_return.reindex(daily_features.index)
    aligned_gold = gold_daily_return.reindex(daily_features.index)
    daily_features["btc_gold_corr_30d"] = aligned_btc.rolling(30).corr(aligned_gold)

    # CRITICAL: Shift daily features by 1 day to prevent look-ahead bias
    # This ensures that hourly features on day D use macro data from day D-1
    daily_features_shifted = daily_features.shift(1)

    # Create hourly index matching BTC hourly data
    hourly_index = btc.index

    # Forward-fill daily features to hourly frequency
    # Method: reindex to hourly, then forward-fill
    hourly_features = (
        daily_features_shifted.reindex(hourly_index.union(daily_features_shifted.index))
        .sort_index()
        .ffill()
        .reindex(hourly_index)
    )

    return hourly_features


def main():
    """Run macro feature preparation pipeline."""
    print("Loading macro data and computing features...")

    hourly_features = compute_macro_features()

    print(f"Generated hourly features: shape={hourly_features.shape}")
    print(f"Columns: {hourly_features.columns.tolist()}")
    print(f"Date range: {hourly_features.index.min()} to {hourly_features.index.max()}")
    print(f"\nFirst non-null row index: {hourly_features.first_valid_index()}")
    print("NaN counts by column:")
    print(hourly_features.isna().sum())

    # Save to processed directory
    output_path = Path("data/processed/macro_features_hourly.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_features.to_parquet(output_path)

    print(f"\nSaved to {output_path}")
    print("\nSample of features (first 10 non-null rows):")
    first_valid_idx = hourly_features.first_valid_index()
    if first_valid_idx is not None:
        print(hourly_features.loc[first_valid_idx:].head(10))


if __name__ == "__main__":
    main()
