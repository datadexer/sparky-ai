#!/usr/bin/env python3
"""Prepare feature matrices and targets for Phase 3 experiments.

Loads BTC/ETH price data and on-chain data, builds comprehensive feature
matrices using FeatureRegistry, generates target variables for all prediction
horizons, and splits data for fair baseline comparison.

Output:
    - data/processed/feature_matrix_btc.parquet
    - data/processed/targets_btc_{horizon}d.parquet (for horizons 1, 3, 7, 14, 30)
    - Logged statistics to roadmap/RESEARCH_LOG.md
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sparky.data.storage import DataStore
from sparky.features.onchain import (
    address_momentum,
    hash_ribbon,
    volume_momentum,
)
from sparky.features.registry import FeatureDefinition, FeatureRegistry
from sparky.features.technical import ema, momentum, rsi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Period definitions (match baseline for fair comparison)
FULL_START = "2019-01-01"
FULL_END = "2025-12-31"
IS_START = "2019-01-01"
IS_END = "2022-01-01"
OOS_END = "2025-09-30"  # Reserve final 3 months as holdout
HOLDOUT_START = "2025-10-01"
HOLDOUT_END = "2025-12-31"

# Prediction horizons (in days)
HORIZONS = [1, 3, 7, 14, 30]


def register_all_features(registry: FeatureRegistry) -> None:
    """Register all technical and on-chain features."""

    # Technical features (apply to all assets)
    registry.register(
        FeatureDefinition(
            name="rsi_14",
            category="technical",
            compute_fn=lambda data: rsi(data["price"]["close"], period=14),
            input_columns=["close"],
            lookback=14,
            data_source="binance",
            expected_range=(0, 100),
            description="14-day RSI using Wilder's smoothing",
            asset="all",
        )
    )

    registry.register(
        FeatureDefinition(
            name="momentum_30d",
            category="technical",
            compute_fn=lambda data: momentum(data["price"]["close"], period=30),
            input_columns=["close"],
            lookback=30,
            data_source="binance",
            description="30-day price momentum (pct change)",
            asset="all",
        )
    )

    registry.register(
        FeatureDefinition(
            name="ema_ratio_20d",
            category="technical",
            compute_fn=lambda data: data["price"]["close"] / ema(data["price"]["close"], 20) - 1,
            input_columns=["close"],
            lookback=20,
            data_source="binance",
            description="Price vs 20-day EMA ratio (deviation)",
            asset="all",
        )
    )

    # Return features REMOVED — Feature ablation experiments showed returns_1d
    # caused data leakage (shuffled_label test failed) and removing it IMPROVED
    # Sharpe by +0.48. The subtle overlap between close_T (used in returns_1d)
    # and the target (which compares close_T+N to open_T+1) created leakage.
    # See scripts/debug_leakage.py and scripts/test_simplified_model.py for details.
    #
    # registry.register(
    #     FeatureDefinition(
    #         name="returns_1d",
    #         category="returns",
    #         compute_fn=lambda data: simple_returns(data["price"]["close"]),
    #         input_columns=["close"],
    #         lookback=1,
    #         data_source="binance",
    #         description="1-day simple returns [REMOVED DUE TO LEAKAGE]",
    #         asset="all",
    #     )
    # )

    # BTC on-chain features (only available from blockchain.com data, starting 2021)
    registry.register(
        FeatureDefinition(
            name="hash_ribbon_btc",
            category="onchain_btc",
            compute_fn=lambda data: (
                hash_ribbon(data["onchain"]["hash_rate"], short=30, long=60)
                if "onchain" in data and "hash_rate" in data["onchain"].columns
                else pd.Series(dtype=float)
            ),
            input_columns=["hash_rate"],
            lookback=60,
            data_source="blockchain_com",
            valid_from="2021-02-17",  # Data starts here
            description="Hash ribbon signal (miner health)",
            asset="btc",
        )
    )

    registry.register(
        FeatureDefinition(
            name="address_momentum_btc",
            category="onchain_btc",
            compute_fn=lambda data: (
                address_momentum(data["onchain"]["active_addresses"], period=30)
                if "onchain" in data and "active_addresses" in data["onchain"].columns
                else pd.Series(dtype=float)
            ),
            input_columns=["active_addresses"],
            lookback=30,
            data_source="blockchain_com",
            valid_from="2021-02-17",
            description="30-day active address momentum",
            asset="btc",
        )
    )

    registry.register(
        FeatureDefinition(
            name="volume_momentum_btc",
            category="onchain_btc",
            compute_fn=lambda data: (
                volume_momentum(data["onchain"]["transfer_volume_usd"], period=30)
                if "onchain" in data and "transfer_volume_usd" in data["onchain"].columns
                else pd.Series(dtype=float)
            ),
            input_columns=["transfer_volume_usd"],
            lookback=30,
            data_source="blockchain_com",
            valid_from="2021-02-17",
            description="30-day on-chain volume momentum",
            asset="btc",
        )
    )

    logger.info(f"Registered {len(registry.list_features())} features")


def generate_targets(prices: pd.Series, horizons: list[int]) -> dict[int, pd.Series]:
    """Generate target variables for all prediction horizons.

    Target timing (CRITICAL from SONNET_HANDOFF.md):
        Day T close    → Features computed
        Day T+1 open   → Trade EXECUTES
        Day T+1+N close → Target: close_{T+1+N} > open_{T+1}

    Args:
        prices: Daily OHLC DataFrame with 'open' and 'close' columns.
        horizons: List of prediction horizons in days.

    Returns:
        Dict mapping horizon to target Series (1=long, 0=flat).
    """
    targets = {}

    # Need both open and close prices
    if isinstance(prices, pd.Series):
        # If only close prices, approximate open as previous close
        close = prices
        open_prices = close.shift(1)
        logger.warning("Only close prices provided, approximating open as previous close")
    else:
        close = prices["close"]
        open_prices = prices["open"]

    for horizon in horizons:
        # Signal generated at T close → executed at T+1 open → target at T+1+N close
        # Shift open by -1 to get "next day's open"
        next_open = open_prices.shift(-1)
        # Shift close by -(1+horizon) to get "close at T+1+N"
        target_close = close.shift(-(1 + horizon))

        # Target: 1 if target_close > next_open, else 0
        target = (target_close > next_open).astype(int)

        # Drop NaN rows (at the end where we don't have future prices)
        target = target.dropna()

        targets[horizon] = target
        logger.info(
            f"Generated target for {horizon}d horizon: "
            f"{len(target)} samples, {target.sum()} longs ({target.mean() * 100:.1f}%)"
        )

    return targets


def main():
    logger.info("=" * 60)
    logger.info("Phase 3 Data Preparation")
    logger.info("=" * 60)

    # 1. Load BTC price data
    logger.info("Loading BTC OHLCV data...")
    store = DataStore()
    btc_ohlcv, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    logger.info(
        f"Loaded {len(btc_ohlcv)} rows, range: {btc_ohlcv.index.min().date()} to {btc_ohlcv.index.max().date()}"
    )

    # Filter to full period
    btc_ohlcv = btc_ohlcv.loc[FULL_START:FULL_END]
    logger.info(f"Filtered to {len(btc_ohlcv)} rows ({FULL_START} to {FULL_END})")

    # 2. Load BTC on-chain data
    logger.info("Loading BTC on-chain data...")
    onchain_path = Path("data/raw/btc/onchain_blockchain_com.parquet")
    if onchain_path.exists():
        btc_onchain, _ = store.load(onchain_path)
        logger.info(
            f"Loaded {len(btc_onchain)} on-chain rows, "
            f"range: {btc_onchain.index.min().date()} to {btc_onchain.index.max().date()}"
        )

        # Resample to daily (currently hourly) using last value of each day
        btc_onchain_daily = btc_onchain.resample("D").last()
        # Align with price data index
        btc_onchain_daily = btc_onchain_daily.reindex(btc_ohlcv.index, method=None)
        logger.info(f"Resampled to daily: {len(btc_onchain_daily)} rows")
    else:
        logger.warning("No on-chain data found, creating empty DataFrame")
        btc_onchain_daily = pd.DataFrame(index=btc_ohlcv.index)

    # 3. Register all features
    logger.info("Registering features...")
    registry = FeatureRegistry()
    register_all_features(registry)

    # 4. Build feature matrix for BTC
    logger.info("Building feature matrix for BTC...")
    data = {"price": btc_ohlcv, "onchain": btc_onchain_daily}

    feature_names = registry.list_features(asset="btc")
    logger.info(f"Building {len(feature_names)} features for BTC")

    X_btc = registry.build_feature_matrix(asset="btc", feature_names=feature_names, data=data, drop_na_rows=True)

    logger.info(f"Feature matrix shape: {X_btc.shape}")
    logger.info(f"Date range: {X_btc.index.min().date()} to {X_btc.index.max().date()}")
    logger.info(f"Features: {list(X_btc.columns)}")

    # Check for NaN patterns
    nan_counts = X_btc.isna().sum()
    if nan_counts.sum() > 0:
        logger.info("NaN counts per feature:")
        for feat, count in nan_counts[nan_counts > 0].items():
            pct = count / len(X_btc) * 100
            logger.info(f"  {feat}: {count} ({pct:.1f}%)")

    # 5. Generate targets for all horizons
    logger.info("Generating target variables...")
    targets = generate_targets(btc_ohlcv[["open", "close"]], HORIZONS)

    # 6. Align targets with features
    logger.info("Aligning targets with features...")
    aligned_targets = {}
    for horizon, target in targets.items():
        # Keep only rows that exist in both features and target
        common_index = X_btc.index.intersection(target.index)
        aligned_targets[horizon] = target.loc[common_index]
        logger.info(
            f"  {horizon}d: {len(aligned_targets[horizon])} samples "
            f"(longs: {aligned_targets[horizon].sum()}, "
            f"{aligned_targets[horizon].mean() * 100:.1f}%)"
        )

    # Also align features to smallest common index across all horizons
    all_indices = [t.index for t in aligned_targets.values()]
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.intersection(idx)

    X_btc_aligned = X_btc.loc[common_index]
    logger.info(f"Feature matrix after alignment: {X_btc_aligned.shape}")

    # 7. Data splits
    logger.info("Data splits:")
    in_sample = X_btc_aligned.loc[IS_START:IS_END]
    out_sample = X_btc_aligned.loc[IS_END:OOS_END]
    holdout = X_btc_aligned.loc[HOLDOUT_START:HOLDOUT_END]

    logger.info(f"  In-sample (train): {len(in_sample)} rows ({IS_START} to {IS_END})")
    logger.info(f"  Out-of-sample (test): {len(out_sample)} rows ({IS_END} to {OOS_END})")
    logger.info(f"  Holdout (never-touched): {len(holdout)} rows ({HOLDOUT_START} to {HOLDOUT_END})")

    # 8. Save feature matrix
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "feature_matrix_btc.parquet"
    X_btc_aligned.to_parquet(feature_path)
    logger.info(f"Saved feature matrix to {feature_path}")

    # 9. Save targets for all horizons
    for horizon, target in aligned_targets.items():
        target_path = output_dir / f"targets_btc_{horizon}d.parquet"
        target_aligned = target.loc[common_index]
        target_aligned.to_frame("target").to_parquet(target_path)
        logger.info(f"Saved {horizon}d target to {target_path}")

    # 10. Log statistics to RESEARCH_LOG.md
    logger.info("Logging statistics to RESEARCH_LOG.md...")
    log_entry = f"""
---
## Phase 3 Data Preparation — {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC

**Data Coverage:**
- BTC OHLCV: {btc_ohlcv.index.min().date()} to {btc_ohlcv.index.max().date()} ({len(btc_ohlcv)} rows)
- BTC on-chain: {btc_onchain.index.min().date() if len(btc_onchain) > 0 else "N/A"} to {btc_onchain.index.max().date() if len(btc_onchain) > 0 else "N/A"} ({len(btc_onchain) if len(btc_onchain) > 0 else 0} rows)

**Feature Matrix:**
- Shape: {X_btc_aligned.shape}
- Date range: {X_btc_aligned.index.min().date()} to {X_btc_aligned.index.max().date()}
- Features: {len(X_btc_aligned.columns)} ({", ".join(X_btc_aligned.columns.tolist())})

**Target Distribution:**
{chr(10).join(f"- {h}d horizon: {aligned_targets[h].loc[common_index].sum()} longs / {len(common_index)} total ({aligned_targets[h].loc[common_index].mean() * 100:.1f}%)" for h in HORIZONS)}

**Data Splits (matching baseline for fair comparison):**
- In-sample: {len(in_sample)} rows ({IS_START} to {IS_END})
- Out-of-sample: {len(out_sample)} rows ({IS_END} to {OOS_END})
- Holdout: {len(holdout)} rows ({HOLDOUT_START} to {HOLDOUT_END})

**Status:** ✓ Data preparation complete. Ready for Phase 3 experiments.
"""

    log_path = Path("roadmap/RESEARCH_LOG.md")
    with open(log_path, "a") as f:
        f.write(log_entry)

    logger.info(f"Appended to {log_path}")

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
