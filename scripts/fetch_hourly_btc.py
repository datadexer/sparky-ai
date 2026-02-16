#!/usr/bin/env python3
"""Fetch Hourly BTC OHLCV Data (2017-2025)

CRITICAL MISSION: Expand training data to 10,000+ observations.
This script fetches HOURLY data instead of daily for 24x sample increase.

Current: 2,178 daily samples (insufficient for deep learning)
Target: ~70,000 hourly samples (2017-2025)
Expected: 52,272 hourly candles for 2019-2025 period

Strategy:
- Train on hourly features (RSI-14h, Momentum-30h, EMA-ratio-20h)
- Predict daily direction (hourly features at day T close → daily target)
- 24x more data → enables LSTM, deep architectures

Data source: CCXT (Binance) with failover to Bybit/OKX
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_hourly_ohlcv(
    symbol: str = "BTC/USDT",
    start_date: str = "2019-01-01",  # Most exchanges have reliable data from 2019
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Fetch hourly OHLCV data via CCXT.

    Args:
        symbol: Trading pair (default: BTC/USDT)
        start_date: Start date YYYY-MM-DD (default: 2017-01-01)
        end_date: End date YYYY-MM-DD (default: 2025-12-31)

    Returns:
        DataFrame with hourly OHLCV data, DatetimeIndex (UTC)
    """
    from sparky.data.price import CCXTPriceFetcher

    logger.info("=" * 80)
    logger.info("FETCH HOURLY BTC OHLCV DATA")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Timeframe: 1 hour (1h)")
    logger.info("=" * 80)

    # Create CCXT fetcher - try multiple exchanges (Binance may be geo-blocked)
    exchanges_to_try = ["bybit", "okx", "kraken", "coinbase"]

    fetcher = None
    successful_exchange = None

    for exchange_id in exchanges_to_try:
        try:
            logger.info(f"Trying exchange: {exchange_id}...")
            fetcher = CCXTPriceFetcher(exchange_id=exchange_id)

            # Test with a small fetch to verify it works
            test_candles = fetcher.exchange.fetch_ohlcv(symbol, "1h", limit=1)
            if test_candles:
                logger.info(f"✓ {exchange_id} works! Using this exchange.")
                successful_exchange = exchange_id
                break
        except Exception as e:
            logger.warning(f"✗ {exchange_id} failed: {e}")
            continue

    if fetcher is None or successful_exchange is None:
        raise RuntimeError("All exchanges failed! Cannot fetch data.")

    logger.info(f"Using exchange: {successful_exchange}")

    # CCXT fetch_ohlcv method needs modification for hourly data
    # The existing fetch_daily_ohlcv uses "1d" timeframe hardcoded
    # We need to use the raw fetch_ohlcv with "1h" timeframe

    start_ts = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_ts = int(
        datetime.strptime(end_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )

    logger.info(f"Fetching hourly candles from {start_date} to {end_date}...")
    logger.info("This may take several minutes due to pagination (CCXT limit: 1000 candles/request)")

    # Paginated fetch for hourly data
    all_candles = []
    since = start_ts
    ms_per_hour = 3_600_000  # 1 hour in milliseconds

    batch_num = 0
    while since < end_ts:
        batch_num += 1

        try:
            # Fetch up to 1000 hourly candles
            candles = fetcher.exchange.fetch_ohlcv(
                symbol, "1h", since=since, limit=1000
            )

            if not candles:
                logger.warning(f"No candles returned at timestamp {since}")
                break

            all_candles.extend(candles)

            # Log progress every 10 batches
            if batch_num % 10 == 0:
                current_date = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
                logger.info(
                    f"Fetched batch {batch_num}: {len(all_candles)} total candles "
                    f"(current: {current_date.date()})"
                )

            # Move to after the last candle
            last_ts = candles[-1][0]
            if last_ts <= since:
                logger.warning("Last timestamp <= since, breaking to prevent infinite loop")
                break
            since = last_ts + ms_per_hour

        except Exception as e:
            logger.error(f"Error fetching batch {batch_num}: {e}")
            break

    logger.info(f"Fetched {len(all_candles)} total hourly candles in {batch_num} batches")

    if not all_candles:
        logger.error("No candles fetched!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Remove duplicates
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < before:
        logger.warning(f"Removed {before - len(df)} duplicate timestamps")

    # Sort by timestamp
    df = df.sort_index()

    # Validate data quality
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        invalid = df[col] <= 0
        if invalid.any():
            logger.warning(f"{invalid.sum()} non-positive {col} values (removing)")
            df = df[~invalid]

    invalid_vol = df["volume"] < 0
    if invalid_vol.any():
        logger.warning(f"{invalid_vol.sum()} negative volume values (removing)")
        df = df[~invalid_vol]

    logger.info("=" * 80)
    logger.info("HOURLY DATA STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Total hourly candles: {len(df):,}")
    logger.info(f"Days covered: {(df.index.max() - df.index.min()).days}")
    logger.info(f"Expected daily samples: {len(df) // 24:,}")
    logger.info(f"Sample increase vs daily: {len(df) / (len(df) // 24):.1f}x")

    return df


def main():
    """Main execution: fetch hourly BTC data and save."""

    # Fetch hourly data
    df_hourly = fetch_hourly_ohlcv(
        symbol="BTC/USDT",
        start_date="2019-01-01",  # Most exchanges have reliable data from 2019
        end_date="2025-12-31"
    )

    if df_hourly.empty:
        logger.error("Failed to fetch hourly data")
        return

    # Save to Parquet
    output_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_hourly.to_parquet(output_path)
    logger.info(f"Saved hourly data to {output_path}")

    # Update data manifest
    try:
        from sparky.data.storage import DataStore
        store = DataStore()
        store.update_manifest(output_path, df_hourly)
        logger.info("Updated data manifest with SHA-256 hash")
    except Exception as e:
        logger.warning(f"Could not update manifest: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS")
    logger.info("=" * 80)
    logger.info(f"Fetched {len(df_hourly):,} hourly candles for BTC")
    logger.info(f"Date range: {df_hourly.index.min().date()} to {df_hourly.index.max().date()}")
    logger.info(f"File: {output_path}")
    logger.info(f"Next step: Run scripts/prepare_hourly_features.py")


if __name__ == "__main__":
    main()
