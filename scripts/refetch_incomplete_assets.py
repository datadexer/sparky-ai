#!/usr/bin/env python3
"""Refetch Incomplete Cross-Asset Hourly Data

This script refetches assets that have incomplete coverage:
- SOL: Only has 721 rows (last 30 days), need full history from 2020-04-01
- AVAX: Only has 721 rows, need full history from 2020-09-01
- MATIC: Only has 721 rows, need full history from 2019-04-01
- DOT: Has 30K rows but stops at 2024-01-28, need to extend to 2026-02-16
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Assets that need refetching
ASSETS = [
    {"symbols": ["SOL/USDT", "SOL/USD"], "name": "sol", "start": "2020-04-01"},
    {"symbols": ["AVAX/USDT", "AVAX/USD"], "name": "avax", "start": "2020-09-01"},
    {"symbols": ["POL/USDT", "POL/USD", "MATIC/USDT", "MATIC/USD"], "name": "matic", "start": "2019-04-01"},
    {"symbols": ["DOT/USDT", "DOT/USD"], "name": "dot", "start": "2020-08-01"},
]

FAILOVER_EXCHANGES = ["coinbase", "okx", "bitfinex", "kraken"]


def fetch_asset_hourly(
    symbols: list[str],
    asset_name: str,
    start_date: str = "2020-01-01",
    end_date: str = "2026-02-16",
) -> pd.DataFrame:
    """Fetch hourly OHLCV for a single asset with exchange failover."""

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Try each exchange + symbol combo until one works
    exchange = None
    working_symbol = None
    for ex_id in FAILOVER_EXCHANGES:
        for sym in symbols:
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
                ex.load_markets()
                # Test with since=start_ts to verify historical support
                candles = ex.fetch_ohlcv(sym, "1h", since=start_ts, limit=5)
                if candles:
                    first_ts = candles[0][0]
                    # Accept if first candle is within 2 years of requested start
                    if first_ts < start_ts + 2 * 365 * 24 * 3600 * 1000:
                        exchange = ex
                        working_symbol = sym
                        first_date = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
                        logger.info(f"\n  {asset_name}: Using {ex_id} with {sym} (data from {first_date.date()})")
                        break
            except Exception as e:
                logger.debug(f"  {ex_id}/{sym} failed: {e}")
                continue
        if exchange:
            break

    if not exchange:
        logger.warning(f"  {asset_name}: No working exchange found!")
        return pd.DataFrame()

    # Paginated fetch
    all_candles = []
    since = start_ts
    ms_per_hour = 3_600_000

    batch_num = 0
    consecutive_empty = 0
    while since < end_ts and consecutive_empty < 3:
        batch_num += 1

        try:
            candles = exchange.fetch_ohlcv(working_symbol, "1h", since=since, limit=1000)

            if not candles:
                consecutive_empty += 1
                since += ms_per_hour * 1000  # Skip forward 1000 hours
                continue

            consecutive_empty = 0
            all_candles.extend(candles)

            # Log progress every 50 batches
            if batch_num % 50 == 0:
                current_date = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
                logger.info(
                    f"  {asset_name}: batch {batch_num}, {len(all_candles):,} candles (current: {current_date.date()})"
                )

            last_ts = candles[-1][0]
            if last_ts <= since:
                break
            since = last_ts + ms_per_hour

        except ccxt.NetworkError as e:
            logger.warning(f"  {asset_name}: Network error at batch {batch_num}: {e}")
            break
        except ccxt.ExchangeError as e:
            logger.warning(f"  {asset_name}: Exchange error at batch {batch_num}: {e}")
            break
        except Exception as e:
            logger.error(f"  {asset_name}: Unexpected error at batch {batch_num}: {e}")
            break

    logger.info(f"  {asset_name}: Fetched {len(all_candles):,} hourly candles")

    if not all_candles:
        logger.warning(f"  {asset_name}: No candles fetched!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Validate
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        invalid = df[col] <= 0
        if invalid.any():
            logger.warning(f"  {asset_name}: {invalid.sum()} invalid {col} values (removing)")
            df = df[~invalid]

    logger.info(f"  {asset_name}: Cleaned to {len(df):,} valid hourly candles")
    logger.info(f"  {asset_name}: Date range: {df.index.min().date()} to {df.index.max().date()}")

    return df


def main():
    """Refetch incomplete assets."""

    logger.info("=" * 80)
    logger.info("REFETCH INCOMPLETE CROSS-ASSET HOURLY DATA")
    logger.info("=" * 80)
    logger.info(f"Assets to refetch: {[a['name'] for a in ASSETS]}")
    logger.info("=" * 80)

    results = []

    for asset in ASSETS:
        symbols = asset["symbols"]
        asset_name = asset["name"]
        start_date = asset["start"]

        try:
            df = fetch_asset_hourly(symbols, asset_name, start_date=start_date)

            if df.empty:
                logger.warning(f"Skipping {asset_name} (no data)")
                continue

            # Save individual asset
            output_path = Path(f"data/raw/{asset_name}/ohlcv_hourly.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"  Saved {asset_name} to {output_path}")

            results.append(
                {
                    "asset": asset_name,
                    "rows": len(df),
                    "start_date": df.index.min(),
                    "end_date": df.index.max(),
                    "path": str(output_path),
                }
            )

        except Exception as e:
            logger.error(f"Failed to fetch {asset_name}: {e}")
            continue

        # Rate limit between assets
        import time

        time.sleep(5)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("REFETCH SUMMARY")
    logger.info("=" * 80)

    if not results:
        logger.error("No assets refetched successfully!")
        return

    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df.to_string()}")

    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS")
    logger.info("=" * 80)
    logger.info(f"Refetched {len(results)}/{len(ASSETS)} assets")


if __name__ == "__main__":
    main()
