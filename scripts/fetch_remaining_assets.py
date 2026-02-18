#!/usr/bin/env python3
"""Fetch Remaining Cross-Asset Hourly Data using Multiple Strategies

This script tries multiple approaches to get complete data for:
- SOL: Need to extend 2024-06-19 to 2026-02-16
- AVAX: Need full history from 2020-09-01
- MATIC/POL: Need full history from 2019-04-01
- DOT: Need to extend 2024-01-28 to 2026-02-16
"""

import logging
import time
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

# Try more exchanges in different order for different assets
EXCHANGE_CONFIGS = {
    "sol": {
        "symbols": ["SOL/USDT", "SOL/USD", "SOL/USDC"],
        "start": "2020-04-01",
        "exchanges": ["kraken", "coinbase", "okx", "bybit", "kucoin"],
    },
    "avax": {
        "symbols": ["AVAX/USDT", "AVAX/USD", "AVAX/USDC"],
        "start": "2020-09-01",
        "exchanges": ["kraken", "coinbase", "okx", "bybit", "kucoin"],
    },
    "matic": {
        "symbols": ["MATIC/USDT", "MATIC/USD", "MATIC/USDC", "POL/USDT", "POL/USD"],
        "start": "2019-04-01",
        "exchanges": ["kraken", "coinbase", "okx", "bybit", "kucoin"],
    },
    "dot": {
        "symbols": ["DOT/USDT", "DOT/USD", "DOT/USDC"],
        "start": "2020-08-01",
        "exchanges": ["kraken", "coinbase", "okx", "bybit", "kucoin"],
    },
}


def fetch_recent_data(
    asset_name: str,
    symbols: list[str],
    exchanges: list[str],
    start_date: str,
    end_date: str = "2026-02-16",
) -> pd.DataFrame:
    """Fetch recent data using Kraken-first strategy (last 720 candles)."""

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # For recent data, Kraken is reliable (no 'since' but gives last 720 hours)
    for ex_id in exchanges:
        for sym in symbols:
            try:
                logger.info(f"  {asset_name}: Trying {ex_id} with {sym}")
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
                ex.load_markets()

                # Kraken: no since param, just get last 720 candles
                if ex_id == "kraken":
                    candles = ex.fetch_ohlcv(sym, "1h", limit=720)
                else:
                    candles = ex.fetch_ohlcv(sym, "1h", since=start_ts, limit=1000)

                if candles and len(candles) > 100:
                    first_date = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
                    last_date = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
                    logger.info(
                        f"  {asset_name}: SUCCESS with {ex_id}/{sym} - {len(candles)} candles from {first_date.date()} to {last_date.date()}"
                    )

                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp")
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.sort_index()

                    return df

            except Exception as e:
                logger.debug(f"  {ex_id}/{sym} failed: {e}")
                continue

    return pd.DataFrame()


def fetch_full_history(
    asset_name: str,
    symbols: list[str],
    exchanges: list[str],
    start_date: str,
    end_date: str = "2026-02-16",
) -> pd.DataFrame:
    """Fetch full historical data with pagination."""

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Try exchanges that support 'since' parameter
    for ex_id in exchanges:
        if ex_id == "kraken":  # Skip kraken for historical (no 'since' support)
            continue

        for sym in symbols:
            try:
                logger.info(f"  {asset_name}: Trying {ex_id} with {sym} for historical data")
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
                ex.load_markets()

                # Test if this exchange supports historical fetch
                test_candles = ex.fetch_ohlcv(sym, "1h", since=start_ts, limit=5)
                if not test_candles:
                    continue

                first_ts = test_candles[0][0]
                first_date = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)

                # Only proceed if we can get data close to our start date
                if first_ts < start_ts + 365 * 24 * 3600 * 1000:  # Within 1 year
                    logger.info(f"  {asset_name}: {ex_id}/{sym} has historical data from {first_date.date()}")

                    # Paginated fetch
                    all_candles = []
                    since = start_ts
                    ms_per_hour = 3_600_000
                    batch_num = 0
                    consecutive_empty = 0

                    while since < end_ts and consecutive_empty < 3 and batch_num < 100:
                        batch_num += 1

                        try:
                            candles = ex.fetch_ohlcv(sym, "1h", since=since, limit=1000)

                            if not candles:
                                consecutive_empty += 1
                                since += ms_per_hour * 1000
                                continue

                            consecutive_empty = 0
                            all_candles.extend(candles)

                            if batch_num % 20 == 0:
                                current_date = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
                                logger.info(
                                    f"  {asset_name}: batch {batch_num}, {len(all_candles):,} candles (current: {current_date.date()})"
                                )

                            last_ts = candles[-1][0]
                            if last_ts <= since:
                                break
                            since = last_ts + ms_per_hour

                            time.sleep(1)  # Rate limit

                        except Exception as e:
                            logger.warning(f"  {asset_name}: Batch {batch_num} error: {e}")
                            break

                    if all_candles:
                        logger.info(f"  {asset_name}: SUCCESS with {ex_id}/{sym} - {len(all_candles)} candles")

                        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                        df = df.set_index("timestamp")
                        df = df[~df.index.duplicated(keep="last")]
                        df = df.sort_index()

                        return df

            except Exception as e:
                logger.debug(f"  {ex_id}/{sym} failed: {e}")
                continue

    return pd.DataFrame()


def main():
    """Fetch remaining assets with multiple strategies."""

    logger.info("=" * 80)
    logger.info("FETCH REMAINING CROSS-ASSET HOURLY DATA")
    logger.info("=" * 80)

    results = []

    for asset_name, config in EXCHANGE_CONFIGS.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"FETCHING {asset_name.upper()}")
        logger.info(f"{'=' * 80}")

        symbols = config["symbols"]
        start_date = config["start"]
        exchanges = config["exchanges"]

        # Check if we already have partial data
        existing_path = Path(f"data/raw/{asset_name}/ohlcv_hourly.parquet")
        existing_df = None
        if existing_path.exists():
            existing_df = pd.read_parquet(existing_path)
            logger.info(
                f"  Existing data: {len(existing_df)} rows from {existing_df.index.min().date()} to {existing_df.index.max().date()}"
            )

        # Strategy 1: Try full historical fetch
        df_historical = fetch_full_history(asset_name, symbols, exchanges, start_date)

        # Strategy 2: Get recent data (last 720 hours from Kraken)
        recent_start = (datetime.now(timezone.utc) - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
        df_recent = fetch_recent_data(asset_name, symbols, exchanges, recent_start)

        # Merge all available data
        dfs_to_merge = []
        if existing_df is not None and len(existing_df) > 0:
            dfs_to_merge.append(existing_df)
        if not df_historical.empty:
            dfs_to_merge.append(df_historical)
        if not df_recent.empty:
            dfs_to_merge.append(df_recent)

        if not dfs_to_merge:
            logger.warning(f"  {asset_name}: No data available from any source!")
            continue

        # Combine and deduplicate
        df_final = pd.concat(dfs_to_merge)
        df_final = df_final[~df_final.index.duplicated(keep="last")]
        df_final = df_final.sort_index()

        # Remove invalid prices
        for col in ["open", "high", "low", "close"]:
            invalid = df_final[col] <= 0
            if invalid.any():
                logger.warning(f"  {asset_name}: {invalid.sum()} invalid {col} values (removing)")
                df_final = df_final[~invalid]

        logger.info(
            f"  {asset_name}: Final dataset: {len(df_final):,} rows from {df_final.index.min().date()} to {df_final.index.max().date()}"
        )

        # Save
        output_path = Path(f"data/raw/{asset_name}/ohlcv_hourly.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_path)
        logger.info(f"  Saved to {output_path}")

        results.append(
            {
                "asset": asset_name,
                "rows": len(df_final),
                "start_date": df_final.index.min(),
                "end_date": df_final.index.max(),
                "path": str(output_path),
            }
        )

        time.sleep(5)  # Rate limit between assets

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 80)

    if results:
        summary_df = pd.DataFrame(results)
        logger.info(f"\n{summary_df.to_string()}")
        logger.info(f"\nTotal rows: {summary_df['rows'].sum():,}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
