#!/usr/bin/env python3
"""Fetch Maximum Hourly BTC Data Coverage from Multiple Exchanges

Strategy: Try multiple exchanges to get the LONGEST possible historical coverage.

Exchanges (oldest to newest):
1. Kraken (2011) - BTC/USD likely back to 2013-2014
2. Bitstamp (2011) - BTC/USD likely back to 2013-2014
3. Coinbase (2012) - BTC/USD likely back to 2014-2015
4. Bitfinex (2012) - BTC/USD likely back to 2013-2014
5. Poloniex (2014) - BTC/USDT from 2015-2016
6. Gemini (2015) - BTC/USD from 2015
7. OKX (newer) - BTC/USDT from 2019

Goal: Combine all sources to get 2013-2025 (if available) = ~105,000 hourly candles
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import ccxt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Exchange configuration (oldest first for maximum historical coverage)
EXCHANGES = [
    {"id": "kraken", "symbol": "BTC/USD", "priority": 1},
    {"id": "bitstamp", "symbol": "BTC/USD", "priority": 2},
    {"id": "bitfinex", "symbol": "BTC/USD", "priority": 3},
    {"id": "coinbase", "symbol": "BTC/USD", "priority": 4},
    {"id": "poloniex", "symbol": "BTC/USDT", "priority": 5},
    {"id": "gemini", "symbol": "BTC/USD", "priority": 6},
    {"id": "okx", "symbol": "BTC/USDT", "priority": 7},
]


def fetch_from_exchange(
    exchange_id: str,
    symbol: str,
    start_year: int = 2013,
    end_year: int = 2025
) -> pd.DataFrame:
    """Fetch hourly data from a single exchange.

    Args:
        exchange_id: Exchange identifier (e.g., 'kraken')
        symbol: Trading pair (e.g., 'BTC/USD')
        start_year: Start year to try fetching from
        end_year: End year

    Returns:
        DataFrame with hourly OHLCV, or empty DataFrame if failed
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Trying {exchange_id} ({symbol})...")
        logger.info(f"{'='*60}")

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})

        start_ts = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime(end_year, 12, 31, tzinfo=timezone.utc).timestamp() * 1000)

        all_candles = []
        since = start_ts
        ms_per_hour = 3_600_000

        batch_num = 0
        consecutive_empty = 0

        while since < end_ts and consecutive_empty < 3:
            batch_num += 1

            try:
                candles = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)

                if not candles:
                    consecutive_empty += 1
                    logger.warning(f"  Empty batch {batch_num}, consecutive empty: {consecutive_empty}")
                    # Try jumping forward 6 months if no data
                    since += ms_per_hour * 24 * 180
                    continue

                consecutive_empty = 0
                all_candles.extend(candles)

                # Log progress every 20 batches
                if batch_num % 20 == 0:
                    current_date = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
                    logger.info(f"  Batch {batch_num}: {len(all_candles):,} candles (current: {current_date.date()})")

                last_ts = candles[-1][0]
                if last_ts <= since:
                    break
                since = last_ts + ms_per_hour

            except Exception as e:
                logger.warning(f"  Batch {batch_num} error: {e}")
                consecutive_empty += 1
                since += ms_per_hour * 1000  # Skip ahead
                continue

        if not all_candles:
            logger.warning(f"  {exchange_id}: No data available")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # Add source column
        df["source"] = exchange_id

        logger.info(f"  ✓ {exchange_id}: {len(df):,} candles ({df.index.min().date()} to {df.index.max().date()})")

        return df

    except Exception as e:
        logger.error(f"  ✗ {exchange_id}: Failed to fetch - {e}")
        return pd.DataFrame()


def merge_multi_source(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge dataframes from multiple exchanges, prioritizing older sources.

    Args:
        dataframes: List of DataFrames with 'source' column

    Returns:
        Merged DataFrame with best coverage
    """
    if not dataframes:
        return pd.DataFrame()

    # Concatenate all
    combined = pd.concat(dataframes, axis=0)

    # Remove duplicates, keeping first occurrence (from oldest exchange)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    logger.info(f"\nMerged data:")
    logger.info(f"  Total candles: {len(combined):,}")
    logger.info(f"  Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    logger.info(f"  Years covered: {(combined.index.max() - combined.index.min()).days / 365.25:.1f}")

    # Source distribution
    source_counts = combined["source"].value_counts()
    logger.info(f"\n  Source distribution:")
    for source, count in source_counts.items():
        logger.info(f"    {source}: {count:,} candles ({100*count/len(combined):.1f}%)")

    return combined


def main():
    """Main execution: fetch from multiple exchanges and merge."""

    logger.info("=" * 80)
    logger.info("FETCH MAXIMUM HOURLY BTC COVERAGE (MULTI-EXCHANGE)")
    logger.info("=" * 80)
    logger.info(f"Target: 2013-2025 (maximum historical coverage)")
    logger.info(f"Exchanges: {len(EXCHANGES)} sources")
    logger.info("=" * 80)

    dataframes = []

    for exchange_config in EXCHANGES:
        df = fetch_from_exchange(
            exchange_id=exchange_config["id"],
            symbol=exchange_config["symbol"],
            start_year=2013,
            end_year=2025
        )

        if not df.empty:
            dataframes.append(df)
            logger.info(f"✓ Added {exchange_config['id']} to merge pool")
        else:
            logger.warning(f"✗ Skipping {exchange_config['id']} (no data)")

    if not dataframes:
        logger.error("No data fetched from any exchange!")
        return

    # Merge all sources
    logger.info("\n" + "=" * 80)
    logger.info("MERGING MULTI-SOURCE DATA")
    logger.info("=" * 80)

    combined = merge_multi_source(dataframes)

    # Drop source column for final output
    combined = combined.drop(columns=["source"])

    # Save to Parquet
    output_path = Path("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    logger.info(f"\n✓ Saved to {output_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS - MAXIMUM COVERAGE ACHIEVED")
    logger.info("=" * 80)
    logger.info(f"Total hourly candles: {len(combined):,}")
    logger.info(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    logger.info(f"Years covered: {(combined.index.max() - combined.index.min()).days / 365.25:.1f}")
    logger.info(f"Daily equivalent: {len(combined) / 24:,.0f} days")
    logger.info(f"Data increase vs daily: {len(combined) / (len(combined) / 24):.1f}x")
    logger.info(f"\nFile: {output_path}")


if __name__ == "__main__":
    main()
