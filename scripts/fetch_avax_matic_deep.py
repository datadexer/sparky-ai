#!/usr/bin/env python3
"""Deep Fetch for AVAX and MATIC Historical Data

Last attempt to get historical data for these two assets.
Using aggressive pagination with multiple exchanges.
"""

import logging
from pathlib import Path
import pandas as pd
import ccxt
from datetime import datetime, timezone, timedelta
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def try_historical_fetch_backwards(
    exchange_id: str,
    symbol: str,
    asset_name: str,
    start_date: str,
) -> pd.DataFrame:
    """Try fetching by going backwards from present using limit parameter."""

    try:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        ex.load_markets()

        if symbol not in ex.symbols:
            return pd.DataFrame()

        logger.info(f"  {asset_name}: Trying {exchange_id}/{symbol} with backwards pagination")

        # Start from now and go backwards
        all_candles = []
        max_batches = 200  # Up to 200K hours = ~22 years

        for batch in range(max_batches):
            try:
                # Just request last N candles without 'since'
                candles = ex.fetch_ohlcv(symbol, "1h", limit=1000)

                if not candles:
                    break

                # Check if we already have this data
                if all_candles:
                    last_existing_ts = all_candles[-1][0]
                    new_data = [c for c in candles if c[0] < last_existing_ts]
                    if not new_data:
                        break  # No new historical data available
                    all_candles.extend(new_data)
                else:
                    all_candles.extend(candles)

                if batch % 10 == 0 and all_candles:
                    oldest_date = datetime.fromtimestamp(min(c[0] for c in all_candles) / 1000, tz=timezone.utc)
                    logger.info(f"  {asset_name}: batch {batch}, {len(all_candles):,} candles, oldest: {oldest_date.date()}")

                # If we keep getting the same data, this exchange doesn't support deep history
                if batch > 5 and len(all_candles) < 2000:
                    logger.info(f"  {asset_name}: {exchange_id} only provides recent data ({len(all_candles)} candles)")
                    break

                time.sleep(ex.rateLimit / 1000 if hasattr(ex, 'rateLimit') else 1)

            except Exception as e:
                logger.debug(f"  Batch {batch} error: {e}")
                break

        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            logger.info(f"  {asset_name}: {exchange_id}/{symbol} yielded {len(df):,} candles from {df.index.min().date()} to {df.index.max().date()}")
            return df

    except Exception as e:
        logger.debug(f"  {exchange_id}/{symbol} failed: {e}")

    return pd.DataFrame()


def aggressive_fetch(asset_name: str, symbols: list[str], start_date: str) -> pd.DataFrame:
    """Try every possible exchange and method."""

    # Exchanges to try (in order of reliability for altcoins)
    exchanges = ["binance", "okx", "bybit", "gate", "kucoin", "huobi", "mexc"]

    best_df = pd.DataFrame()

    for exchange_id in exchanges:
        for symbol in symbols:
            df = try_historical_fetch_backwards(exchange_id, symbol, asset_name, start_date)

            if not df.empty:
                # Keep the longest dataset
                if len(df) > len(best_df):
                    best_df = df
                    logger.info(f"  {asset_name}: New best dataset from {exchange_id}/{symbol} with {len(df):,} rows")

    return best_df


def main():
    """Fetch AVAX and MATIC with aggressive strategies."""

    logger.info("=" * 80)
    logger.info("DEEP FETCH: AVAX + MATIC HISTORICAL DATA")
    logger.info("=" * 80)

    assets = [
        {
            "name": "avax",
            "symbols": ["AVAX/USDT", "AVAX/BUSD", "AVAX/USD", "AVAX/USDC"],
            "start": "2020-09-01",
        },
        {
            "name": "matic",
            "symbols": ["MATIC/USDT", "MATIC/BUSD", "MATIC/USD", "MATIC/USDC", "POL/USDT", "POL/USD"],
            "start": "2019-04-01",
        },
    ]

    for asset in assets:
        logger.info(f"\n{'='*80}")
        logger.info(f"FETCHING {asset['name'].upper()}")
        logger.info(f"{'='*80}")

        # Try aggressive fetch
        df_new = aggressive_fetch(asset["name"], asset["symbols"], asset["start"])

        # Merge with existing
        existing_path = Path(f"data/raw/{asset['name']}/ohlcv_hourly.parquet")
        if existing_path.exists():
            df_existing = pd.read_parquet(existing_path)
            logger.info(f"  Existing: {len(df_existing):,} rows")

            if not df_new.empty:
                df_combined = pd.concat([df_existing, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                df_combined = df_combined.sort_index()
            else:
                df_combined = df_existing
        else:
            df_combined = df_new

        if not df_combined.empty:
            output_path = Path(f"data/raw/{asset['name']}/ohlcv_hourly.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_combined.to_parquet(output_path)

            logger.info(f"  FINAL: {len(df_combined):,} rows from {df_combined.index.min().date()} to {df_combined.index.max().date()}")
            logger.info(f"  Saved to {output_path}")
        else:
            logger.warning(f"  No data available for {asset['name']}")

        time.sleep(10)  # Rate limit between assets

    logger.info("\n" + "=" * 80)
    logger.info("DEEP FETCH COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
