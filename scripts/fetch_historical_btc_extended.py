#!/usr/bin/env python3
"""Fetch Extended BTC Historical Data (2013-2018)

Expand training dataset by fetching BTC OHLCV back to 2013.
Current data: 2019-2025
Target: 2013-2025 (10+ years)

Data sources:
1. CCXT (Binance historical via API)
2. Alternative: CoinGecko historical chart data
3. Fallback: Manual CSV import if needed
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


def main():
    logger.info("=" * 80)
    logger.info("FETCH EXTENDED BTC HISTORICAL DATA")
    logger.info("=" * 80)
    logger.info("Target period: 2013-01-01 to 2018-12-31")
    logger.info("Source: CCXT (Binance) + CoinGecko fallback")
    logger.info("=" * 80)

    # Check current data range
    from sparky.data.storage import DataStore
    store = DataStore()

    try:
        current_data, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
        logger.info(f"Current data range: {current_data.index[0]} to {current_data.index[-1]}")
        logger.info(f"Current rows: {len(current_data)}")
    except Exception as e:
        logger.info(f"Could not load current data: {e}")
        current_data = None

    # Fetch extended historical data
    logger.info("\n" + "=" * 80)
    logger.info("Fetching 2013-2018 BTC OHLCV data...")
    logger.info("=" * 80)

    # Try CCXT first
    try:
        logger.info("Attempting CCXT (Binance)...")
        from sparky.data.ccxt_fetcher import CCXTPriceFetcher

        fetcher = CCXTPriceFetcher()

        # Binance launched in 2017, so we can get 2017-2018 from there
        # For 2013-2016, need alternative source

        start_2017 = datetime(2017, 1, 1, tzinfo=timezone.utc)
        end_2018 = datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        logger.info(f"Fetching 2017-2018 from Binance...")
        btc_2017_2018 = fetcher.fetch_ohlcv(
            symbol="BTC/USDT",
            start_date=start_2017,
            end_date=end_2018,
            timeframe="1d"
        )

        logger.info(f"Fetched {len(btc_2017_2018)} rows for 2017-2018")

    except Exception as e:
        logger.error(f"CCXT failed: {e}")
        btc_2017_2018 = None

    # Try CoinGecko for full historical range (2013-2018)
    try:
        logger.info("\nAttempting CoinGecko historical charts...")
        from sparky.data.coingecko_fetcher import CoinGeckoFetcher

        fetcher = CoinGeckoFetcher()

        # CoinGecko has full BTC history back to 2013
        start_2013 = datetime(2013, 1, 1, tzinfo=timezone.utc)
        end_2018 = datetime(2018, 12, 31, tzinfo=timezone.utc)

        logger.info(f"Fetching 2013-2018 from CoinGecko...")
        btc_2013_2018 = fetcher.fetch_historical_chart(
            coin_id="bitcoin",
            vs_currency="usd",
            start_date=start_2013,
            end_date=end_2018
        )

        logger.info(f"Fetched {len(btc_2013_2018)} rows for 2013-2018")

        # CoinGecko returns: timestamp, price, market_cap, total_volume
        # Need to convert to OHLCV format (use price as close, estimate OHLC)

        if btc_2013_2018 is not None and len(btc_2013_2018) > 0:
            # Convert to OHLCV (approximate: use price as close, estimate open/high/low)
            ohlcv_data = pd.DataFrame({
                'open': btc_2013_2018['price'],  # Approximate
                'high': btc_2013_2018['price'] * 1.01,  # Estimate: +1% from close
                'low': btc_2013_2018['price'] * 0.99,  # Estimate: -1% from close
                'close': btc_2013_2018['price'],
                'volume': btc_2013_2018.get('total_volume', 0)
            }, index=btc_2013_2018.index)

            logger.info(f"Converted to OHLCV format: {len(ohlcv_data)} rows")

            # Merge with current data
            if current_data is not None:
                # Combine old and new data
                combined = pd.concat([ohlcv_data, current_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()

                logger.info(f"Combined data: {len(combined)} rows ({combined.index[0]} to {combined.index[-1]})")

                # Save extended dataset
                output_path = Path("data/raw/btc/ohlcv_extended.parquet")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(output_path)

                logger.info(f"Saved extended data to {output_path}")

                # Statistics
                logger.info("\n" + "=" * 80)
                logger.info("EXTENDED DATASET STATISTICS")
                logger.info("=" * 80)
                logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")
                logger.info(f"Total rows: {len(combined)}")
                logger.info(f"Years covered: {(combined.index[-1] - combined.index[0]).days / 365.25:.1f}")
                logger.info(f"New rows added: {len(combined) - len(current_data)}")

                return combined

    except Exception as e:
        logger.error(f"CoinGecko failed: {e}")
        logger.info("\nFallback: Consider manual CSV import or other data sources")

    logger.info("\n" + "=" * 80)
    logger.info("MANUAL DATA IMPORT INSTRUCTIONS")
    logger.info("=" * 80)
    logger.info("If automated fetch fails, download BTC historical data manually:")
    logger.info("1. CoinGecko: https://www.coingecko.com/en/coins/bitcoin/historical_data")
    logger.info("2. Yahoo Finance: Search 'BTC-USD' → Historical Data")
    logger.info("3. CryptoDataDownload: https://www.cryptodatadownload.com/data/binance/")
    logger.info("")
    logger.info("Save as CSV and place in: data/raw/btc/historical_2013_2018.csv")
    logger.info("Expected columns: Date, Open, High, Low, Close, Volume")


if __name__ == "__main__":
    main()
