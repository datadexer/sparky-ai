#!/usr/bin/env python3
"""Fetch Cross-Asset Hourly Data (7 Cryptos)

APPROACH 2: Cross-Asset Training (HIGH PRIORITY)

Train on multiple assets, test on BTC only.
Pooled training set: 490,000+ hourly observations.

Assets (7 major cryptos):
1. BTC (Bitcoin) - dominant asset
2. ETH (Ethereum) - largest altcoin
3. SOL (Solana) - high-performance blockchain
4. ADA (Cardano) - proof-of-stake
5. DOT (Polkadot) - interoperability
6. MATIC (Polygon) - Ethereum scaling
7. AVAX (Avalanche) - smart contracts

Strategy:
- Fetch hourly OHLCV for all 7 assets (2017-2025)
- Compute identical features for each
- Pool all assets with asset_id as categorical feature
- Train generic crypto momentum predictor
- Test ONLY on BTC 2024-2025 holdout

Expected: 7 assets × 70,000 hourly candles = 490,000 training samples
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Asset configuration
ASSETS = [
    {"symbol": "BTC/USDT", "name": "btc", "asset_id": 0},
    {"symbol": "ETH/USDT", "name": "eth", "asset_id": 1},
    {"symbol": "SOL/USDT", "name": "sol", "asset_id": 2},
    {"symbol": "ADA/USDT", "name": "ada", "asset_id": 3},
    {"symbol": "DOT/USDT", "name": "dot", "asset_id": 4},
    {"symbol": "MATIC/USDT", "name": "matic", "asset_id": 5},
    {"symbol": "AVAX/USDT", "name": "avax", "asset_id": 6},
]


def fetch_asset_hourly(
    symbol: str,
    asset_name: str,
    start_date: str = "2017-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Fetch hourly OHLCV for a single asset.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        asset_name: Asset identifier (e.g., "btc")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with hourly OHLCV data
    """
    from sparky.data.price import CCXTPriceFetcher

    logger.info(f"\nFetching {symbol} hourly data...")

    fetcher = CCXTPriceFetcher(exchange_id="binance")

    # Use _paginated_fetch with "1h" timeframe
    # (We can't use fetch_daily_ohlcv as it's hardcoded to "1d")

    import ccxt
    from datetime import datetime, timezone

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

    # Paginated fetch
    all_candles = []
    since = start_ts
    ms_per_hour = 3_600_000

    batch_num = 0
    while since < end_ts:
        batch_num += 1

        try:
            candles = fetcher.exchange.fetch_ohlcv(
                symbol, "1h", since=since, limit=1000
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Log progress every 20 batches
            if batch_num % 20 == 0:
                current_date = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
                logger.info(
                    f"  {symbol}: batch {batch_num}, {len(all_candles):,} candles "
                    f"(current: {current_date.date()})"
                )

            last_ts = candles[-1][0]
            if last_ts <= since:
                break
            since = last_ts + ms_per_hour

        except ccxt.NetworkError as e:
            logger.warning(f"  {symbol}: Network error at batch {batch_num}: {e}")
            break
        except ccxt.ExchangeError as e:
            logger.warning(f"  {symbol}: Exchange error at batch {batch_num}: {e}")
            break
        except Exception as e:
            logger.error(f"  {symbol}: Unexpected error at batch {batch_num}: {e}")
            break

    logger.info(f"  {symbol}: Fetched {len(all_candles):,} hourly candles")

    if not all_candles:
        logger.warning(f"  {symbol}: No candles fetched!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
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
            logger.warning(f"  {symbol}: {invalid.sum()} invalid {col} values (removing)")
            df = df[~invalid]

    logger.info(f"  {symbol}: Cleaned to {len(df):,} valid hourly candles")
    logger.info(f"  {symbol}: Date range: {df.index.min().date()} to {df.index.max().date()}")

    return df


def main():
    """Main execution: fetch all 7 assets."""

    logger.info("=" * 80)
    logger.info("FETCH CROSS-ASSET HOURLY DATA (7 CRYPTOS)")
    logger.info("=" * 80)
    logger.info(f"Assets: {len(ASSETS)}")
    logger.info(f"Timeframe: 1 hour (1h)")
    logger.info(f"Period: 2017-01-01 to 2025-12-31")
    logger.info("=" * 80)

    results = []

    for asset in ASSETS:
        symbol = asset["symbol"]
        asset_name = asset["name"]
        asset_id = asset["asset_id"]

        try:
            df = fetch_asset_hourly(symbol, asset_name)

            if df.empty:
                logger.warning(f"Skipping {asset_name} (no data)")
                continue

            # Save individual asset
            output_path = Path(f"data/raw/{asset_name}/ohlcv_hourly.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"  Saved {asset_name} to {output_path}")

            results.append({
                "asset": asset_name,
                "asset_id": asset_id,
                "symbol": symbol,
                "rows": len(df),
                "start_date": df.index.min(),
                "end_date": df.index.max(),
                "path": str(output_path),
            })

        except Exception as e:
            logger.error(f"Failed to fetch {asset_name}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-ASSET FETCH SUMMARY")
    logger.info("=" * 80)

    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df.to_string()}")

    total_rows = summary_df["rows"].sum()
    logger.info(f"\nTotal hourly candles across all assets: {total_rows:,}")
    logger.info(f"Average candles per asset: {total_rows / len(results):,.0f}")
    logger.info(f"Expected pooled training samples: {total_rows:,}")

    # Save summary
    summary_path = Path("data/cross_asset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary to {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS")
    logger.info("=" * 80)
    logger.info(f"Fetched {len(results)}/{len(ASSETS)} assets")
    logger.info(f"Total samples: {total_rows:,}")
    logger.info(f"Next step: Run scripts/prepare_cross_asset_features.py")


if __name__ == "__main__":
    main()
