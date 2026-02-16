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

# Asset configuration — symbols per exchange (Binance blocked from this location)
# Per-asset start dates: many altcoins launched after 2017
ASSETS = [
    {"symbols": ["BTC/USDT", "BTC/USD"], "name": "btc", "asset_id": 0, "start": "2017-01-01", "skip": True},  # Already have max coverage data
    {"symbols": ["ETH/USDT", "ETH/USD"], "name": "eth", "asset_id": 1, "start": "2017-01-01"},
    {"symbols": ["SOL/USDT", "SOL/USD"], "name": "sol", "asset_id": 2, "start": "2020-04-01"},  # SOL launched 2020
    {"symbols": ["ADA/USDT", "ADA/USD"], "name": "ada", "asset_id": 3, "start": "2017-10-01"},
    {"symbols": ["DOT/USDT", "DOT/USD"], "name": "dot", "asset_id": 4, "start": "2020-08-01"},  # DOT launched 2020
    {"symbols": ["POL/USDT", "POL/USD", "MATIC/USDT", "MATIC/USD"], "name": "matic", "asset_id": 5, "start": "2019-04-01"},  # MATIC renamed to POL
    {"symbols": ["AVAX/USDT", "AVAX/USD"], "name": "avax", "asset_id": 6, "start": "2020-09-01"},  # AVAX launched 2020
    {"symbols": ["LINK/USDT", "LINK/USD"], "name": "link", "asset_id": 7, "start": "2017-09-01"},
]

# Coinbase has best historical coverage and no rate limit issues.
# OKX supports historical `since` parameter for hourly data.
# Bitfinex supports `since` but rate-limits aggressively.
# Kraken ignores `since` and only returns last 720 candles — use as last resort.
FAILOVER_EXCHANGES = ["coinbase", "okx", "bitfinex", "kraken"]


def fetch_asset_hourly(
    symbols: list[str],
    asset_name: str,
    start_date: str = "2017-01-01",
    end_date: str = "2026-02-16",
) -> pd.DataFrame:
    """Fetch hourly OHLCV for a single asset with exchange failover.

    Tries each exchange in FAILOVER_EXCHANGES order with each symbol variant
    until one succeeds.

    Args:
        symbols: List of trading pair variants (e.g., ["ETH/USDT", "ETH/USD"])
        asset_name: Asset identifier (e.g., "eth")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with hourly OHLCV data
    """
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

    # Try each exchange + symbol combo until one works with historical data
    exchange = None
    working_symbol = None
    for ex_id in FAILOVER_EXCHANGES:
        for sym in symbols:
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
                ex.load_markets()  # Required for OKX/Coinbase historical data
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
            except Exception:
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
            candles = exchange.fetch_ohlcv(
                working_symbol, "1h", since=since, limit=1000
            )

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
                    f"  {asset_name}: batch {batch_num}, {len(all_candles):,} candles "
                    f"(current: {current_date.date()})"
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
            logger.warning(f"  {asset_name}: {invalid.sum()} invalid {col} values (removing)")
            df = df[~invalid]

    logger.info(f"  {asset_name}: Cleaned to {len(df):,} valid hourly candles")
    logger.info(f"  {asset_name}: Date range: {df.index.min().date()} to {df.index.max().date()}")

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
        symbols = asset["symbols"]
        asset_name = asset["name"]
        asset_id = asset["asset_id"]
        start_date = asset.get("start", "2017-01-01")

        if asset.get("skip"):
            logger.info(f"Skipping {asset_name} (already have data)")
            # Record existing data
            existing_path = Path(f"data/raw/{asset_name}/ohlcv_hourly.parquet")
            if existing_path.exists():
                import pandas as _pd
                _df = _pd.read_parquet(existing_path)
                results.append({
                    "asset": asset_name, "asset_id": asset_id,
                    "symbol": str(symbols), "rows": len(_df),
                    "start_date": _df.index.min(), "end_date": _df.index.max(),
                    "path": str(existing_path),
                })
            continue

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

            results.append({
                "asset": asset_name,
                "asset_id": asset_id,
                "symbol": str(symbols),
                "rows": len(df),
                "start_date": df.index.min(),
                "end_date": df.index.max(),
                "path": str(output_path),
            })

        except Exception as e:
            logger.error(f"Failed to fetch {asset_name}: {e}")
            continue

        # Rate limit between assets to avoid exchange bans
        import time
        time.sleep(5)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-ASSET FETCH SUMMARY")
    logger.info("=" * 80)

    if not results:
        logger.error("No assets fetched successfully!")
        return

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
