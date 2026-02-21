#!/usr/bin/env python3
"""CoinAPI derivatives data backfill — discover symbols and fetch history."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sparky.data.coinapi import CoinAPIFetcher, sync_coinapi

# CoinAPI exchange IDs: BINANCEFTS (futures), OKEX (not OKX).
# Bybit excluded: sparse data (625 rows, 31d gaps) + percentage format.
# CoinAPI metric IDs use DERIVATIVES_ prefix.
DISCOVER_TARGETS = [
    {"exchange_id": "BINANCEFTS", "metric_id": "DERIVATIVES_FUNDING_RATE_CURRENT"},
    {"exchange_id": "BINANCEFTS", "metric_id": "DERIVATIVES_OPEN_INTEREST"},
    {"exchange_id": "OKEX", "metric_id": "DERIVATIVES_OPEN_INTEREST"},
]

# Confirmed symbol IDs from CoinAPI discovery.
# Funding rates: period_id=8HRS to get settlement-level snapshots.
FETCH_TARGETS = [
    {
        "metric_id": "DERIVATIVES_FUNDING_RATE_CURRENT",
        "symbol_id": "BINANCEFTS_PERP_BTC_USDT",
        "asset": "btc",
        "parquet_name": "funding_rate_btc_binance",
        "column_name": "funding_rate",
        "period_id": "8HRS",
    },
    {
        "metric_id": "DERIVATIVES_FUNDING_RATE_CURRENT",
        "symbol_id": "BINANCEFTS_PERP_ETH_USDT",
        "asset": "eth",
        "parquet_name": "funding_rate_eth_binance",
        "column_name": "funding_rate",
        "period_id": "8HRS",
    },
    {
        "metric_id": "DERIVATIVES_OPEN_INTEREST",
        "symbol_id": "BINANCEFTS_PERP_BTC_USDT",
        "asset": "btc",
        "parquet_name": "oi_btc_binance",
        "column_name": "open_interest_usd",
    },
    {
        "metric_id": "DERIVATIVES_OPEN_INTEREST",
        "symbol_id": "BINANCEFTS_PERP_ETH_USDT",
        "asset": "eth",
        "parquet_name": "oi_eth_binance",
        "column_name": "open_interest_usd",
    },
    {
        "metric_id": "DERIVATIVES_OPEN_INTEREST",
        "symbol_id": "OKEX_PERP_BTC_USDT",
        "asset": "btc",
        "parquet_name": "oi_btc_okx",
        "column_name": "open_interest_usd",
    },
    {
        "metric_id": "DERIVATIVES_OPEN_INTEREST",
        "symbol_id": "OKEX_PERP_ETH_USDT",
        "asset": "eth",
        "parquet_name": "oi_eth_okx",
        "column_name": "open_interest_usd",
    },
]


def discover():
    fetcher = CoinAPIFetcher()
    for target in DISCOVER_TARGETS:
        exchange = target["exchange_id"]
        metric = target["metric_id"]
        print(f"\n{'=' * 60}")
        print(f"Discovering: {exchange} / {metric}")
        print(f"{'=' * 60}")
        try:
            symbols = fetcher.discover_symbols(exchange, metric)
            if not symbols:
                print("  No symbols found")
                continue
            # Show BTC/ETH perpetual symbols
            for s in symbols:
                sid = s.get("symbol_id", "")
                if "BTC" in sid or "ETH" in sid:
                    print(f"  {sid}")
            print(f"  ({len(symbols)} total symbols)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nEstimated cost: ${fetcher.credits_used:.3f}")


def fetch(targets=None, dry_run=False):
    targets = targets or FETCH_TARGETS
    if dry_run:
        print("DRY RUN — would fetch:")
        for t in targets:
            print(f"  {t['metric_id']} / {t['symbol_id']} → {t['parquet_name']}")
        return

    results = sync_coinapi(targets)
    print("\nResults:")
    for name, count in results.items():
        print(f"  {name}: {count} rows")


def main():
    parser = argparse.ArgumentParser(description="CoinAPI derivatives backfill")
    parser.add_argument("--discover", action="store_true", help="Discover available symbols")
    parser.add_argument("--fetch", action="store_true", help="Run backfill")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    args = parser.parse_args()

    if args.discover:
        discover()
    elif args.fetch:
        fetch(dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
