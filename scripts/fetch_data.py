#!/usr/bin/env python3
"""Fetch historical data from all sources.

Full data fetch (first run) or incremental update (subsequent runs).
Stores everything in data/raw/{asset}/ as Parquet files.

Usage:
    python scripts/fetch_data.py                    # Full fetch
    python scripts/fetch_data.py --incremental      # Incremental update
    python scripts/fetch_data.py --dry-run           # Show what would be fetched
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparky.data.storage import DataStore
from sparky.data.price import CCXTPriceFetcher
from sparky.data.onchain_bgeometrics import BGeometricsFetcher
from sparky.data.onchain_coinmetrics import CoinMetricsFetcher
from sparky.data.onchain_blockchain_com import BlockchainComFetcher
from sparky.data.market_context import CoinGeckoFetcher
from sparky.data.source_selector import SourceSelector
from sparky.data.quality import DataQualityChecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Data paths
DATA_RAW = Path("data/raw")
RESULTS_DIR = Path("results")

# Date ranges
BTC_START = "2017-01-01"
ETH_START = "2017-07-30"  # ETH had sufficient liquidity by mid-2017

store = DataStore()


def fetch_price_data(incremental: bool = False) -> dict:
    """Fetch BTC and ETH OHLCV from CCXT/Binance."""
    results = {}
    fetcher = CCXTPriceFetcher()

    for symbol, asset, start in [
        ("BTC/USDT", "btc", BTC_START),
        ("ETH/USDT", "eth", ETH_START),
    ]:
        path = DATA_RAW / asset / "ohlcv.parquet"
        effective_start = start

        if incremental:
            last_ts = store.get_last_timestamp(path)
            if last_ts:
                effective_start = last_ts.strftime("%Y-%m-%d")
                logger.info(f"Incremental: {asset} OHLCV from {effective_start}")

        try:
            df = fetcher.fetch_daily_ohlcv(symbol, effective_start)
            if not df.empty:
                if incremental and path.exists():
                    store.append(df, path, metadata={"source": "binance", "asset": asset})
                else:
                    store.save(df, path, metadata={"source": "binance", "asset": asset})
                results[f"{asset}_ohlcv"] = {
                    "rows": len(df),
                    "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
                }
            else:
                results[f"{asset}_ohlcv"] = {"rows": 0, "error": "empty response"}
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            results[f"{asset}_ohlcv"] = {"rows": 0, "error": str(e)}

    return results


def fetch_bgeometrics_data(
    incremental: bool = False, token: str | None = None
) -> dict:
    """Fetch BTC computed on-chain from BGeometrics."""
    path = DATA_RAW / "btc" / "onchain_bgeometrics.parquet"
    effective_start = BTC_START

    if incremental:
        last_ts = store.get_last_timestamp(path)
        if last_ts:
            effective_start = last_ts.strftime("%Y-%m-%d")

    try:
        fetcher = BGeometricsFetcher(token=token)
        df = fetcher.fetch_all_metrics(effective_start)
        if not df.empty:
            if incremental and path.exists():
                store.append(df, path, metadata={"source": "bgeometrics", "asset": "btc"})
            else:
                store.save(df, path, metadata={"source": "bgeometrics", "asset": "btc"})
            return {
                "rows": len(df),
                "metrics": list(df.columns),
                "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
            }
        return {"rows": 0, "error": "empty response"}
    except Exception as e:
        logger.error(f"Failed to fetch BGeometrics: {e}")
        return {"rows": 0, "error": str(e)}


def fetch_coinmetrics_data(incremental: bool = False) -> dict:
    """Fetch BTC + ETH raw on-chain from CoinMetrics."""
    results = {}

    try:
        fetcher = CoinMetricsFetcher()
    except ImportError as e:
        logger.error(f"CoinMetrics client not available: {e}")
        return {"error": str(e)}

    for asset, start in [("btc", BTC_START), ("eth", ETH_START)]:
        path = DATA_RAW / asset / "onchain_coinmetrics.parquet"
        effective_start = start

        if incremental:
            last_ts = store.get_last_timestamp(path)
            if last_ts:
                effective_start = last_ts.strftime("%Y-%m-%d")

        try:
            df = fetcher.fetch_asset_metrics(asset, effective_start)
            if not df.empty:
                if incremental and path.exists():
                    store.append(df, path, metadata={"source": "coinmetrics", "asset": asset})
                else:
                    store.save(df, path, metadata={"source": "coinmetrics", "asset": asset})
                results[f"{asset}_coinmetrics"] = {
                    "rows": len(df),
                    "metrics": list(df.columns),
                    "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
                }
            else:
                results[f"{asset}_coinmetrics"] = {"rows": 0, "error": "empty response"}
        except Exception as e:
            logger.error(f"Failed to fetch CoinMetrics {asset}: {e}")
            results[f"{asset}_coinmetrics"] = {"rows": 0, "error": str(e)}

    return results


def fetch_blockchain_com_data() -> dict:
    """Fetch BTC raw on-chain from Blockchain.com (validation reference)."""
    path = DATA_RAW / "btc" / "onchain_blockchain_com.parquet"

    try:
        fetcher = BlockchainComFetcher()
        df = fetcher.fetch_all_metrics(timespan="5years")
        if not df.empty:
            store.save(df, path, metadata={"source": "blockchain_com", "asset": "btc"})
            return {
                "rows": len(df),
                "metrics": list(df.columns),
                "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
            }
        return {"rows": 0, "error": "empty response"}
    except Exception as e:
        logger.error(f"Failed to fetch Blockchain.com: {e}")
        return {"rows": 0, "error": str(e)}


def fetch_coingecko_data() -> dict:
    """Fetch market context snapshot from CoinGecko."""
    path = DATA_RAW / "market_context.parquet"

    try:
        fetcher = CoinGeckoFetcher()
        df = fetcher.fetch_market_data(top_n=250)
        if not df.empty:
            store.save(df, path, metadata={"source": "coingecko", "type": "market_context"})
            return {"rows": len(df), "coins": len(df)}
        return {"rows": 0, "error": "empty response"}
    except Exception as e:
        logger.error(f"Failed to fetch CoinGecko: {e}")
        return {"rows": 0, "error": str(e)}


def run_source_selection() -> dict:
    """Run source selector on fetched on-chain data."""
    selector = SourceSelector()
    results = {}

    # Load BTC on-chain data from all sources
    bg_path = DATA_RAW / "btc" / "onchain_bgeometrics.parquet"
    cm_path = DATA_RAW / "btc" / "onchain_coinmetrics.parquet"
    bc_path = DATA_RAW / "btc" / "onchain_blockchain_com.parquet"

    bg_df = store.load(bg_path)[0] if bg_path.exists() else None
    cm_df = store.load(cm_path)[0] if cm_path.exists() else None
    bc_df = store.load(bc_path)[0] if bc_path.exists() else None

    unified_btc, scores = selector.select_btc_onchain(
        bgeometrics_df=bg_df,
        coinmetrics_df=cm_df,
        blockchain_com_df=bc_df,
    )

    if not unified_btc.empty:
        unified_path = DATA_RAW / "btc" / "onchain_unified.parquet"
        store.save(
            unified_btc, unified_path,
            metadata={"source": "unified", "asset": "btc"},
        )
        results["btc_unified"] = {
            "rows": len(unified_btc),
            "metrics": list(unified_btc.columns),
            "scores": [
                {
                    "source": s.source,
                    "metric": s.metric,
                    "completeness": round(s.completeness, 3),
                    "freshness_days": round(s.freshness_days, 1),
                    "reference_mape": round(s.reference_mape, 4) if s.reference_mape else None,
                    "selected": s.selected,
                }
                for s in scores
            ],
        }

    # ETH: just use CoinMetrics
    eth_cm_path = DATA_RAW / "eth" / "onchain_coinmetrics.parquet"
    if eth_cm_path.exists():
        eth_df = store.load(eth_cm_path)[0]
        eth_unified = selector.select_eth_onchain(eth_df)
        if not eth_unified.empty:
            eth_unified_path = DATA_RAW / "eth" / "onchain_unified.parquet"
            store.save(
                eth_unified, eth_unified_path,
                metadata={"source": "coinmetrics", "asset": "eth"},
            )
            results["eth_unified"] = {
                "rows": len(eth_unified),
                "metrics": list(eth_unified.columns),
            }

    return results


def run_quality_checks() -> dict:
    """Run quality checks on all fetched data."""
    checker = DataQualityChecker()
    reports = {}

    # Check BTC OHLCV
    btc_ohlcv_path = DATA_RAW / "btc" / "ohlcv.parquet"
    if btc_ohlcv_path.exists():
        df = store.load(btc_ohlcv_path)[0]
        report = checker.run_all_checks(
            df, asset="btc", source="binance",
            price_range=(0.01, 1_000_000),
        )
        checker.save_report(report, "btc_ohlcv_quality.json")
        reports["btc_ohlcv"] = report

    # Check ETH OHLCV
    eth_ohlcv_path = DATA_RAW / "eth" / "ohlcv.parquet"
    if eth_ohlcv_path.exists():
        df = store.load(eth_ohlcv_path)[0]
        report = checker.run_all_checks(
            df, asset="eth", source="binance",
            price_range=(0.01, 100_000),
        )
        checker.save_report(report, "eth_ohlcv_quality.json")
        reports["eth_ohlcv"] = report

    # Cross-validate BTC price: Binance vs CoinMetrics
    cm_path = DATA_RAW / "btc" / "onchain_coinmetrics.parquet"
    if btc_ohlcv_path.exists() and cm_path.exists():
        ccxt_df = store.load(btc_ohlcv_path)[0]
        cm_df = store.load(cm_path)[0]
        cross_val = checker.cross_validate_price(ccxt_df, cm_df)
        reports["btc_price_crossval"] = cross_val

    return reports


def main():
    parser = argparse.ArgumentParser(description="Fetch historical crypto data")
    parser.add_argument("--incremental", action="store_true", help="Incremental update only")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without fetching")
    parser.add_argument("--bgeometrics-token", help="BGeometrics API token")
    parser.add_argument("--skip-bgeometrics", action="store_true", help="Skip BGeometrics")
    parser.add_argument("--skip-blockchain-com", action="store_true", help="Skip Blockchain.com")
    parser.add_argument("--skip-coingecko", action="store_true", help="Skip CoinGecko")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN — would fetch:")
        logger.info("  BTC/USDT + ETH/USDT OHLCV from Binance")
        logger.info("  BTC on-chain from BGeometrics (9 computed indicators)")
        logger.info("  BTC + ETH on-chain from CoinMetrics (10+ raw metrics each)")
        logger.info("  BTC on-chain from Blockchain.com (8 metrics, 5 years)")
        logger.info("  Market context snapshot from CoinGecko (top 250 coins)")
        return

    all_results = {}

    # 1. Price data (most important, fewest rate limit concerns)
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching price data (CCXT/Binance)")
    logger.info("=" * 60)
    all_results["price"] = fetch_price_data(args.incremental)

    # 2. CoinMetrics (free, no auth, generous rate limit)
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching CoinMetrics on-chain data")
    logger.info("=" * 60)
    all_results["coinmetrics"] = fetch_coinmetrics_data(args.incremental)

    # 3. BGeometrics (limited: 8 req/hour, 15 req/day)
    if not args.skip_bgeometrics:
        logger.info("=" * 60)
        logger.info("STEP 3: Fetching BGeometrics on-chain data")
        logger.info("=" * 60)
        all_results["bgeometrics"] = fetch_bgeometrics_data(
            args.incremental, args.bgeometrics_token
        )

    # 4. Blockchain.com (validation reference)
    if not args.skip_blockchain_com:
        logger.info("=" * 60)
        logger.info("STEP 4: Fetching Blockchain.com reference data")
        logger.info("=" * 60)
        all_results["blockchain_com"] = fetch_blockchain_com_data()

    # 5. CoinGecko (market context snapshot)
    if not args.skip_coingecko:
        logger.info("=" * 60)
        logger.info("STEP 5: Fetching CoinGecko market context")
        logger.info("=" * 60)
        all_results["coingecko"] = fetch_coingecko_data()

    # 6. Source selection
    logger.info("=" * 60)
    logger.info("STEP 6: Running source selection")
    logger.info("=" * 60)
    all_results["source_selection"] = run_source_selection()

    # 7. Quality checks
    logger.info("=" * 60)
    logger.info("STEP 7: Running quality checks")
    logger.info("=" * 60)
    all_results["quality"] = run_quality_checks()

    # Save summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "fetch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"DONE — summary saved to {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
