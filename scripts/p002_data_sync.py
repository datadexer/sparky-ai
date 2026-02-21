"""P002 initial data backfill — one-time sync of all new data sources."""

import logging
import sys
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sync_funding_rates_all():
    """Sync funding rates for BTC + ETH across all exchanges."""
    from sparky.data.funding_rate import sync_funding_rates

    results = {}
    for asset in ["BTC", "ETH"]:
        logger.info(f"=== Funding Rates: {asset} ===")
        try:
            r = sync_funding_rates(asset=asset)
            results[asset] = r
            for exch, count in r.items():
                logger.info(f"  {exch}: {count} rows")
        except Exception:
            logger.error(f"Failed {asset} funding rates:\n{traceback.format_exc()}")
            results[asset] = {"error": traceback.format_exc()}
    return results


def sync_coinmetrics_btc():
    """Sync CoinMetrics BTC metrics (including new flow metrics)."""
    from sparky.data.onchain_coinmetrics import CoinMetricsFetcher
    from sparky.data.storage import DataStore

    logger.info("=== CoinMetrics BTC ===")
    store = DataStore()
    path = Path("data/raw/onchain/coinmetrics_btc.parquet")

    try:
        fetcher = CoinMetricsFetcher()
        last_ts = store.get_last_timestamp(path)
        start = last_ts.strftime("%Y-%m-%d") if last_ts else "2010-01-01"
        logger.info(f"  Fetching from {start}")

        df = fetcher.fetch_asset_metrics("btc", start)
        if not df.empty:
            store.append(df, path, metadata={"source": "coinmetrics", "asset": "btc"})
            logger.info(f"  Synced {len(df)} rows, columns: {list(df.columns)}")
            if "FlowInExNtv" in df.columns:
                logger.info(f"  FlowInExNtv: {df['FlowInExNtv'].dropna().shape[0]} non-null")
            if "FlowOutExNtv" in df.columns:
                logger.info(f"  FlowOutExNtv: {df['FlowOutExNtv'].dropna().shape[0]} non-null")
        else:
            logger.info("  No new data")
        return {"rows": len(df), "columns": list(df.columns) if not df.empty else []}
    except Exception:
        logger.error(f"CoinMetrics failed:\n{traceback.format_exc()}")
        return {"error": traceback.format_exc()}


def sync_bgeometrics_all():
    """Sync all BGeometrics metrics (rate limited — will stop on 429)."""
    from sparky.data.onchain_bgeometrics import sync_bgeometrics

    logger.info("=== BGeometrics (rate limited — 15 req/day) ===")
    try:
        results = sync_bgeometrics()
        for metric, count in results.items():
            status = f"{count} rows" if count > 0 else "no new data"
            logger.info(f"  {metric}: {status}")
        return results
    except Exception:
        logger.error(f"BGeometrics failed:\n{traceback.format_exc()}")
        return {"error": traceback.format_exc()}


def verify_datasets():
    """Verify all datasets are accessible via loader."""
    from sparky.data.loader import load

    logger.info("\n=== Dataset Verification ===")
    inventory = []

    # Funding rates
    fr_aliases = [
        "funding_rate_btc_binance",
        "funding_rate_eth_binance",
        "funding_rate_btc_hyperliquid",
        "funding_rate_eth_hyperliquid",
        "funding_rate_btc_coinbase",
        "funding_rate_eth_coinbase",
    ]
    for alias in fr_aliases:
        try:
            df = load(alias, purpose="analysis")
            entry = {
                "dataset": alias,
                "rows": len(df),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "status": "complete",
            }
        except Exception as e:
            entry = {"dataset": alias, "rows": 0, "status": f"MISSING — {e}"}
        inventory.append(entry)
        logger.info(f"  {entry['dataset']}: {entry.get('rows', 0)} rows — {entry['status']}")

    # BGeometrics advanced
    bg_aliases = [
        "btc_sth_sopr",
        "btc_lth_sopr",
        "btc_sth_mvrv",
        "btc_lth_mvrv",
        "btc_nupl_sth",
        "btc_nupl_lth",
        "btc_exchange_inflow",
        "btc_exchange_outflow",
        "btc_exchange_netflow",
        "btc_exchange_reserve",
        "btc_lth_position_change_30d",
        "btc_open_interest_futures",
        "btc_funding_rate_aggregate",
        "btc_stablecoin_supply",
        "btc_etf_aggregate",
        "btc_vdd_multiple",
        "btc_realized_pl_ratio",
    ]
    for alias in bg_aliases:
        try:
            df = load(alias, purpose="analysis")
            entry = {
                "dataset": alias,
                "rows": len(df),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "status": "complete",
            }
        except Exception as e:
            entry = {"dataset": alias, "rows": 0, "status": f"MISSING — {e}"}
        inventory.append(entry)
        logger.info(f"  {entry['dataset']}: {entry.get('rows', 0)} rows — {entry['status']}")

    # CoinMetrics
    try:
        df = load("btc_coinmetrics", purpose="analysis")
        cols = list(df.columns)
        has_flow = "FlowInExNtv" in cols and "FlowOutExNtv" in cols
        entry = {
            "dataset": "btc_coinmetrics",
            "rows": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "status": "complete" if has_flow else "MISSING flow metrics",
            "columns": cols,
        }
    except Exception as e:
        entry = {"dataset": "btc_coinmetrics", "rows": 0, "status": f"MISSING — {e}"}
    inventory.append(entry)
    logger.info(f"  btc_coinmetrics: {entry.get('rows', 0)} rows — {entry['status']}")

    return inventory


def write_inventory(inventory, fr_results, cm_results, bg_results):
    """Write sync inventory to data/sync_inventory.md."""
    lines = ["# P002 Data Sync Inventory\n"]
    lines.append(f"Generated: {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}\n")

    lines.append("\n## Funding Rates\n")
    lines.append("| Dataset | Rows | Start | End | Status |")
    lines.append("|---------|------|-------|-----|--------|")
    for entry in inventory:
        if entry["dataset"].startswith("funding_rate"):
            lines.append(
                f"| {entry['dataset']} | {entry.get('rows', 0)} | "
                f"{entry.get('start', 'N/A')} | {entry.get('end', 'N/A')} | {entry['status']} |"
            )

    lines.append("\n## BGeometrics On-Chain\n")
    lines.append("| Dataset | Rows | Start | End | Status |")
    lines.append("|---------|------|-------|-----|--------|")
    for entry in inventory:
        if entry["dataset"].startswith("btc_") and entry["dataset"] != "btc_coinmetrics":
            lines.append(
                f"| {entry['dataset']} | {entry.get('rows', 0)} | "
                f"{entry.get('start', 'N/A')} | {entry.get('end', 'N/A')} | {entry['status']} |"
            )

    lines.append("\n## CoinMetrics\n")
    for entry in inventory:
        if entry["dataset"] == "btc_coinmetrics":
            lines.append(f"- **btc_coinmetrics**: {entry.get('rows', 0)} rows, {entry['status']}")
            if "columns" in entry:
                lines.append(f"- Columns: {', '.join(entry['columns'])}")

    lines.append("\n## Notes\n")
    lines.append(
        "- BGeometrics rate limit: 15 req/day. Re-run `scripts/p002_data_sync.py` daily until all metrics backfilled."
    )
    lines.append("- Loader handles IS/OOS truncation automatically.")

    Path("data/sync_inventory.md").write_text("\n".join(lines) + "\n")
    logger.info("Wrote data/sync_inventory.md")


if __name__ == "__main__":
    logger.info("P002 Data Sync — Initial Backfill")
    logger.info("=" * 50)

    # 1. Funding rates (fastest)
    fr_results = sync_funding_rates_all()

    # 2. CoinMetrics (fast, no rate limit issues)
    cm_results = sync_coinmetrics_btc()

    # 3. BGeometrics (slow, rate limited)
    bg_results = sync_bgeometrics_all()

    # 4. Verify everything
    inventory = verify_datasets()

    # 5. Write inventory
    write_inventory(inventory, fr_results, cm_results, bg_results)

    logger.info("\nDone. Check data/sync_inventory.md for full report.")
