"""BGeometrics daily incremental sync — priority-ordered.

STH-SOPR and STH-MVRV first, then remaining P002 metrics.
Stops gracefully on 429. Skips metrics already synced within 2 days
to avoid wasting rate-limited requests on no-op checks.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Priority order: critical P002 metrics first
PRIORITY_METRICS = [
    # Tier 1 — needed for STH regime signals
    "sth_sopr",
    "sth_mvrv",
    # Tier 2 — holder segmentation
    "lth_sopr",
    "lth_mvrv",
    "nupl_sth",
    "nupl_lth",
    # Tier 3 — exchange flows
    "exchange_inflow_btc",
    "exchange_outflow_btc",
    "exchange_netflow_btc",
    "exchange_reserve_btc",
    # Tier 4 — whale / LTH behavior
    "lth_position_change_30d",  # transient 500s — retries next day
    # Tier 5 — additional indicators
    "vdd_multiple",
    "realized_pl_ratio",  # transient 500s — retries next day
    # Tier 6 — best-effort
    "stablecoin_supply",
    "etf_btc_total",  # CSV-only endpoint
    # "open_interest_futures",  # DEPRECATED — ends Oct 2024, 701 rows kept for backtesting
    "funding_rate_aggregate",
]

STALE_DAYS = 2  # Only re-sync metrics whose last row is older than this


def get_stale_metrics() -> list[str]:
    """Return PRIORITY_METRICS that are missing or stale (last row > STALE_DAYS old)."""
    from sparky.data.storage import DataStore

    store = DataStore()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=STALE_DAYS)
    stale = []

    for metric in PRIORITY_METRICS:
        path = Path(f"data/raw/onchain/bgeometrics/{metric}.parquet")
        last_ts = store.get_last_timestamp(path)
        if last_ts is None:
            stale.append(metric)
        elif last_ts < cutoff:
            stale.append(metric)
        else:
            logger.info(f"Skipping {metric} (up-to-date: {last_ts.date()})")

    return stale


def main():
    from sparky.data.onchain_bgeometrics import sync_bgeometrics

    stale = get_stale_metrics()
    if not stale:
        logger.info("All metrics up-to-date. Nothing to sync.")
        return

    logger.info(f"Syncing {len(stale)}/{len(PRIORITY_METRICS)} stale/missing metrics")
    results = sync_bgeometrics(metrics=stale)

    synced = {k: v for k, v in results.items() if v > 0}
    failed = {k: v for k, v in results.items() if v == 0}
    not_reached = [m for m in stale if m not in results]

    logger.info(f"Synced: {len(synced)} metrics, {sum(synced.values())} total rows")
    for metric, count in synced.items():
        logger.info(f"  {metric}: {count} rows")

    if failed:
        logger.info(f"No new data: {list(failed.keys())}")
    if not_reached:
        logger.info(f"Not reached (rate limited): {not_reached}")


if __name__ == "__main__":
    main()
