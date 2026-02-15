"""Blockchain.com API on-chain data fetcher.

Validation reference source for BTC on-chain metrics.
NOT a primary source — used to cross-validate BGeometrics and CoinMetrics.

API details:
- Base URL: https://api.blockchain.info
- Auth: None
- Rate limit: 10-30 req/min
- Endpoints: /charts/{metric}?timespan=5years&format=json&sampled=false
- Response: {"values": [{"x": unix_ts, "y": value}, ...]}
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.blockchain.info"

# Map our metric names to Blockchain.com chart endpoints
METRIC_ENDPOINTS = {
    "hash_rate": "hash-rate",
    "active_addresses": "n-unique-addresses",
    "transaction_count": "n-transactions",
    "transfer_volume_usd": "estimated-transaction-volume-usd",
    "miner_revenue": "miners-revenue",
    "total_fees_btc": "total-fees-btc",
    "mempool_size": "mempool-size",
    "mempool_count": "mempool-count",
}

REQUEST_INTERVAL = 2.0  # Conservative: 30 req/min max


class BlockchainComFetcher:
    """Fetch BTC on-chain metrics from Blockchain.com.

    Primarily used as a validation reference, not a primary data source.

    Usage:
        fetcher = BlockchainComFetcher()
        df = fetcher.fetch_metric("hash_rate", timespan="5years")
    """

    def __init__(self):
        self.session = requests.Session()
        self._last_request_time = 0.0
        self._request_count = 0

    def _rate_limit(self) -> None:
        """Enforce polite rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

    def fetch_metric(
        self,
        metric_name: str,
        timespan: str = "5years",
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a single BTC metric from Blockchain.com charts API.

        Args:
            metric_name: One of the keys in METRIC_ENDPOINTS.
            timespan: Timespan string (e.g., "5years", "2years", "1year").
            start_date: Optional start date "YYYY-MM-DD" (uses timespan if not given).

        Returns:
            DataFrame with DatetimeIndex (UTC) and single column named metric_name.
        """
        if metric_name not in METRIC_ENDPOINTS:
            available = ", ".join(METRIC_ENDPOINTS.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")

        chart_name = METRIC_ENDPOINTS[metric_name]
        url = f"{BASE_URL}/charts/{chart_name}"

        params = {"format": "json", "sampled": "false"}
        if start_date:
            # Blockchain.com uses start parameter as ISO date
            params["start"] = start_date
        else:
            params["timespan"] = timespan

        self._rate_limit()

        try:
            resp = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            self._request_count += 1
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[DATA] Blockchain.com request failed: {e}")
            raise

        values = data.get("values", [])
        if not values:
            logger.warning(f"[DATA] No data returned for Blockchain.com {metric_name}")
            return pd.DataFrame()

        records = []
        for entry in values:
            try:
                ts = pd.Timestamp(entry["x"], unit="s", tz="UTC")
                val = float(entry["y"])
                records.append({"date": ts, metric_name: val})
            except (KeyError, ValueError, TypeError):
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.set_index("date")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"[DATA] Fetched {len(df)} rows for Blockchain.com {metric_name}")
        return df

    def fetch_all_metrics(
        self,
        timespan: str = "5years",
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch multiple metrics and merge into a single DataFrame.

        Args:
            timespan: Timespan string.
            metrics: List of metric names. Default: all available.

        Returns:
            DataFrame with DatetimeIndex and one column per metric.
        """
        if metrics is None:
            metrics = list(METRIC_ENDPOINTS.keys())

        dfs = []
        for metric in metrics:
            try:
                df = self.fetch_metric(metric, timespan=timespan)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"[DATA] Failed to fetch {metric}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how="outer")

        logger.info(
            f"[DATA] Blockchain.com: {len(result)} rows, "
            f"{len(result.columns)} metrics"
        )
        return result

    @property
    def available_metrics(self) -> list[str]:
        """List available metric names."""
        return list(METRIC_ENDPOINTS.keys())
