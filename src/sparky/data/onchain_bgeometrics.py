"""BGeometrics on-chain data fetcher.

Fetches computed on-chain indicators (MVRV, SOPR, NUPL, etc.) from
the BGeometrics/bitcoin-data.com API. BTC only.

API details (discovered during Phase 1):
- Base URL: https://bitcoin-data.com
- Auth: Token via URL parameter (?token=...)
- Rate limit: Free plan = 8 req/hour, 15 req/day (very limited!)
- Response: Paginated Spring Data REST JSON, values as strings
- Swagger: https://bitcoin-data.com/api/swagger-ui/index.html
- Endpoints: /v1/{metric} with ?startday=&endday=&page=&size= params
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://bitcoin-data.com"

# Map our metric names to BGeometrics API endpoints and response field names
METRIC_ENDPOINTS = {
    "mvrv_zscore": {"endpoint": "/v1/mvrv", "field": "mvrv"},
    "sopr": {"endpoint": "/v1/sopr", "field": "sopr"},
    "nupl": {"endpoint": "/v1/nupl", "field": "nupl"},
    "realized_price": {"endpoint": "/v1/realized-price", "field": "realizedPrice"},
    "cdd": {"endpoint": "/v1/cdd", "field": "cdd"},
    "puell_multiple": {"endpoint": "/v1/puell-multiple", "field": "puellMultiple"},
    "active_addresses": {"endpoint": "/v1/active-addresses", "field": "activeAddresses"},
    "hash_rate": {"endpoint": "/v1/hashrate", "field": "hashrate"},
    "supply_in_profit": {"endpoint": "/v1/supply-in-profit", "field": "supplyInProfit"},
}

# Polite rate limiting: 1 request/second + respect 8 req/hour limit
REQUEST_INTERVAL = 1.0
MAX_PAGE_SIZE = 5000  # Request large pages to minimize API calls


class BGeometricsFetcher:
    """Fetch BTC on-chain computed indicators from BGeometrics.

    Usage:
        fetcher = BGeometricsFetcher(token="your_token")
        df = fetcher.fetch_metric("sopr", "2020-01-01", "2024-12-31")
        df_all = fetcher.fetch_all_metrics("2020-01-01", "2024-12-31")
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.session = requests.Session()
        self._last_request_time = 0.0
        self._request_count = 0

    def _rate_limit(self) -> None:
        """Enforce polite rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a rate-limited GET request."""
        self._rate_limit()

        url = f"{BASE_URL}{endpoint}"
        if self.token:
            params["token"] = self.token

        try:
            resp = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            self._request_count += 1
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[DATA] BGeometrics request failed: {e}")
            raise

    def fetch_metric(
        self,
        metric_name: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a single on-chain metric.

        Args:
            metric_name: One of the keys in METRIC_ENDPOINTS.
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD" (default: today).

        Returns:
            DataFrame with DatetimeIndex (UTC) and single column named metric_name.
        """
        if metric_name not in METRIC_ENDPOINTS:
            available = ", ".join(METRIC_ENDPOINTS.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")

        config = METRIC_ENDPOINTS[metric_name]
        endpoint = config["endpoint"]
        field = config["field"]

        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info(f"[DATA] Fetching BGeometrics {metric_name} ({start_date} to {end_date})")

        all_records = []
        page = 0

        while True:
            params = {
                "startday": start_date,
                "endday": end_date,
                "page": page,
                "size": MAX_PAGE_SIZE,
            }

            data = self._get(endpoint, params)
            content = data.get("content", [])

            if not content:
                break

            for record in content:
                date_str = record.get("d")
                value_str = record.get(field)
                if date_str and value_str is not None:
                    try:
                        all_records.append({
                            "date": pd.Timestamp(date_str, tz="UTC"),
                            metric_name: float(value_str),
                        })
                    except (ValueError, TypeError):
                        continue

            # Check if we've reached the last page
            if data.get("last", True):
                break

            page += 1

        if not all_records:
            logger.warning(f"[DATA] No data returned for BGeometrics {metric_name}")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.set_index("date")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"[DATA] Fetched {len(df)} rows for BGeometrics {metric_name}")
        return df

    def fetch_all_metrics(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch multiple metrics and merge into a single DataFrame.

        Args:
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD" (default: today).
            metrics: List of metric names. Default: all available.

        Returns:
            DataFrame with DatetimeIndex and one column per metric.
        """
        if metrics is None:
            metrics = list(METRIC_ENDPOINTS.keys())

        dfs = []
        for metric in metrics:
            try:
                df = self.fetch_metric(metric, start_date, end_date)
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
            f"[DATA] BGeometrics: {len(result)} rows, "
            f"{len(result.columns)} metrics, "
            f"{self._request_count} API calls used"
        )
        return result

    @property
    def available_metrics(self) -> list[str]:
        """List available metric names."""
        return list(METRIC_ENDPOINTS.keys())
