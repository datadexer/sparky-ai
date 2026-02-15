"""CoinMetrics Community API on-chain data fetcher.

Fetches raw on-chain metrics for BTC and ETH from the CoinMetrics
Community API (free, no auth required).

API details:
- Base URL: https://community-api.coinmetrics.io/v4
- Auth: None (Community tier)
- Rate limit: ~1.6 req/sec (10 per 6-second window)
- Python client: coinmetrics-api-client
- Docs: https://docs.coinmetrics.io/api/v4

CoinMetrics provides raw on-chain data (hash rate, active addresses,
tx count, NVT, etc.) but NOT computed indicators (MVRV, SOPR, NUPL).
Those come from BGeometrics.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Metrics per asset (from configs/data_sources.yaml)
ASSET_METRICS = {
    "btc": [
        "HashRate",
        "AdrActCnt",
        "TxCnt",
        "TxTfrValAdjUSD",
        "FeeTotUSD",
        "RevUSD",
        "SplyCur",
        "CapMrktCurUSD",
        "NVTAdj",
        "PriceUSD",
    ],
    "eth": [
        "HashRate",
        "AdrActCnt",
        "TxCnt",
        "TxTfrValAdjUSD",
        "FeeTotUSD",
        "SplyCur",
        "CapMrktCurUSD",
        "NVTAdj",
        "PriceUSD",
        "AdrBalCnt",
    ],
}

# Polite rate limiting
REQUEST_INTERVAL = 0.7  # ~1.4 RPS, under 1.6 limit


class CoinMetricsFetcher:
    """Fetch on-chain metrics from CoinMetrics Community API.

    Usage:
        fetcher = CoinMetricsFetcher()
        df = fetcher.fetch_metrics("btc", ["HashRate", "AdrActCnt"], "2020-01-01")
        df_all = fetcher.fetch_asset_metrics("btc", "2020-01-01")
    """

    def __init__(self):
        try:
            from coinmetrics.api_client import CoinMetricsClient
        except ImportError:
            raise ImportError(
                "coinmetrics-api-client is required. "
                "Install with: pip install coinmetrics-api-client"
            )
        self.client = CoinMetricsClient()
        self._last_request_time = 0.0
        self._request_count = 0

    def _rate_limit(self) -> None:
        """Enforce polite rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

    def fetch_metrics(
        self,
        asset: str,
        metrics: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch multiple metrics for a single asset.

        Args:
            asset: "btc" or "eth".
            metrics: List of CoinMetrics metric names (e.g., ["HashRate", "AdrActCnt"]).
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD" (default: today).

        Returns:
            DataFrame with DatetimeIndex (UTC) and one column per metric.
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        metrics_str = ",".join(metrics)
        logger.info(
            f"[DATA] Fetching CoinMetrics {asset} [{metrics_str}] "
            f"({start_date} to {end_date})"
        )

        self._rate_limit()

        try:
            result = self.client.get_asset_metrics(
                assets=asset,
                metrics=metrics_str,
                start_time=start_date,
                end_time=end_date,
                frequency="1d",
            )
            self._last_request_time = time.time()
            self._request_count += 1

            df = result.to_dataframe()
        except Exception as e:
            logger.error(f"[DATA] CoinMetrics request failed: {e}")
            raise

        if df.empty:
            logger.warning(f"[DATA] No data returned for CoinMetrics {asset}")
            return pd.DataFrame()

        # Parse time column to DatetimeIndex
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")

        # Drop the asset column if present (we know which asset we asked for)
        if "asset" in df.columns:
            df = df.drop(columns=["asset"])

        # Convert metric columns to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            f"[DATA] Fetched {len(df)} rows for CoinMetrics {asset} "
            f"({len(df.columns)} metrics)"
        )
        return df

    def fetch_asset_metrics(
        self,
        asset: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch all configured metrics for an asset.

        Args:
            asset: "btc" or "eth".
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD" (default: today).

        Returns:
            DataFrame with DatetimeIndex and one column per metric.
        """
        asset_lower = asset.lower()
        if asset_lower not in ASSET_METRICS:
            raise ValueError(
                f"Unknown asset '{asset}'. Available: {list(ASSET_METRICS.keys())}"
            )

        metrics = ASSET_METRICS[asset_lower]
        return self.fetch_metrics(asset_lower, metrics, start_date, end_date)

    @property
    def available_assets(self) -> list[str]:
        """List supported assets."""
        return list(ASSET_METRICS.keys())

    def get_available_metrics(self, asset: str) -> list[str]:
        """List available metrics for an asset.

        Args:
            asset: "btc" or "eth".

        Returns:
            List of metric names for the asset, or empty list if unknown.
        """
        return ASSET_METRICS.get(asset.lower(), [])
