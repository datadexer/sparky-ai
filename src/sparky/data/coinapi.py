"""CoinAPI derivatives data fetcher (funding rates + open interest)."""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from sparky.data.storage import DataStore

logger = logging.getLogger(__name__)

BASE_URL = "https://rest.coinapi.io"


class CoinAPIFetcher:
    """Fetch derivatives metrics from CoinAPI (metered — manual use only)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("COINAPI_KEY")
        if not self.api_key:
            key_file = Path("coinapi_key.txt")
            if key_file.exists():
                self.api_key = key_file.read_text().strip()
        if not self.api_key:
            raise ValueError("CoinAPI key required. Set COINAPI_KEY env var or create coinapi_key.txt")
        self.session = requests.Session()
        self.session.headers["X-CoinAPI-Key"] = self.api_key
        self._credits_used = 0.0

    def _get(self, endpoint: str, params: Optional[dict] = None) -> list:
        url = f"{BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=30)
        except requests.RequestException as e:
            logger.error(f"[COINAPI] Request failed: {e}")
            raise

        if resp.status_code == 402:
            raise RuntimeError("CoinAPI credits exhausted (402)")
        if resp.status_code == 429:
            logger.warning("[COINAPI] Rate limited (429), waiting 60s")
            import time

            time.sleep(60)
            resp = self.session.get(url, params=params or {}, timeout=30)
            if resp.status_code == 429:
                raise RuntimeError("CoinAPI rate limited after retry")
            resp.raise_for_status()
        elif resp.status_code == 404:
            logger.warning(f"[COINAPI] 404 for {endpoint}")
            return []
        elif resp.status_code >= 400:
            logger.warning(f"[COINAPI] {resp.status_code} for {endpoint}")
            return []

        # Track estimated cost (~$0.005 per 100 rows)
        data = resp.json()
        self._credits_used += len(data) * 0.00005
        return data

    def discover_symbols(self, exchange_id: str, metric_id: str) -> list[dict]:
        """List available symbols for an exchange+metric combo."""
        data = self._get(
            "/v1/metrics/symbol/listing",
            {"metric_id": metric_id, "exchange_id": exchange_id},
        )
        return data

    def fetch_metric_history(
        self,
        metric_id: str,
        symbol_id: str,
        time_start: str,
        time_end: Optional[str] = None,
        period_id: Optional[str] = None,
        limit: int = 100000,
    ) -> pd.DataFrame:
        """Fetch historical metric data.

        Returns DataFrame with DatetimeIndex (UTC) and 'value' column.
        CoinAPI returns OHLC-style aggregates (first/last/min/max) — we take 'last'.
        Use period_id='8HRS' for 8h funding rate snapshots.
        """
        params = {
            "metric_id": metric_id,
            "symbol_id": symbol_id,
            "time_start": time_start,
            "limit": limit,
        }
        if time_end:
            params["time_end"] = time_end
        if period_id:
            params["period_id"] = period_id

        data = self._get("/v1/metrics/symbol/history", params)
        if not data:
            return pd.DataFrame()

        rows = []
        for record in data:
            ts = record.get("time_period_start") or record.get("time_exchange")
            # CoinAPI returns OHLC aggregates: prefer 'last', fall back to 'value_decimal'/'value'
            val = record.get("last") or record.get("value_decimal") or record.get("value")
            if ts and val is not None:
                try:
                    rows.append(
                        {
                            "timestamp": pd.Timestamp(ts, tz="UTC"),
                            "value": float(val),
                        }
                    )
                except (ValueError, TypeError):
                    continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            f"[COINAPI] Fetched {len(df)} rows for {metric_id} / {symbol_id} (est. cost: ${self._credits_used:.3f})"
        )
        return df

    @property
    def credits_used(self) -> float:
        return self._credits_used


def sync_coinapi(targets: list[dict]) -> dict[str, int]:
    """Sync CoinAPI targets to parquet via DataStore.

    Each target: {metric_id, symbol_id, asset, parquet_name, column_name, period_id?}
    Saves to data/coinapi/{parquet_name}.parquet.
    Returns {parquet_name: n_rows}.
    """
    store = DataStore()
    fetcher = CoinAPIFetcher()
    results = {}

    for target in targets:
        metric_id = target["metric_id"]
        symbol_id = target["symbol_id"]
        parquet_name = target["parquet_name"]
        column_name = target.get("column_name", "value")
        period_id = target.get("period_id")
        parquet_path = Path(f"data/coinapi/{parquet_name}.parquet")

        last_ts = store.get_last_timestamp(parquet_path)
        start = last_ts.isoformat() if last_ts else "2019-01-01T00:00:00"

        try:
            df = fetcher.fetch_metric_history(metric_id, symbol_id, start, period_id=period_id)
        except RuntimeError as e:
            logger.error(f"[COINAPI] Fatal error fetching {parquet_name}: {e}")
            break
        except Exception as e:
            logger.warning(f"[COINAPI] Failed to fetch {parquet_name}: {e}")
            results[parquet_name] = 0
            continue

        if df.empty:
            results[parquet_name] = 0
            continue

        df = df.rename(columns={"value": column_name})
        meta = {
            "source": "coinapi",
            "metric_id": metric_id,
            "symbol_id": symbol_id,
            "asset": target.get("asset", "btc"),
        }
        store.append(df, parquet_path, metadata=meta)
        results[parquet_name] = len(df)
        logger.info(f"[COINAPI] Synced {len(df)} rows to {parquet_path}")

    logger.info(f"[COINAPI] Total estimated cost: ${fetcher.credits_used:.3f}")
    return results
