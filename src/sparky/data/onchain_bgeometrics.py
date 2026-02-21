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
- Derivatives (funding rate, OI, basis): Advanced-only — NOT available on free tier

RATE LIMIT STRATEGY (free tier):
- 8 requests per hour, 15 per day — every request is precious
- Cache aggressively: one full historical fetch per metric → Parquet, never re-fetch
- Always use incremental fetches (get_last_timestamp → fetch only delta)
- On 429: log warning and STOP — do not retry and burn more quota
- CoinMetrics Community is fallback for any metric BGeometrics can't serve
- Free tier is sufficient through Phase 4
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from sparky.data.storage import DataStore

logger = logging.getLogger(__name__)

BASE_URL = "https://bitcoin-data.com"
_MODULE_DIR = Path(__file__).resolve().parent
RATE_LIMIT_STATE_PATH = _MODULE_DIR.parents[2] / "data" / ".bgeometrics_rate_limit.json"

# Map our metric names to BGeometrics API endpoints and response field names.
# NOTE: Only computed indicators available on free tier.
# Derivatives (funding rate, open interest, basis) require Advanced plan — skip entirely.
# /v1/ endpoints support date filtering (startday/endday params).
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
    # Holder-segmented metrics
    "sth_sopr": {"endpoint": "/v1/sth-sopr", "field": "sthSopr"},
    "lth_sopr": {"endpoint": "/v1/lth-sopr", "field": "lthSopr"},
    "sth_mvrv": {"endpoint": "/v1/sth-mvrv", "field": "sthMvrv"},
    "lth_mvrv": {"endpoint": "/v1/lth-mvrv", "field": "lthMvrv"},
    "nupl_sth": {"endpoint": "/v1/nupl-sth", "field": "nuplSth"},
    "nupl_lth": {"endpoint": "/v1/nupl-lth", "field": "nuplLth"},
    # Exchange flows
    "exchange_inflow_btc": {"endpoint": "/v1/exchange-inflow-btc", "field": "exchangeInflowBtc"},
    "exchange_outflow_btc": {
        "endpoint": "/v1/exchange-outflow-btc",
        "field": "exchangeOutflowBtc",
    },
    "exchange_netflow_btc": {
        "endpoint": "/v1/exchange-netflow-btc",
        "field": "exchangeNetflowBtc",
    },
    "exchange_reserve_btc": {
        "endpoint": "/v1/exchange-reserve-btc",
        "field": "exchangeReserveBtc",
    },
    # Whale / LTH behavior
    "lth_position_change_30d": {
        "endpoint": "/v1/lth-net-position-change-30d-btc",
        "field": "lthNetPositionChange30dBtc",
    },
    # Derivatives (best-effort — may 403 on free tier)
    "open_interest_futures": {
        "endpoint": "/v1/open-interest-futures",
        "field": "openInterestFutures",
        "best_effort": True,
    },
    "funding_rate_aggregate": {
        "endpoint": "/v1/funding-rate",
        "field": "fundingRate",
        "best_effort": True,
    },
    # Stablecoin & ETF (multi-field)
    "stablecoin_supply": {
        "endpoint": "/v1/stablecoin-supply",
        "field": None,
        "multi_field": True,
    },
    "etf_aggregate": {
        "endpoint": "/v1/etf-aggregate",
        "field": None,
        "multi_field": True,
    },
    # Additional indicators
    "vdd_multiple": {"endpoint": "/v1/vdd-multiple", "field": "vddMultiple"},
    "realized_pl_ratio": {
        "endpoint": "/v1/realized-profit-loss-ratio",
        "field": "realizedProfitLossRatio",
    },
}

# Polite rate limiting: 1 request/second + respect 8 req/hour limit
REQUEST_INTERVAL = 1.0
MAX_PAGE_SIZE = 5000  # Request large pages to minimize API calls
HOURLY_REQUEST_BUDGET = 8  # Free tier: 8 req/hour


class BGeometricsFetcher:
    """Fetch BTC on-chain computed indicators from BGeometrics.

    Usage:
        fetcher = BGeometricsFetcher(token="your_token")
        df = fetcher.fetch_metric("sopr", "2020-01-01", "2024-12-31")
        df_all = fetcher.fetch_all_metrics("2020-01-01", "2024-12-31")
    """

    def __init__(self, token: Optional[str] = None, rate_limit_path: Optional[Path] = None):
        self.token = token or os.environ.get("BGEOMETRICS_TOKEN")
        self.session = requests.Session()
        self._rate_limit_path = rate_limit_path or RATE_LIMIT_STATE_PATH
        self._last_request_time = 0.0
        self._request_count = 0
        self._rate_limited = False  # Set to True if we hit 429
        self._last_hour_reset: Optional[datetime] = None
        self._load_rate_limit_state()

    def _load_rate_limit_state(self) -> None:
        """Load persistent rate limit state from disk."""
        try:
            if self._rate_limit_path.exists():
                with open(self._rate_limit_path) as f:
                    state = json.load(f)
                self._request_count = state.get("request_count", 0)
                # Support both old field name (last_reset_hour) and new (last_hour_reset)
                last_reset = state.get("last_hour_reset") or state.get("last_reset_hour")
                if last_reset:
                    reset_time = datetime.fromisoformat(last_reset)
                    now = datetime.now(timezone.utc)
                    if (now - reset_time).total_seconds() > 3600:
                        self._request_count = 0
                        self._last_hour_reset = now
                    else:
                        self._last_hour_reset = reset_time
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(f"[DATA] Could not load rate limit state: {e}")

    def _save_rate_limit_state(self) -> None:
        """Persist rate limit state to disk."""
        try:
            self._rate_limit_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            state = {
                "request_count": self._request_count,
                "last_request_time": now.isoformat(),
                "last_hour_reset": (self._last_hour_reset or now).isoformat(),
            }
            with open(self._rate_limit_path, "w") as f:
                json.dump(state, f)
        except (OSError, TypeError) as e:
            logger.warning(f"[DATA] Could not save rate limit state: {e}")

    def _rate_limit(self) -> None:
        """Enforce polite rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

    def _check_budget(self) -> None:
        """Warn if approaching hourly request budget."""
        if self._request_count >= HOURLY_REQUEST_BUDGET:
            logger.warning(
                f"[DATA] BGeometrics: {self._request_count} requests used "
                f"(hourly budget: {HOURLY_REQUEST_BUDGET}). "
                "Consider stopping to avoid rate limits."
            )

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a rate-limited GET request.

        On 429 (rate limited): logs warning and raises immediately.
        Do NOT retry — every request burns quota on the free tier.
        """
        if self._rate_limited:
            raise RuntimeError(
                "BGeometrics rate limit hit this session. Wait for the next hour window before retrying."
            )

        self._check_budget()
        self._rate_limit()

        url = f"{BASE_URL}{endpoint}"
        if self.token:
            params["token"] = self.token

        try:
            resp = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            self._request_count += 1
            self._save_rate_limit_state()

            if resp.status_code == 429:
                self._rate_limited = True
                logger.warning(
                    f"[DATA] BGeometrics 429 rate limited after "
                    f"{self._request_count} requests. STOPPING — do not retry. "
                    f"Wait for next hour window."
                )
                raise requests.HTTPError(
                    f"429 Rate Limited after {self._request_count} requests",
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[DATA] BGeometrics request failed: {e}")
            raise

    def _discover_field(self, endpoint: str) -> list[str]:
        """Fetch 1 record from endpoint and return non-'d' field names."""
        try:
            data = self._get(endpoint, {"page": 0, "size": 1})
            content = data if isinstance(data, list) else data.get("content", [])
            if content:
                return [k for k in content[0].keys() if k != "d"]
        except Exception as e:
            logger.debug(f"[DATA] Field discovery failed for {endpoint}: {e}")
        return []

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
            DataFrame with DatetimeIndex (UTC) and column(s) for the metric.
            Multi-field endpoints return multiple columns named {metric}_{field}.
        """
        if metric_name not in METRIC_ENDPOINTS:
            available = ", ".join(METRIC_ENDPOINTS.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")

        config = METRIC_ENDPOINTS[metric_name]
        endpoint = config["endpoint"]
        field = config["field"]
        is_best_effort = config.get("best_effort", False)
        is_multi_field = config.get("multi_field", False)

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

            try:
                data = self._get(endpoint, params)
            except requests.HTTPError as e:
                if (
                    is_best_effort
                    and hasattr(e, "response")
                    and e.response is not None
                    and e.response.status_code in (401, 403)
                ):
                    logger.warning(
                        f"[DATA] Best-effort metric {metric_name} returned {e.response.status_code}, skipping"
                    )
                    return pd.DataFrame()
                raise

            # Handle both paginated dict and raw list responses
            if isinstance(data, list):
                content = data
                is_last = True
            else:
                content = data.get("content", [])
                is_last = data.get("last", True)

            if not content:
                break

            if is_multi_field:
                for record in content:
                    date_str = record.get("d")
                    if not date_str:
                        continue
                    row = {"date": pd.Timestamp(date_str, tz="UTC")}
                    for k, v in record.items():
                        if k == "d":
                            continue
                        try:
                            row[f"{metric_name}_{k}"] = float(v)
                        except (ValueError, TypeError):
                            continue
                    if len(row) > 1:
                        all_records.append(row)
            else:
                for record in content:
                    date_str = record.get("d")
                    value_str = record.get(field)
                    if date_str and value_str is not None:
                        try:
                            all_records.append(
                                {
                                    "date": pd.Timestamp(date_str, tz="UTC"),
                                    metric_name: float(value_str),
                                }
                            )
                        except (ValueError, TypeError):
                            continue

            if is_last:
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
            except RuntimeError:
                # Rate limit hit — stop fetching entirely, return what we have
                logger.warning(f"[DATA] Rate limit hit, stopping. Got {len(dfs)} metrics so far.")
                break
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
    def request_count(self) -> int:
        """Number of API requests made by this instance."""
        return self._request_count

    @property
    def available_metrics(self) -> list[str]:
        """List available metric names."""
        return list(METRIC_ENDPOINTS.keys())


def sync_bgeometrics(
    metrics: list[str] | None = None,
    token: str | None = None,
) -> dict[str, int]:
    """Fetch on-chain metrics and save to parquet via DataStore.

    Incremental: checks DataStore.get_last_timestamp() and fetches only new dates.
    Saves one parquet per metric to data/raw/onchain/bgeometrics/{metric}.parquet.
    Also saves a wide-join parquet: data/raw/onchain/bgeometrics_combined.parquet.
    Returns dict of {metric: n_new_rows}.
    """
    store = DataStore()
    fetcher = BGeometricsFetcher(token=token)
    if metrics is None:
        metrics = list(METRIC_ENDPOINTS.keys())

    results = {}
    all_dfs = []

    for metric in metrics:
        parquet_path = Path(f"data/raw/onchain/bgeometrics/{metric}.parquet")

        last_ts = store.get_last_timestamp(parquet_path)
        if last_ts:
            start = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = "2010-01-01"  # BGeometrics has data from 2009

        try:
            df = fetcher.fetch_metric(metric, start)
        except RuntimeError:
            logger.warning(f"Rate limit hit during sync of {metric}. Stopping.")
            break
        except Exception as e:
            logger.warning(f"Failed to fetch {metric}: {e}")
            results[metric] = 0
            continue

        if df.empty:
            logger.info(f"No new data for {metric}")
            results[metric] = 0
        else:
            meta = {"source": "bgeometrics", "metric": metric, "asset": "btc"}
            store.append(df, parquet_path, metadata=meta)
            results[metric] = len(df)
            logger.info(f"Synced {len(df)} new rows for {metric}")

        if parquet_path.exists():
            full_df, _ = store.load(parquet_path)
            all_dfs.append(full_df)

    if all_dfs:
        combined = all_dfs[0]
        for df in all_dfs[1:]:
            combined = combined.join(df, how="outer")
        combined = combined.sort_index()

        combined_path = Path("data/raw/onchain/bgeometrics_combined.parquet")
        store.save(
            combined,
            combined_path,
            metadata={
                "source": "bgeometrics",
                "asset": "btc",
                "metrics": list(combined.columns),
                "n_metrics": len(combined.columns),
            },
        )
        logger.info(f"Saved combined parquet: {len(combined)} rows, {len(combined.columns)} metrics")

    return results
