"""CoinGecko market context data fetcher.

Fetches broad market context data (market cap, volume, supply, ATH distance, etc.)
for top cryptocurrencies. Used for market regime detection and context features.

API details:
- Base URL: https://api.coingecko.com/api/v3
- Auth: Demo key (free signup) or none
- Rate limit: ~30 req/min
- Monthly quota: 10,000 calls on free tier
- Schedule: 1 batch call/day is sufficient
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.coingecko.com/api/v3"
REQUEST_INTERVAL = 2.5  # Conservative: ~24 req/min


class CoinGeckoFetcher:
    """Fetch market context data from CoinGecko.

    Usage:
        fetcher = CoinGeckoFetcher()
        df = fetcher.fetch_market_data(top_n=50)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-cg-demo-key"] = api_key
        self._last_request_time = 0.0
        self._request_count = 0

    def _rate_limit(self) -> None:
        """Enforce polite rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

    def fetch_market_data(
        self,
        top_n: int = 250,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """Fetch current market data for top N coins.

        Args:
            top_n: Number of top coins by market cap.
            vs_currency: Quote currency (default: usd).

        Returns:
            DataFrame with coin ID as index and columns:
            market_cap, total_volume, circulating_supply, fdv,
            price_change_24h_pct, price_change_7d_pct, price_change_30d_pct,
            ath_distance_pct, current_price.
        """
        self._rate_limit()

        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": min(top_n, 250),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "7d,30d",
        }

        try:
            resp = self.session.get(
                f"{BASE_URL}/coins/markets", params=params, timeout=30
            )
            self._last_request_time = time.time()
            self._request_count += 1
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[DATA] CoinGecko request failed: {e}")
            raise

        if not data:
            logger.warning("[DATA] No data returned from CoinGecko")
            return pd.DataFrame()

        records = []
        for coin in data:
            try:
                ath = coin.get("ath", 0)
                current = coin.get("current_price", 0)
                ath_distance = ((current - ath) / ath * 100) if ath else None

                records.append({
                    "coin_id": coin["id"],
                    "symbol": coin.get("symbol", "").upper(),
                    "current_price": current,
                    "market_cap": coin.get("market_cap"),
                    "total_volume": coin.get("total_volume"),
                    "circulating_supply": coin.get("circulating_supply"),
                    "fdv": coin.get("fully_diluted_valuation"),
                    "price_change_24h_pct": coin.get("price_change_percentage_24h"),
                    "price_change_7d_pct": coin.get(
                        "price_change_percentage_7d_in_currency"
                    ),
                    "price_change_30d_pct": coin.get(
                        "price_change_percentage_30d_in_currency"
                    ),
                    "ath_distance_pct": ath_distance,
                })
            except (KeyError, TypeError):
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.set_index("coin_id")

        logger.info(f"[DATA] Fetched market data for {len(df)} coins from CoinGecko")
        return df

    def fetch_historical_market_chart(
        self,
        coin_id: str,
        days: int = 365,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """Fetch historical daily market data for a single coin.

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum").
            days: Number of days of history (max depends on plan).
            vs_currency: Quote currency.

        Returns:
            DataFrame with DatetimeIndex and columns: price, market_cap, volume.
        """
        self._rate_limit()

        try:
            resp = self.session.get(
                f"{BASE_URL}/coins/{coin_id}/market_chart",
                params={
                    "vs_currency": vs_currency,
                    "days": days,
                    "interval": "daily",
                },
                timeout=30,
            )
            self._last_request_time = time.time()
            self._request_count += 1
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[DATA] CoinGecko historical request failed: {e}")
            raise

        if not data.get("prices"):
            logger.warning(f"[DATA] No historical data for {coin_id}")
            return pd.DataFrame()

        # Parse [timestamp_ms, value] arrays
        prices = {
            pd.Timestamp(ts, unit="ms", tz="UTC"): val
            for ts, val in data.get("prices", [])
        }
        market_caps = {
            pd.Timestamp(ts, unit="ms", tz="UTC"): val
            for ts, val in data.get("market_caps", [])
        }
        volumes = {
            pd.Timestamp(ts, unit="ms", tz="UTC"): val
            for ts, val in data.get("total_volumes", [])
        }

        df = pd.DataFrame({
            "price": pd.Series(prices),
            "market_cap": pd.Series(market_caps),
            "volume": pd.Series(volumes),
        })
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            f"[DATA] Fetched {len(df)} days of historical data for {coin_id}"
        )
        return df
