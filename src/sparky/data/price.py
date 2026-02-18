"""CCXT-backed price and derivatives data fetcher.

Uses CCXT library with Binance as default, failover to Bybit/OKX.
Handles pagination, validation, and rate limiting.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# CCXT fetchOHLCV returns max 1000 candles per request
MAX_CANDLES_PER_REQUEST = 1000
MS_PER_DAY = 86_400_000


class CCXTPriceFetcher:
    """Fetch daily OHLCV data via CCXT.

    Usage:
        fetcher = CCXTPriceFetcher()
        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2017-01-01", "2024-12-31")
    """

    FAILOVER_EXCHANGES = ["bybit", "okx", "coinbase"]

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id
        self.exchange = self._create_exchange(exchange_id)

    def _create_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Create a CCXT exchange instance with rate limiting enabled."""
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({"enableRateLimit": True})

    def fetch_daily_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV candles with pagination.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            start_date: Start date as "YYYY-MM-DD".
            end_date: End date as "YYYY-MM-DD" (default: today).

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume, quote_volume.
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        logger.info(
            f"[DATA] Fetching {symbol} daily OHLCV from {self.exchange_id} ({start_date} to {end_date or 'now'})"
        )

        all_candles = self._fetch_with_failover(symbol, start_ts, end_ts)

        if not all_candles:
            logger.warning(f"[DATA] No candles returned for {symbol}")
            return pd.DataFrame()

        df = self._candles_to_dataframe(all_candles)
        df = self._validate(df, symbol)

        logger.info(
            f"[DATA] Fetched {len(df)} daily candles for {symbol} ({df.index.min().date()} to {df.index.max().date()})"
        )
        return df

    def _fetch_with_failover(self, symbol: str, start_ts: int, end_ts: int) -> list:
        """Fetch candles, falling back to alternative exchanges on failure."""
        exchanges = [self.exchange_id] + self.FAILOVER_EXCHANGES
        original_exchange = self.exchange

        for exch_id in exchanges:
            try:
                if exch_id != self.exchange_id:
                    logger.info(f"[DATA] Failing over to {exch_id}")
                    self.exchange = self._create_exchange(exch_id)

                result = self._paginated_fetch(symbol, start_ts, end_ts)
                # Restore original exchange after successful failover
                self.exchange = original_exchange
                return result

            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"[DATA] {exch_id} failed: {e}")
                continue

        # Restore original exchange even on total failure
        self.exchange = original_exchange
        logger.error(f"[DATA] All exchanges failed for {symbol}")
        return []

    def _paginated_fetch(self, symbol: str, start_ts: int, end_ts: int) -> list:
        """Fetch all candles via pagination (CCXT limit per request)."""
        all_candles = []
        since = start_ts

        while since < end_ts:
            candles = self.exchange.fetch_ohlcv(symbol, "1d", since=since, limit=MAX_CANDLES_PER_REQUEST)

            if not candles:
                break

            all_candles.extend(candles)

            # Move to after the last candle
            last_ts = candles[-1][0]
            if last_ts <= since:
                break  # Prevent infinite loop
            since = last_ts + MS_PER_DAY

        return all_candles

    def _candles_to_dataframe(self, candles: list) -> pd.DataFrame:
        """Convert CCXT candle list to DataFrame.

        CCXT format: [timestamp_ms, open, high, low, close, volume]
        """
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df

    def _validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLCV data quality.

        Checks: no duplicates, prices > 0, volume >= 0.
        """
        # Remove any remaining duplicates
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < before:
            logger.warning(f"[DATA] Removed {before - len(df)} duplicate timestamps for {symbol}")

        # Validate prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            invalid = df[col] <= 0
            if invalid.any():
                logger.warning(f"[DATA] {invalid.sum()} non-positive {col} values for {symbol}")
                df = df[~invalid]

        # Validate volume
        invalid_vol = df["volume"] < 0
        if invalid_vol.any():
            logger.warning(f"[DATA] {invalid_vol.sum()} negative volume values for {symbol}")
            df = df[~invalid_vol]

        return df
