"""CCXT-backed funding rate data fetcher."""

import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

from sparky.data.storage import DataStore

logger = logging.getLogger(__name__)

__all__ = ["FundingRateFetcher", "sync_funding_rates"]

MAX_RECORDS_PER_REQUEST = 1000


class FundingRateFetcher:
    """Fetch historical funding rates via CCXT.

    Factory methods:
        .binance() — 8h granularity, BTC/USDT:USDT
        .hyperliquid() — 1h granularity, BTC/USDC:USDC
        .coinbase() — 1h granularity, BTC/USD:USD
    """

    def __init__(self, exchange_id: str, symbol: str, granularity: str):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.granularity = granularity
        self.exchange = self._create_exchange(exchange_id)

    def _create_exchange(self, exchange_id: str) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({"enableRateLimit": True})

    @classmethod
    def binance(cls, asset: str = "BTC") -> "FundingRateFetcher":
        return cls("binance", f"{asset}/USDT:USDT", "8h")

    @classmethod
    def hyperliquid(cls, asset: str = "BTC") -> "FundingRateFetcher":
        return cls("hyperliquid", f"{asset}/USDC:USDC", "1h")

    @classmethod
    def coinbase(cls, asset: str = "BTC") -> "FundingRateFetcher":
        return cls("coinbase", f"{asset}/USD:USD", "1h")

    def fetch_funding_rates(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch funding rates with pagination.

        Returns DataFrame with DatetimeIndex (UTC) and columns:
        funding_rate, exchange, granularity
        """
        start_dt = pd.Timestamp(start_date, tz="UTC")
        start_ts = int(start_dt.timestamp() * 1000)
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        logger.info(f"[DATA] Fetching {self.symbol} funding rates from {self.exchange_id}")

        all_records = self._paginated_fetch(start_ts, end_ts)

        if not all_records:
            logger.warning(f"[DATA] No funding rates returned for {self.symbol}")
            return pd.DataFrame()

        df = self._records_to_dataframe(all_records)
        df = self._validate(df)
        return df

    def _paginated_fetch(self, start_ts: int, end_ts: int) -> list:
        all_records = []
        since = start_ts
        while since < end_ts:
            try:
                records = self.exchange.fetch_funding_rate_history(
                    self.symbol, since=since, limit=MAX_RECORDS_PER_REQUEST
                )
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"[DATA] Funding rate fetch failed: {e}")
                break

            if not records:
                break

            all_records.extend(records)
            last_ts = records[-1].get("timestamp", 0)
            if last_ts <= since:
                break
            since = last_ts + 1

        return all_records

    def _records_to_dataframe(self, records: list) -> pd.DataFrame:
        rows = []
        for r in records:
            ts = r.get("timestamp")
            rate = r.get("fundingRate")
            if ts is not None and rate is not None:
                rows.append(
                    {
                        "timestamp": pd.Timestamp(ts, unit="ms", tz="UTC"),
                        "funding_rate": float(rate),
                        "exchange": self.exchange_id,
                        "granularity": self.granularity,
                    }
                )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index("timestamp")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        max_gap_hours = 24 if self.granularity == "8h" else 2
        if len(df) > 1:
            gaps = df.index.to_series().diff()
            max_gap = gaps.max()
            if max_gap > pd.Timedelta(hours=max_gap_hours):
                logger.warning(
                    f"[DATA] Gap of {max_gap} detected in {self.exchange_id} funding rates "
                    f"(max allowed: {max_gap_hours}h)"
                )
        return df


def sync_funding_rates(
    asset: str = "BTC",
    exchanges: list[str] | None = None,
) -> dict[str, int]:
    """Sync funding rates for all exchanges and save to parquet.

    Returns dict of {exchange: n_new_rows}.
    """
    store = DataStore()
    if exchanges is None:
        exchanges = ["hyperliquid", "coinbase"]

    # Binance factory preserved but excluded from default sync (451 geo-restriction).
    # Coinbase fetchFundingRateHistory not supported by CCXT — will log warning and skip.
    factory = {
        "binance": FundingRateFetcher.binance,
        "hyperliquid": FundingRateFetcher.hyperliquid,
        "coinbase": FundingRateFetcher.coinbase,
    }

    results = {}
    for exch in exchanges:
        if exch not in factory:
            logger.warning(f"[DATA] Unknown exchange: {exch}")
            continue

        fetcher = factory[exch](asset)
        parquet_path = f"data/raw/funding_rates/{asset.lower()}_{exch}.parquet"

        last_ts = store.get_last_timestamp(parquet_path)
        start = last_ts.isoformat() if last_ts else "2019-01-01"

        try:
            df = fetcher.fetch_funding_rates(start)
        except Exception as e:
            logger.warning(f"[DATA] Failed to fetch funding rates from {exch}: {e}")
            results[exch] = 0
            continue

        if df.empty:
            results[exch] = 0
            continue

        df_save = df[["funding_rate"]].copy()
        meta = {
            "source": exch,
            "asset": asset.lower(),
            "granularity": fetcher.granularity,
        }
        store.append(df_save, parquet_path, metadata=meta)
        results[exch] = len(df_save)
        logger.info(f"[DATA] Synced {len(df_save)} funding rates from {exch}")

    return results
