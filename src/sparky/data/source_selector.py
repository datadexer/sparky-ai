"""Source selector for on-chain metrics.

Cross-validates overlapping metrics from BGeometrics, CoinMetrics,
and Blockchain.com (reference), then selects the best source per metric
based on completeness, freshness, and agreement with reference.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Overlapping BTC metrics across sources (our name → source field mapping)
# These metrics can be cross-validated between sources
OVERLAPPING_METRICS = {
    "hash_rate": {
        "bgeometrics": "hash_rate",
        "coinmetrics": "HashRate",
        "blockchain_com": "hash_rate",
    },
    "active_addresses": {
        "bgeometrics": "active_addresses",
        "coinmetrics": "AdrActCnt",
        "blockchain_com": "active_addresses",
    },
}

# BGeometrics-exclusive computed indicators (no alternative source)
BGEOMETRICS_EXCLUSIVE = [
    "mvrv_zscore",
    "sopr",
    "nupl",
    "realized_price",
    "cdd",
    "puell_multiple",
    "supply_in_profit",
]

DIVERGENCE_THRESHOLD = 0.10  # Flag if sources diverge >10% from reference


@dataclass
class SourceScore:
    """Score for a source on a specific metric."""

    source: str
    metric: str
    completeness: float  # % non-null values (0-1)
    freshness_days: float  # Days since last data point
    reference_mape: Optional[float]  # MAPE vs Blockchain.com reference
    selected: bool = False


class SourceSelector:
    """Select best data source per on-chain metric.

    For overlapping BTC metrics: scores each source and selects best.
    For BGeometrics-exclusive metrics: uses BGeometrics (no alternative).
    For ETH metrics: uses CoinMetrics (only free source with ETH).

    Usage:
        selector = SourceSelector()
        result = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=cm_df,
            blockchain_com_df=bc_df,
        )
    """

    def select_btc_onchain(
        self,
        bgeometrics_df: Optional[pd.DataFrame] = None,
        coinmetrics_df: Optional[pd.DataFrame] = None,
        blockchain_com_df: Optional[pd.DataFrame] = None,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> tuple[pd.DataFrame, list[SourceScore]]:
        """Select best source per BTC on-chain metric.

        Args:
            bgeometrics_df: DataFrame from BGeometricsFetcher.
            coinmetrics_df: DataFrame from CoinMetricsFetcher.
            blockchain_com_df: DataFrame from BlockchainComFetcher (reference).
            reference_date: Date for freshness calculation (default: now).

        Returns:
            Tuple of (unified DataFrame, list of SourceScore for provenance).
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now(tz="UTC")

        scores = []
        result_columns = {}

        # 1. Handle overlapping metrics — cross-validate and select best
        for metric, source_map in OVERLAPPING_METRICS.items():
            candidates = {}
            if bgeometrics_df is not None and source_map["bgeometrics"] in bgeometrics_df.columns:
                candidates["bgeometrics"] = bgeometrics_df[source_map["bgeometrics"]]
            if coinmetrics_df is not None and source_map["coinmetrics"] in coinmetrics_df.columns:
                candidates["coinmetrics"] = coinmetrics_df[source_map["coinmetrics"]]

            reference = None
            if blockchain_com_df is not None and source_map["blockchain_com"] in blockchain_com_df.columns:
                reference = blockchain_com_df[source_map["blockchain_com"]]

            if not candidates:
                logger.warning(f"[DATA] No source available for {metric}")
                continue

            best_source = None
            best_score_val = -1.0

            for source_name, series in candidates.items():
                score = self._score_series(
                    series, source_name, metric, reference, reference_date
                )
                scores.append(score)

                # Composite score: completeness * 0.4 + (1 - freshness_normalized) * 0.3 + agreement * 0.3
                freshness_norm = min(score.freshness_days / 30.0, 1.0)
                # NaN MAPE → neutral 0.5 agreement (not perfect 1.0)
                if score.reference_mape is not None:
                    agreement = 1.0 - min(score.reference_mape, 1.0)
                else:
                    agreement = 0.5
                composite = (
                    score.completeness * 0.4
                    + (1.0 - freshness_norm) * 0.3
                    + agreement * 0.3
                )

                if composite > best_score_val:
                    best_score_val = composite
                    best_source = source_name

                # Flag divergence
                if score.reference_mape is not None and score.reference_mape > DIVERGENCE_THRESHOLD:
                    logger.warning(
                        f"[DATA] {source_name}/{metric} diverges "
                        f"{score.reference_mape:.1%} from reference (>{DIVERGENCE_THRESHOLD:.0%})"
                    )

            # Mark the selected source
            for s in scores:
                if s.metric == metric and s.source == best_source:
                    s.selected = True
                    result_columns[metric] = candidates[best_source].rename(metric)
                    logger.info(f"[DATA] Selected {best_source} for {metric}")

        # 2. Handle BGeometrics-exclusive metrics
        if bgeometrics_df is not None:
            for metric in BGEOMETRICS_EXCLUSIVE:
                if metric in bgeometrics_df.columns:
                    result_columns[metric] = bgeometrics_df[metric]
                    score = self._score_series(
                        bgeometrics_df[metric],
                        "bgeometrics",
                        metric,
                        None,
                        reference_date,
                    )
                    score.selected = True
                    scores.append(score)

        # 3. Build unified DataFrame
        if not result_columns:
            return pd.DataFrame(), scores

        result = pd.DataFrame(result_columns)
        result = result.sort_index()

        logger.info(
            f"[DATA] SourceSelector: {len(result)} rows, "
            f"{len(result.columns)} metrics selected"
        )
        return result, scores

    def select_eth_onchain(
        self,
        coinmetrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Select ETH on-chain metrics (CoinMetrics only source).

        Args:
            coinmetrics_df: DataFrame from CoinMetricsFetcher for ETH.

        Returns:
            DataFrame with ETH on-chain metrics.
        """
        if coinmetrics_df is None or coinmetrics_df.empty:
            logger.warning("[DATA] No CoinMetrics ETH data available")
            return pd.DataFrame()

        logger.info(
            f"[DATA] ETH on-chain: {len(coinmetrics_df)} rows, "
            f"{len(coinmetrics_df.columns)} metrics from CoinMetrics"
        )
        return coinmetrics_df

    def _score_series(
        self,
        series: pd.Series,
        source: str,
        metric: str,
        reference: Optional[pd.Series],
        reference_date: pd.Timestamp,
    ) -> SourceScore:
        """Score a single metric series from a source."""
        # Completeness: % non-null
        completeness = 1.0 - (series.isna().sum() / max(len(series), 1))

        # Freshness: days since last non-null value
        valid = series.dropna()
        if len(valid) > 0 and hasattr(valid.index, "max"):
            last_date = valid.index.max()
            if hasattr(last_date, "tz") and last_date.tz is None:
                last_date = last_date.tz_localize("UTC")
            freshness_days = (reference_date - last_date).total_seconds() / 86400
        else:
            freshness_days = 999.0

        # Reference agreement: MAPE vs Blockchain.com
        reference_mape = None
        if reference is not None:
            aligned = pd.DataFrame({"source": series, "reference": reference}).dropna()
            if len(aligned) > 0:
                mape = np.mean(
                    np.abs(aligned["source"] - aligned["reference"])
                    / np.abs(aligned["reference"].replace(0, np.nan))
                )
                reference_mape = float(mape) if not np.isnan(mape) else None

        return SourceScore(
            source=source,
            metric=metric,
            completeness=completeness,
            freshness_days=max(freshness_days, 0.0),
            reference_mape=reference_mape,
        )
