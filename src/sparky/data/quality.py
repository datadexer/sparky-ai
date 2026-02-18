"""Data quality checks for market and on-chain data.

Validates completeness, range, staleness, and cross-source agreement.
Crypto trades 24/7 — gaps are real problems, not weekends.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUALITY_REPORTS_DIR = Path("data/quality_reports")


class DataQualityChecker:
    """Run data quality checks and generate reports.

    Usage:
        checker = DataQualityChecker()
        result = checker.check_completeness(df, max_gap_days=3)
        report = checker.run_all_checks(df, asset="btc", source="binance")
    """

    def check_completeness(self, df: pd.DataFrame, max_gap_days: int = 3) -> dict[str, Any]:
        """Check for missing data and gaps.

        Crypto trades 24/7, so any gap > max_gap_days is a problem.

        Args:
            df: DataFrame with DatetimeIndex.
            max_gap_days: Maximum acceptable gap in days.

        Returns:
            Dict with: total_rows, null_counts, null_pct, gaps (list of date ranges),
            pass (bool).
        """
        result = {
            "check": "completeness",
            "total_rows": len(df),
            "null_counts": {},
            "null_pct": {},
            "gaps": [],
            "pass": True,
        }

        if df.empty:
            result["pass"] = False
            return result

        # Null counts per column
        for col in df.columns:
            nulls = int(df[col].isna().sum())
            result["null_counts"][col] = nulls
            result["null_pct"][col] = round(nulls / len(df) * 100, 2)

        # Check for date gaps
        if isinstance(df.index, pd.DatetimeIndex):
            sorted_idx = df.index.sort_values()
            diffs = sorted_idx.to_series().diff()
            large_gaps = diffs[diffs > pd.Timedelta(days=max_gap_days)]

            for gap_end, gap_size in large_gaps.items():
                result["gaps"].append(
                    {
                        "gap_start": str(gap_end - gap_size),
                        "gap_end": str(gap_end),
                        "gap_days": gap_size.days,
                    }
                )

        # Fail if >5% null in any column or any large gaps
        max_null_pct = max(result["null_pct"].values()) if result["null_pct"] else 0
        if max_null_pct > 5.0 or len(result["gaps"]) > 0:
            result["pass"] = False

        return result

    def check_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> dict[str, Any]:
        """Check if values fall within expected range.

        Args:
            df: DataFrame to check.
            column: Column name.
            min_val: Minimum acceptable value (inclusive).
            max_val: Maximum acceptable value (inclusive).

        Returns:
            Dict with: column, min_val, max_val, actual_min, actual_max,
            out_of_range_count, pass.
        """
        result = {
            "check": "range",
            "column": column,
            "min_val": min_val,
            "max_val": max_val,
            "pass": True,
        }

        if column not in df.columns:
            result["pass"] = False
            result["error"] = f"Column '{column}' not found"
            return result

        series = df[column].dropna()
        result["actual_min"] = float(series.min()) if len(series) > 0 else None
        result["actual_max"] = float(series.max()) if len(series) > 0 else None

        out_of_range = 0
        if min_val is not None:
            out_of_range += int((series < min_val).sum())
        if max_val is not None:
            out_of_range += int((series > max_val).sum())

        result["out_of_range_count"] = out_of_range
        if out_of_range > 0:
            result["pass"] = False

        return result

    def check_staleness(
        self,
        df: pd.DataFrame,
        max_stale_days: int = 2,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, Any]:
        """Check if data is too old.

        Args:
            df: DataFrame with DatetimeIndex.
            max_stale_days: Maximum acceptable age in days.
            reference_date: Date to compare against (default: now UTC).
                Useful for deterministic tests.

        Returns:
            Dict with: last_date, days_old, max_stale_days, pass.
        """
        result = {
            "check": "staleness",
            "max_stale_days": max_stale_days,
            "pass": True,
        }

        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            result["pass"] = False
            result["last_date"] = None
            result["days_old"] = None
            return result

        last_date = df.index.max()
        now = reference_date if reference_date is not None else pd.Timestamp.now(tz="UTC")

        if last_date.tz is None:
            last_date = last_date.tz_localize("UTC")

        days_old = (now - last_date).total_seconds() / 86400
        result["last_date"] = str(last_date)
        result["days_old"] = round(days_old, 1)
        result["pass"] = days_old <= max_stale_days

        return result

    def cross_validate_price(
        self,
        ccxt_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        price_col: str = "close",
        reference_col: str = "PriceUSD",
        max_pct_diff: float = 0.02,
    ) -> dict[str, Any]:
        """Cross-validate prices between CCXT and another source.

        Args:
            ccxt_df: OHLCV DataFrame from CCXT.
            reference_df: Reference price DataFrame (e.g., CoinMetrics).
            price_col: Column in ccxt_df to compare.
            reference_col: Column in reference_df to compare.
            max_pct_diff: Maximum acceptable mean percentage difference.

        Returns:
            Dict with: mean_pct_diff, max_pct_diff_observed, dates_compared, pass.
        """
        result = {
            "check": "cross_validate_price",
            "max_pct_diff_threshold": max_pct_diff,
            "pass": True,
        }

        if price_col not in ccxt_df.columns or reference_col not in reference_df.columns:
            result["pass"] = False
            result["error"] = "Required columns not found"
            return result

        aligned = pd.DataFrame(
            {
                "ccxt": ccxt_df[price_col],
                "reference": reference_df[reference_col],
            }
        ).dropna()

        if aligned.empty:
            result["pass"] = False
            result["dates_compared"] = 0
            return result

        pct_diff = np.abs(aligned["ccxt"] - aligned["reference"]) / aligned["reference"]
        result["dates_compared"] = len(aligned)
        result["mean_pct_diff"] = round(float(pct_diff.mean()), 6)
        result["max_pct_diff_observed"] = round(float(pct_diff.max()), 6)
        result["pass"] = result["mean_pct_diff"] <= max_pct_diff

        return result

    def run_all_checks(
        self,
        df: pd.DataFrame,
        asset: str,
        source: str,
        price_range: Optional[tuple[float, float]] = None,
    ) -> dict[str, Any]:
        """Run all applicable quality checks on a DataFrame.

        Args:
            df: DataFrame to check.
            asset: Asset name (e.g., "btc", "eth").
            source: Data source name.
            price_range: Optional (min, max) price range for validation.

        Returns:
            Full quality report dict.
        """
        report = {
            "asset": asset,
            "source": source,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "overall_pass": True,
        }

        # Completeness
        completeness = self.check_completeness(df)
        report["checks"]["completeness"] = completeness
        if not completeness["pass"]:
            report["overall_pass"] = False

        # Staleness
        staleness = self.check_staleness(df)
        report["checks"]["staleness"] = staleness
        if not staleness["pass"]:
            report["overall_pass"] = False

        # Price range (if applicable)
        if price_range and "close" in df.columns:
            range_check = self.check_range(df, "close", *price_range)
            report["checks"]["price_range"] = range_check
            if not range_check["pass"]:
                report["overall_pass"] = False

        return report

    def save_report(self, report: dict, filename: str) -> Path:
        """Save quality report to JSON.

        Args:
            report: Quality report dict.
            filename: Report filename (without path).

        Returns:
            Path to saved report.
        """
        QUALITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = QUALITY_REPORTS_DIR / filename

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"[DATA] Quality report saved to {path}")
        return path
