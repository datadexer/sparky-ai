#!/usr/bin/env python3
"""Comprehensive Validation of Cross-Asset Hourly Data

Validates all 7 crypto asset parquet files for:
- Row counts
- Date range coverage
- Column integrity
- Missing values
- Time series gaps
- OHLC sanity checks
- Price validity
- File size analysis
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Files to validate
PARQUET_FILES = [
    "/home/akamath/sparky-ai/data/raw/eth/ohlcv_hourly.parquet",
    "/home/akamath/sparky-ai/data/raw/sol/ohlcv_hourly.parquet",
    "/home/akamath/sparky-ai/data/raw/avax/ohlcv_hourly.parquet",
    "/home/akamath/sparky-ai/data/raw/dot/ohlcv_hourly.parquet",
    "/home/akamath/sparky-ai/data/raw/link/ohlcv_hourly.parquet",
    "/home/akamath/sparky-ai/data/raw/ada/ohlcv_hourly.parquet",
    "/home/akamath/sparky-ai/data/raw/matic/ohlcv_hourly.parquet",
]

EXPECTED_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_file(file_path: str) -> dict:
    """Validate a single parquet file.

    Returns:
        dict with validation results
    """
    asset_name = Path(file_path).parent.name
    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATING: {asset_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"File: {file_path}")

    results = {
        "asset": asset_name,
        "file_path": file_path,
        "file_exists": False,
        "file_size_bytes": 0,
        "row_count": 0,
        "column_names": [],
        "start_date": None,
        "end_date": None,
        "date_range_days": 0,
        "missing_values": {},
        "gaps_over_2h": 0,
        "max_gap_hours": 0,
        "ohlc_violations": 0,
        "zero_or_negative_prices": 0,
        "errors": [],
        "warnings": [],
    }

    # Check file exists
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        error = f"FILE NOT FOUND: {file_path}"
        logger.error(error)
        results["errors"].append(error)
        return results

    results["file_exists"] = True
    results["file_size_bytes"] = file_path_obj.stat().st_size
    logger.info(f"File size: {results['file_size_bytes']:,} bytes ({results['file_size_bytes'] / 1024:.1f} KB)")

    # Flag suspiciously small files
    if results["file_size_bytes"] < 100_000:  # < 100 KB
        warning = f"SUSPICIOUSLY SMALL FILE: {results['file_size_bytes'] / 1024:.1f} KB (expected > 100 KB)"
        logger.warning(warning)
        results["warnings"].append(warning)

    # Read parquet file
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded parquet file")
    except Exception as e:
        error = f"Failed to read parquet: {e}"
        logger.error(error)
        results["errors"].append(error)
        return results

    # 1. Row count
    results["row_count"] = len(df)
    logger.info(f"Row count: {results['row_count']:,}")

    if results["row_count"] < 1000:
        warning = f"LOW ROW COUNT: {results['row_count']} (expected thousands for 2017-2025 hourly data)"
        logger.warning(warning)
        results["warnings"].append(warning)

    # 2. Column names
    results["column_names"] = df.columns.tolist()
    logger.info(f"Columns: {results['column_names']}")

    # Check for expected columns
    missing_cols = set(EXPECTED_COLUMNS) - set(results["column_names"])
    if missing_cols:
        error = f"MISSING COLUMNS: {missing_cols}"
        logger.error(error)
        results["errors"].append(error)

    # 3. Date range coverage
    if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
        results["start_date"] = df.index.min()
        results["end_date"] = df.index.max()
        results["date_range_days"] = (results["end_date"] - results["start_date"]).days

        logger.info(f"Start date: {results['start_date']}")
        logger.info(f"End date: {results['end_date']}")
        logger.info(f"Date range: {results['date_range_days']} days")

        # Check if date range is reasonable (2017-2025 should be ~8 years = 2920 days)
        expected_days = 365 * 8  # ~2920 days
        if results["date_range_days"] < 365:
            warning = f"SHORT DATE RANGE: {results['date_range_days']} days (expected ~{expected_days} days for 2017-2025)"
            logger.warning(warning)
            results["warnings"].append(warning)
    else:
        error = "No datetime index found"
        logger.error(error)
        results["errors"].append(error)

    # 4. Missing values
    missing_counts = df.isnull().sum()
    results["missing_values"] = missing_counts.to_dict()

    logger.info("Missing values per column:")
    for col, count in results["missing_values"].items():
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        logger.info(f"  {col}: {count} ({pct:.2f}%)")

        if count > 0:
            warning = f"MISSING VALUES in {col}: {count} ({pct:.2f}%)"
            results["warnings"].append(warning)

    # 5. Time series gaps (only if we have a datetime index)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        time_diffs = df.index.to_series().diff()
        gaps_over_2h = time_diffs > pd.Timedelta(hours=2)
        results["gaps_over_2h"] = gaps_over_2h.sum()

        if results["gaps_over_2h"] > 0:
            max_gap = time_diffs.max()
            results["max_gap_hours"] = max_gap.total_seconds() / 3600
            warning = f"GAPS > 2 hours: {results['gaps_over_2h']} gaps found (max gap: {results['max_gap_hours']:.1f} hours)"
            logger.warning(warning)
            results["warnings"].append(warning)

            # Show first few gaps
            gap_indices = time_diffs[gaps_over_2h].head(5)
            logger.info("First 5 gaps:")
            for idx, gap in gap_indices.items():
                logger.info(f"  {idx}: {gap.total_seconds() / 3600:.1f} hours")
        else:
            logger.info("No gaps > 2 hours detected")

    # 6. OHLC sanity checks
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        violations = 0

        # high >= low
        high_low_violations = (df["high"] < df["low"]).sum()
        violations += high_low_violations
        if high_low_violations > 0:
            error = f"OHLC VIOLATION: high < low in {high_low_violations} rows"
            logger.error(error)
            results["errors"].append(error)

        # high >= open
        high_open_violations = (df["high"] < df["open"]).sum()
        violations += high_open_violations
        if high_open_violations > 0:
            error = f"OHLC VIOLATION: high < open in {high_open_violations} rows"
            logger.error(error)
            results["errors"].append(error)

        # high >= close
        high_close_violations = (df["high"] < df["close"]).sum()
        violations += high_close_violations
        if high_close_violations > 0:
            error = f"OHLC VIOLATION: high < close in {high_close_violations} rows"
            logger.error(error)
            results["errors"].append(error)

        # low <= open
        low_open_violations = (df["low"] > df["open"]).sum()
        violations += low_open_violations
        if low_open_violations > 0:
            error = f"OHLC VIOLATION: low > open in {low_open_violations} rows"
            logger.error(error)
            results["errors"].append(error)

        # low <= close
        low_close_violations = (df["low"] > df["close"]).sum()
        violations += low_close_violations
        if low_close_violations > 0:
            error = f"OHLC VIOLATION: low > close in {low_close_violations} rows"
            logger.error(error)
            results["errors"].append(error)

        results["ohlc_violations"] = violations

        if violations == 0:
            logger.info("OHLC sanity checks: PASSED (all constraints satisfied)")
        else:
            logger.error(f"OHLC sanity checks: FAILED ({violations} total violations)")

    # 7. Zero or negative prices
    price_cols = ["open", "high", "low", "close"]
    zero_neg_count = 0

    for col in price_cols:
        if col in df.columns:
            invalid = (df[col] <= 0).sum()
            zero_neg_count += invalid

            if invalid > 0:
                error = f"INVALID PRICES in {col}: {invalid} zero or negative values"
                logger.error(error)
                results["errors"].append(error)

    results["zero_or_negative_prices"] = zero_neg_count

    if zero_neg_count == 0:
        logger.info("Price validation: PASSED (all prices > 0)")
    else:
        logger.error(f"Price validation: FAILED ({zero_neg_count} zero/negative prices)")

    # Summary
    logger.info(f"\n{'='*40}")
    logger.info(f"VALIDATION SUMMARY: {asset_name.upper()}")
    logger.info(f"{'='*40}")
    logger.info(f"Errors: {len(results['errors'])}")
    logger.info(f"Warnings: {len(results['warnings'])}")

    if len(results['errors']) == 0 and len(results['warnings']) == 0:
        logger.info("STATUS: ✓ PASSED")
    elif len(results['errors']) == 0:
        logger.info("STATUS: ⚠ PASSED WITH WARNINGS")
    else:
        logger.info("STATUS: ✗ FAILED")

    return results


def main():
    """Main validation execution."""

    logger.info("=" * 80)
    logger.info("CROSS-ASSET HOURLY DATA VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Validating {len(PARQUET_FILES)} parquet files")
    logger.info("=" * 80)

    all_results = []

    for file_path in PARQUET_FILES:
        results = validate_file(file_path)
        all_results.append(results)

    # Generate comprehensive report
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE VALIDATION REPORT")
    logger.info("=" * 80)

    # Summary table
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Asset": r["asset"].upper(),
            "File Size (KB)": f"{r['file_size_bytes'] / 1024:.1f}",
            "Rows": f"{r['row_count']:,}",
            "Start Date": str(r['start_date'].date()) if r['start_date'] else "N/A",
            "End Date": str(r['end_date'].date()) if r['end_date'] else "N/A",
            "Days": r['date_range_days'],
            "Errors": len(r['errors']),
            "Warnings": len(r['warnings']),
        })

    summary_df = pd.DataFrame(summary_data)
    logger.info("\n" + summary_df.to_string(index=False))

    # Critical issues
    logger.info("\n" + "=" * 80)
    logger.info("CRITICAL ISSUES")
    logger.info("=" * 80)

    critical_issues = []

    for r in all_results:
        if not r["file_exists"]:
            critical_issues.append(f"{r['asset'].upper()}: FILE NOT FOUND")

        if r["row_count"] < 1000:
            critical_issues.append(f"{r['asset'].upper()}: Only {r['row_count']} rows (expected thousands)")

        if r["file_size_bytes"] < 100_000:
            critical_issues.append(f"{r['asset'].upper()}: File size only {r['file_size_bytes'] / 1024:.1f} KB")

        if r["date_range_days"] < 365:
            critical_issues.append(f"{r['asset'].upper()}: Date range only {r['date_range_days']} days")

        if r["ohlc_violations"] > 0:
            critical_issues.append(f"{r['asset'].upper()}: {r['ohlc_violations']} OHLC violations")

        if r["zero_or_negative_prices"] > 0:
            critical_issues.append(f"{r['asset'].upper()}: {r['zero_or_negative_prices']} invalid prices")

    if critical_issues:
        logger.error(f"Found {len(critical_issues)} critical issues:")
        for issue in critical_issues:
            logger.error(f"  - {issue}")
    else:
        logger.info("No critical issues found")

    # File size comparison
    logger.info("\n" + "=" * 80)
    logger.info("FILE SIZE ANALYSIS")
    logger.info("=" * 80)

    sizes = [(r["asset"], r["file_size_bytes"]) for r in all_results if r["file_exists"]]
    sizes.sort(key=lambda x: x[1], reverse=True)

    logger.info("Files ranked by size:")
    for asset, size in sizes:
        logger.info(f"  {asset.upper()}: {size:,} bytes ({size / 1024:.1f} KB)")

    if sizes:
        largest = sizes[0]
        logger.info(f"\nLargest file: {largest[0].upper()} ({largest[1] / 1024:.1f} KB)")

        # Compare others to largest
        logger.info("\nSize comparison to largest file:")
        for asset, size in sizes[1:]:
            ratio = size / largest[1]
            logger.info(f"  {asset.upper()}: {ratio:.1%} of {largest[0].upper()}")

            if ratio < 0.05:  # < 5% of largest
                logger.warning(f"    ⚠ WARNING: {asset.upper()} is suspiciously small ({ratio:.1%} of {largest[0].upper()})")

    # Overall status
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL VALIDATION STATUS")
    logger.info("=" * 80)

    total_errors = sum(len(r["errors"]) for r in all_results)
    total_warnings = sum(len(r["warnings"]) for r in all_results)

    logger.info(f"Files validated: {len(all_results)}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total warnings: {total_warnings}")

    if total_errors == 0 and total_warnings == 0:
        logger.info("\n✓ ALL VALIDATIONS PASSED")
    elif total_errors == 0:
        logger.info(f"\n⚠ PASSED WITH {total_warnings} WARNINGS")
    else:
        logger.error(f"\n✗ VALIDATION FAILED ({total_errors} errors, {total_warnings} warnings)")

    # Save detailed results
    output_path = Path("/home/akamath/sparky-ai/data/validation_report.csv")
    summary_df.to_csv(output_path, index=False)
    logger.info(f"\nDetailed report saved to: {output_path}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
