#!/usr/bin/env python3
"""
Validate macro and on-chain data collected by Research Agent.

Checks:
1. Row count and date range coverage
2. Column names and data types
3. Missing values / NaN counts per column
4. Basic statistics (mean, min, max) for sanity checks
5. Data range sanity checks for macro data
6. On-chain metrics availability
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Expected value ranges for macro data (typical ranges)
MACRO_RANGES = {
    "dxy_daily": {
        "open": (80, 120),
        "high": (80, 120),
        "low": (80, 120),
        "close": (80, 120),
        "volume": (0, 1e12),
    },
    "gold_daily": {
        "open": (1000, 2500),
        "high": (1000, 2500),
        "low": (1000, 2500),
        "close": (1000, 2500),
        "volume": (0, 1e12),
    },
    "spx_daily": {
        "open": (1000, 6000),
        "high": (1000, 6000),
        "low": (1000, 6000),
        "close": (1000, 6000),
        "volume": (0, 1e12),
    },
    "vix_daily": {
        "open": (10, 80),
        "high": (10, 80),
        "low": (10, 80),
        "close": (10, 80),
        "volume": (0, 1e12),
    },
}


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n{title}")
    print("-" * 80)


def validate_parquet_file(file_path: Path, file_type: str, expected_ranges: Dict = None) -> Dict:
    """
    Validate a parquet file and return validation results.

    Args:
        file_path: Path to the parquet file
        file_type: Type of file (macro or onchain)
        expected_ranges: Optional dict of expected value ranges for columns

    Returns:
        Dict with validation results
    """
    print_subheader(f"Validating: {file_path.name}")

    if not file_path.exists():
        print(f"[ERROR] File does not exist: {file_path}")
        return {"status": "error", "message": "File does not exist"}

    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)

        results = {
            "status": "success",
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "index_type": str(type(df.index).__name__),
            "issues": [],
        }

        # 1. Basic info
        print(f"\n[INFO] File size: {results['file_size_mb']:.2f} MB")
        print(f"[INFO] Shape: {df.shape} (rows x columns)")
        print(f"[INFO] Index type: {results['index_type']}")

        # 2. Date range coverage (if DatetimeIndex)
        if isinstance(df.index, pd.DatetimeIndex):
            date_min = df.index.min()
            date_max = df.index.max()
            date_range_days = (date_max - date_min).days
            results["date_min"] = str(date_min.date())
            results["date_max"] = str(date_max.date())
            results["date_range_days"] = date_range_days

            print(f"[INFO] Date range: {date_min.date()} to {date_max.date()}")
            print(f"[INFO] Date range span: {date_range_days} days")

            # Check for timezone
            if df.index.tz is None:
                results["issues"].append("DatetimeIndex has no timezone")
                print("[WARNING] DatetimeIndex has no timezone")
            else:
                print(f"[INFO] Timezone: {df.index.tz}")
                results["timezone"] = str(df.index.tz)
        else:
            print(f"[WARNING] Index is not DatetimeIndex: {type(df.index)}")
            results["issues"].append(f"Index is not DatetimeIndex: {type(df.index)}")

        # 3. Columns and data types
        print(f"\n[INFO] Columns and data types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")

        # 4. Missing values
        print(f"\n[INFO] Missing values (NaN counts):")
        nan_counts = df.isna().sum()
        results["nan_counts"] = {}
        for col in df.columns:
            nan_count = nan_counts[col]
            nan_pct = 100 * nan_count / len(df)
            results["nan_counts"][col] = {
                "count": int(nan_count),
                "percentage": round(nan_pct, 2)
            }

            status = "OK" if nan_count == 0 else "WARNING" if nan_pct < 5 else "ERROR"
            print(f"  [{status}] {col}: {nan_count} ({nan_pct:.2f}%)")

            if nan_count > 0:
                msg = f"{col} has {nan_count} NaN values ({nan_pct:.2f}%)"
                results["issues"].append(msg)

        # 5. Basic statistics for numeric columns
        print(f"\n[INFO] Basic statistics (numeric columns):")
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            results["statistics"] = {}

            for col in numeric_cols:
                col_stats = {
                    "count": int(stats.loc["count", col]),
                    "mean": float(stats.loc["mean", col]),
                    "std": float(stats.loc["std", col]),
                    "min": float(stats.loc["min", col]),
                    "25%": float(stats.loc["25%", col]),
                    "50%": float(stats.loc["50%", col]),
                    "75%": float(stats.loc["75%", col]),
                    "max": float(stats.loc["max", col]),
                }
                results["statistics"][col] = col_stats

                print(f"\n  {col}:")
                print(f"    count: {col_stats['count']}")
                print(f"    mean:  {col_stats['mean']:.2f}")
                print(f"    std:   {col_stats['std']:.2f}")
                print(f"    min:   {col_stats['min']:.2f}")
                print(f"    25%:   {col_stats['25%']:.2f}")
                print(f"    50%:   {col_stats['50%']:.2f}")
                print(f"    75%:   {col_stats['75%']:.2f}")
                print(f"    max:   {col_stats['max']:.2f}")

                # 6. Range sanity checks (if expected ranges provided)
                if expected_ranges and col in expected_ranges:
                    min_val, max_val = expected_ranges[col]
                    actual_min = col_stats["min"]
                    actual_max = col_stats["max"]

                    if actual_min < min_val or actual_max > max_val:
                        msg = f"{col} values outside expected range [{min_val}, {max_val}]: actual [{actual_min:.2f}, {actual_max:.2f}]"
                        results["issues"].append(msg)
                        print(f"    [WARNING] Values outside expected range [{min_val}, {max_val}]")
                    else:
                        print(f"    [OK] Values within expected range [{min_val}, {max_val}]")

        # 7. Check for duplicates in index
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            msg = f"Index has {dup_count} duplicate entries"
            results["issues"].append(msg)
            print(f"\n[WARNING] {msg}")
        else:
            print(f"\n[OK] No duplicate index entries")

        # 8. Summary
        print(f"\n[SUMMARY]")
        if len(results["issues"]) == 0:
            print(f"  [OK] No issues found")
        else:
            print(f"  [WARNING] {len(results['issues'])} issue(s) found:")
            for i, issue in enumerate(results["issues"], 1):
                print(f"    {i}. {issue}")

        return results

    except Exception as e:
        print(f"[ERROR] Failed to validate file: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def validate_macro_data() -> List[Dict]:
    """Validate all macro data files."""
    print_header("MACRO DATA VALIDATION")

    macro_dir = Path("/home/akamath/sparky-ai/data/raw/macro")

    macro_files = [
        ("dxy_daily.parquet", "dxy_daily"),
        ("gold_daily.parquet", "gold_daily"),
        ("spx_daily.parquet", "spx_daily"),
        ("vix_daily.parquet", "vix_daily"),
    ]

    results = []
    for filename, file_key in macro_files:
        file_path = macro_dir / filename
        expected_ranges = MACRO_RANGES.get(file_key)
        result = validate_parquet_file(file_path, "macro", expected_ranges)
        results.append(result)

    return results


def validate_onchain_data() -> List[Dict]:
    """Validate all on-chain data files."""
    print_header("ON-CHAIN DATA VALIDATION")

    onchain_dir = Path("/home/akamath/sparky-ai/data/raw/onchain")

    onchain_files = [
        "blockchain_com_btc_daily.parquet",
        "coinmetrics_btc_daily.parquet",
    ]

    results = []
    for filename in onchain_files:
        file_path = onchain_dir / filename
        result = validate_parquet_file(file_path, "onchain")
        results.append(result)

    # Check for specific on-chain metrics
    print_subheader("On-Chain Metrics Availability Check")

    expected_metrics = {
        "coinmetrics_btc_daily.parquet": [
            "CapMVRVCur",  # MVRV ratio
            "HashRate",
            "AdrActCnt",   # Active addresses
            "TxCnt",       # Transaction count
            "PriceUSD",
        ],
        "blockchain_com_btc_daily.parquet": [
            "hash_rate",
            "unique_addresses",
            "n_transactions",
        ],
    }

    for filename, metrics in expected_metrics.items():
        file_path = onchain_dir / filename
        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"\n{filename}:")
            for metric in metrics:
                if metric in df.columns:
                    print(f"  [OK] {metric} - available")
                else:
                    print(f"  [MISSING] {metric} - NOT FOUND")

    return results


def print_final_summary(macro_results: List[Dict], onchain_results: List[Dict]) -> None:
    """Print final validation summary."""
    print_header("FINAL VALIDATION SUMMARY")

    total_files = len(macro_results) + len(onchain_results)
    successful = sum(1 for r in macro_results + onchain_results if r.get("status") == "success")
    errors = sum(1 for r in macro_results + onchain_results if r.get("status") == "error")

    print(f"\n[INFO] Total files validated: {total_files}")
    print(f"[INFO] Successful: {successful}")
    print(f"[INFO] Errors: {errors}")

    # Count total issues
    total_issues = sum(len(r.get("issues", [])) for r in macro_results + onchain_results)
    print(f"[INFO] Total issues found: {total_issues}")

    if total_issues == 0 and errors == 0:
        print("\n" + "=" * 80)
        print("  ✓ ALL VALIDATIONS PASSED - DATA QUALITY LOOKS GOOD")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"  ⚠ VALIDATION COMPLETED WITH {total_issues} ISSUE(S)")
        print("=" * 80)

        # List all issues
        print("\nAll issues:")
        for result in macro_results + onchain_results:
            if result.get("issues"):
                file_name = Path(result.get("file_path", "unknown")).name
                print(f"\n  {file_name}:")
                for issue in result["issues"]:
                    print(f"    - {issue}")


def main():
    """Main validation function."""
    print("=" * 80)
    print("  SPARKY AI - DATA VALIDATION AGENT")
    print("  Validating Research Agent's Macro and On-Chain Data Collection")
    print("=" * 80)

    try:
        # Validate macro data
        macro_results = validate_macro_data()

        # Validate on-chain data
        onchain_results = validate_onchain_data()

        # Print final summary
        print_final_summary(macro_results, onchain_results)

        # Exit with appropriate code
        total_issues = sum(len(r.get("issues", [])) for r in macro_results + onchain_results)
        errors = sum(1 for r in macro_results + onchain_results if r.get("status") == "error")

        if errors > 0:
            sys.exit(1)
        elif total_issues > 0:
            sys.exit(2)  # Warnings but no errors
        else:
            sys.exit(0)  # All good

    except Exception as e:
        print(f"\n[FATAL ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
