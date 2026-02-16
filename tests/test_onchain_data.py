"""
Tests for on-chain metrics data loading and validation.

These tests require local data files and are skipped in CI.
"""

from pathlib import Path

import pandas as pd
import pytest

_onchain_dir = Path(__file__).parent.parent / "data" / "raw" / "onchain"
_has_onchain_data = (_onchain_dir / "coinmetrics_btc_daily.parquet").exists()

pytestmark = pytest.mark.skipif(
    not _has_onchain_data,
    reason="On-chain data files not available (requires local data files)"
)


@pytest.fixture
def onchain_dir():
    """Path to on-chain data directory."""
    return _onchain_dir


def test_coinmetrics_file_exists(onchain_dir):
    """Test that CoinMetrics parquet file exists."""
    file_path = onchain_dir / "coinmetrics_btc_daily.parquet"
    assert file_path.exists(), f"CoinMetrics file not found at {file_path}"


def test_blockchain_com_file_exists(onchain_dir):
    """Test that Blockchain.com parquet file exists."""
    file_path = onchain_dir / "blockchain_com_btc_daily.parquet"
    assert file_path.exists(), f"Blockchain.com file not found at {file_path}"


def test_coinmetrics_data_structure(onchain_dir):
    """Test CoinMetrics data structure and content."""
    file_path = onchain_dir / "coinmetrics_btc_daily.parquet"
    df = pd.read_parquet(file_path)

    # Check index is DatetimeIndex with UTC timezone
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None, "Index should have timezone info"
    assert str(df.index.tz) == "UTC", "Index should be in UTC timezone"

    # Check expected columns exist
    expected_cols = {
        "AdrActCnt",
        "AdrBalCnt",
        "BlkCnt",
        "CapMVRVCur",
        "CapMrktCurUSD",
        "FeeTotNtv",
        "FlowInExNtv",
        "FlowOutExNtv",
        "HashRate",
        "PriceUSD",
        "SplyCur",
        "TxCnt",
        "TxTfrCnt",
    }
    assert set(df.columns) == expected_cols

    # Check data types (all should be numeric)
    assert all(df[col].dtype in [float, int] for col in df.columns)

    # Check no NaN values
    assert df.isna().sum().sum() == 0, "CoinMetrics data should have no NaN values"

    # Check date range (should start from 2017-01-01)
    assert df.index.min() >= pd.Timestamp("2017-01-01", tz="UTC")
    assert df.index.max() >= pd.Timestamp("2026-01-01", tz="UTC")

    # Check data is sorted by index
    assert df.index.is_monotonic_increasing

    # Sanity checks on values
    assert (df["PriceUSD"] > 0).all(), "Price should be positive"
    assert (df["SplyCur"] > 0).all(), "Supply should be positive"
    assert (df["HashRate"] > 0).all(), "HashRate should be positive"


def test_blockchain_com_data_structure(onchain_dir):
    """Test Blockchain.com data structure and content."""
    file_path = onchain_dir / "blockchain_com_btc_daily.parquet"
    df = pd.read_parquet(file_path)

    # Check index is DatetimeIndex with UTC timezone
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None, "Index should have timezone info"
    assert str(df.index.tz) == "UTC", "Index should be in UTC timezone"

    # Check expected columns exist
    expected_cols = {"hash_rate", "unique_addresses", "n_transactions"}
    assert set(df.columns) == expected_cols

    # Check data types (all should be numeric)
    assert all(df[col].dtype in [float, int] for col in df.columns)

    # Check minimal NaN values (should be < 1%)
    nan_pct = 100 * df.isna().sum().sum() / (len(df) * len(df.columns))
    assert nan_pct < 1.0, f"Too many NaN values: {nan_pct:.2f}%"

    # Check date range
    assert df.index.min() >= pd.Timestamp("2017-01-01", tz="UTC")
    assert df.index.max() >= pd.Timestamp("2026-01-01", tz="UTC")

    # Check data is sorted by index
    assert df.index.is_monotonic_increasing

    # Sanity checks on values (ignoring NaN)
    assert (df["hash_rate"].dropna() > 0).all(), "HashRate should be positive"
    assert (df["n_transactions"].dropna() > 0).all(), "Transactions should be positive"


def test_data_cross_validation(onchain_dir):
    """
    Test that overlapping metrics from different sources are reasonably correlated.
    """
    df_cm = pd.read_parquet(onchain_dir / "coinmetrics_btc_daily.parquet")
    df_bc = pd.read_parquet(onchain_dir / "blockchain_com_btc_daily.parquet")

    # Merge on date
    df = df_cm.join(df_bc, how="inner", rsuffix="_bc")

    # Both sources have hash_rate and transaction count
    # They should be highly correlated (> 0.9)
    corr_hash = df["HashRate"].corr(df["hash_rate"])
    assert corr_hash > 0.9, f"HashRate correlation too low: {corr_hash:.3f}"

    # Transaction counts should also be correlated
    # Note: TxCnt vs n_transactions may have different definitions
    corr_tx = df["TxCnt"].corr(df["n_transactions"])
    assert corr_tx > 0.7, f"Transaction count correlation too low: {corr_tx:.3f}"


def test_data_completeness(onchain_dir):
    """Test that we have sufficient data coverage."""
    df_cm = pd.read_parquet(onchain_dir / "coinmetrics_btc_daily.parquet")
    df_bc = pd.read_parquet(onchain_dir / "blockchain_com_btc_daily.parquet")

    # Should have at least 8 years of data (2017-2026)
    # Note: Blockchain.com starts on 2017-02-18, so it's ~8.98 years
    assert len(df_cm) >= 365 * 8, "CoinMetrics should have at least 8 years of data"
    assert len(df_bc) >= 365 * 8, "Blockchain.com should have at least 8 years of data"

    # CoinMetrics should have more data points (starts earlier)
    assert len(df_cm) >= len(df_bc)
