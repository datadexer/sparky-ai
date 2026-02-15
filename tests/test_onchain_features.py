"""Tests for onchain features using synthetic data with known expected outputs."""

import numpy as np
import pandas as pd
import pytest

from sparky.features.onchain import (
    # BTC features
    hash_ribbon,
    nvt_zscore,
    mvrv_signal,
    sopr_signal,
    address_momentum,
    volume_momentum,
    nupl_regime,
    puell_signal,
    supply_in_profit_extreme,
    # ETH features
    gas_fee_zscore,
    eth_address_momentum,
    eth_transfer_value_zscore,
    eth_nvt_signal,
    eth_btc_ratio_momentum,
    eth_btc_correlation_regime,
)


# =============================================================================
# BTC Feature Tests
# =============================================================================


def test_hash_ribbon_bullish():
    """Test hash ribbon returns 1 when short MA > long MA."""
    # Create data where short MA will be higher than long MA
    # First 60 values low, then spike up - short MA will respond faster
    data = [10.0] * 60 + [20.0] * 40
    hash_rate = pd.Series(data)

    result = hash_ribbon(hash_rate, short=30, long=60)

    # When NaN values are compared, the result is False (0.0), not NaN
    # First 59 values will be 0.0 (before long MA is available)
    assert (result.iloc[:59] == 0.0).all()

    # After the spike, short MA should exceed long MA
    # Check the last value where short MA is definitely higher
    assert result.iloc[-1] == 1.0


def test_hash_ribbon_bearish():
    """Test hash ribbon returns 0 when short MA <= long MA."""
    # Create declining hash rate - short MA will be lower
    data = list(range(100, 0, -1))
    hash_rate = pd.Series(data)

    result = hash_ribbon(hash_rate, short=30, long=60)

    # Last value should be 0 (bearish)
    assert result.iloc[-1] == 0.0


def test_hash_ribbon_nan_handling():
    """Test hash ribbon handles insufficient data."""
    hash_rate = pd.Series([10.0] * 50)  # Only 50 values, need 60

    result = hash_ribbon(hash_rate, short=30, long=60)

    # We need 60 for long MA, but only have 50 values
    # The comparison of NaN values returns 0.0, not NaN
    assert (result == 0.0).all()


def test_nvt_zscore_high():
    """Test NVT z-score calculation - high NVT should give positive z-score."""
    # Create 90 values around 100, then spike to 200
    data = [100.0] * 90 + [200.0] * 10
    nvt = pd.Series(data)

    result = nvt_zscore(nvt, window=90)

    # First 89 should be NaN
    assert result.iloc[:89].isna().all()

    # After spike, z-score should be strongly positive
    # z = (200 - 100) / std
    assert result.iloc[-1] > 2.0


def test_nvt_zscore_low():
    """Test NVT z-score calculation - low NVT should give negative z-score."""
    # Create stable values then drop
    data = [100.0] * 90 + [50.0] * 10
    nvt = pd.Series(data)

    result = nvt_zscore(nvt, window=90)

    # After drop, z-score should be negative
    assert result.iloc[-1] < -2.0


def test_nvt_zscore_zero_std():
    """Test NVT z-score handles zero std (constant values)."""
    nvt = pd.Series([100.0] * 100)

    result = nvt_zscore(nvt, window=90)

    # When std=0, division by zero should give NaN
    assert result.iloc[-1] is np.nan or pd.isna(result.iloc[-1])


def test_mvrv_signal_overheated():
    """Test MVRV signal returns -1 when z-score > 7."""
    mvrv_zscore = pd.Series([5.0, 8.0, 10.0, 6.0])

    result = mvrv_signal(mvrv_zscore)

    expected = pd.Series([0.0, -1.0, -1.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_mvrv_signal_undervalued():
    """Test MVRV signal returns 1 when z-score < 0."""
    mvrv_zscore = pd.Series([1.0, -1.0, -2.0, 0.5])

    result = mvrv_signal(mvrv_zscore)

    expected = pd.Series([0.0, 1.0, 1.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_mvrv_signal_neutral():
    """Test MVRV signal returns 0 for neutral range."""
    mvrv_zscore = pd.Series([0.0, 3.5, 7.0])

    result = mvrv_signal(mvrv_zscore)

    expected = pd.Series([0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_sopr_signal_capitulation():
    """Test SOPR signal returns 1 when SOPR < 1 (capitulation)."""
    sopr = pd.Series([1.05, 0.98, 0.95, 1.0, 1.02])

    result = sopr_signal(sopr)

    expected = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_sopr_signal_normal():
    """Test SOPR signal returns 0 when SOPR >= 1."""
    sopr = pd.Series([1.0, 1.1, 1.5])

    result = sopr_signal(sopr)

    expected = pd.Series([0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_address_momentum_positive():
    """Test address momentum calculates correct pct_change."""
    # Create rising addresses: 100, 110, 120, 130
    active_addresses = pd.Series([100.0, 110.0, 120.0, 130.0])

    result = address_momentum(active_addresses, period=1)

    # pct_change with period=1: [NaN, 0.10, 0.0909..., 0.0833...]
    assert pd.isna(result.iloc[0])
    assert np.isclose(result.iloc[1], 0.10)
    assert np.isclose(result.iloc[2], 120.0/110.0 - 1)


def test_address_momentum_longer_period():
    """Test address momentum with longer period."""
    # 30 days: 100 -> 150 = 50% increase
    active_addresses = pd.Series([100.0] * 30 + [150.0])

    result = address_momentum(active_addresses, period=30)

    # First 30 should be NaN
    assert result.iloc[:30].isna().all()
    # Last value: (150 - 100) / 100 = 0.5
    assert np.isclose(result.iloc[-1], 0.5)


def test_volume_momentum_positive():
    """Test volume momentum calculates correct pct_change."""
    tx_volume = pd.Series([1000.0, 1100.0, 1210.0])

    result = volume_momentum(tx_volume, period=1)

    assert pd.isna(result.iloc[0])
    assert np.isclose(result.iloc[1], 0.10)
    assert np.isclose(result.iloc[2], 0.10)


def test_volume_momentum_negative():
    """Test volume momentum with declining volume."""
    tx_volume = pd.Series([1000.0, 900.0, 810.0])

    result = volume_momentum(tx_volume, period=1)

    assert np.isclose(result.iloc[1], -0.10)
    assert np.isclose(result.iloc[2], -0.10)


def test_nupl_regime_all_buckets():
    """Test NUPL regime classification for all buckets."""
    nupl = pd.Series([-0.1, 0.0, 0.1, 0.3, 0.6, 0.8])

    result = nupl_regime(nupl)

    expected = pd.Series([-2.0, -1.0, -1.0, 0.0, 1.0, 2.0])
    pd.testing.assert_series_equal(result, expected)


def test_nupl_regime_boundaries():
    """Test NUPL regime at exact boundaries."""
    nupl = pd.Series([0.0, 0.25, 0.5, 0.75])

    result = nupl_regime(nupl)

    # Boundaries: 0 -> -1, 0.25 -> 0, 0.5 -> 1, 0.75 -> 2
    expected = pd.Series([-1.0, 0.0, 1.0, 2.0])
    pd.testing.assert_series_equal(result, expected)


def test_puell_signal_buy():
    """Test Puell Multiple buy signal when < 0.5."""
    puell_multiple = pd.Series([0.6, 0.4, 0.3, 0.5, 0.49])

    result = puell_signal(puell_multiple)

    expected = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0])
    pd.testing.assert_series_equal(result, expected)


def test_puell_signal_normal():
    """Test Puell Multiple returns 0 when >= 0.5."""
    puell_multiple = pd.Series([0.5, 1.0, 2.0])

    result = puell_signal(puell_multiple)

    expected = pd.Series([0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_supply_in_profit_extreme_euphoria():
    """Test supply in profit returns -1 when > 95 (euphoria)."""
    sip = pd.Series([90.0, 96.0, 98.0, 95.0])

    result = supply_in_profit_extreme(sip)

    expected = pd.Series([0.0, -1.0, -1.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_supply_in_profit_extreme_bear():
    """Test supply in profit returns 1 when < 50 (deep bear)."""
    sip = pd.Series([60.0, 45.0, 30.0, 50.0])

    result = supply_in_profit_extreme(sip)

    expected = pd.Series([0.0, 1.0, 1.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_supply_in_profit_extreme_normal():
    """Test supply in profit returns 0 in normal range."""
    sip = pd.Series([50.0, 70.0, 95.0])

    result = supply_in_profit_extreme(sip)

    expected = pd.Series([0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected)


# =============================================================================
# ETH Feature Tests
# =============================================================================


def test_gas_fee_zscore_high():
    """Test gas fee z-score with high fees."""
    # 90 days at 1M, then spike to 3M
    data = [1_000_000.0] * 90 + [3_000_000.0] * 10
    fee_total_usd = pd.Series(data)

    result = gas_fee_zscore(fee_total_usd, window=90)

    # First 89 should be NaN
    assert result.iloc[:89].isna().all()

    # Spike should give positive z-score
    assert result.iloc[-1] > 2.0


def test_gas_fee_zscore_low():
    """Test gas fee z-score with low fees."""
    # 90 days at 1M, then drop to 100K
    data = [1_000_000.0] * 90 + [100_000.0] * 10
    fee_total_usd = pd.Series(data)

    result = gas_fee_zscore(fee_total_usd, window=90)

    # Drop should give negative z-score
    assert result.iloc[-1] < -2.0


def test_gas_fee_zscore_zero_std():
    """Test gas fee z-score handles constant fees."""
    fee_total_usd = pd.Series([1_000_000.0] * 100)

    result = gas_fee_zscore(fee_total_usd, window=90)

    # Constant values should give NaN (zero std)
    assert pd.isna(result.iloc[-1])


def test_eth_address_momentum_positive():
    """Test ETH address momentum with rising addresses."""
    active_addresses = pd.Series([500_000.0, 550_000.0, 600_000.0])

    result = eth_address_momentum(active_addresses, period=1)

    assert pd.isna(result.iloc[0])
    assert np.isclose(result.iloc[1], 0.10)
    assert np.isclose(result.iloc[2], 600_000.0/550_000.0 - 1)


def test_eth_address_momentum_negative():
    """Test ETH address momentum with declining addresses."""
    active_addresses = pd.Series([500_000.0, 450_000.0, 400_000.0])

    result = eth_address_momentum(active_addresses, period=1)

    assert np.isclose(result.iloc[1], -0.10)
    # pct_change: (400_000 - 450_000) / 450_000 = -50_000 / 450_000
    assert np.isclose(result.iloc[2], (400_000.0 - 450_000.0) / 450_000.0)


def test_eth_transfer_value_zscore_high():
    """Test ETH transfer value z-score with high transfers."""
    # 60 days at 10B, then spike to 30B
    data = [10_000_000_000.0] * 60 + [30_000_000_000.0] * 10
    transfer_value_adj = pd.Series(data)

    result = eth_transfer_value_zscore(transfer_value_adj, window=60)

    # First 59 should be NaN
    assert result.iloc[:59].isna().all()

    # Spike should give positive z-score
    assert result.iloc[-1] > 2.0


def test_eth_transfer_value_zscore_low():
    """Test ETH transfer value z-score with low transfers."""
    # 60 days at 10B, then drop to 2B
    data = [10_000_000_000.0] * 60 + [2_000_000_000.0] * 10
    transfer_value_adj = pd.Series(data)

    result = eth_transfer_value_zscore(transfer_value_adj, window=60)

    # Drop should give negative z-score
    assert result.iloc[-1] < -2.0


def test_eth_nvt_signal_high():
    """Test ETH NVT signal with high NVT (overvalued)."""
    # Stable market cap, declining transfer value -> rising NVT
    market_cap = pd.Series([100_000_000_000.0] * 100)
    transfer_value_base = [10_000_000_000.0] * 90
    transfer_value_drop = [2_000_000_000.0] * 10  # NVT will spike
    transfer_value_adj = pd.Series(transfer_value_base + transfer_value_drop)

    result = eth_nvt_signal(market_cap, transfer_value_adj, window=90)

    # First 89 should be NaN
    assert result.iloc[:89].isna().all()

    # When transfer value drops, NVT spikes -> positive z-score
    assert result.iloc[-1] > 0


def test_eth_nvt_signal_zero_transfer():
    """Test ETH NVT signal handles zero transfer value."""
    market_cap = pd.Series([100_000_000_000.0] * 100)
    transfer_value_adj = pd.Series([0.0] * 100)

    result = eth_nvt_signal(market_cap, transfer_value_adj, window=90)

    # Division by zero should give NaN
    assert result.isna().all()


def test_eth_btc_ratio_momentum_eth_outperforms():
    """Test ETH/BTC ratio momentum when ETH outperforms."""
    # ETH: 1000 -> 1500 (+50%)
    # BTC: 20000 -> 22000 (+10%)
    # Ratio: 0.05 -> 0.0682 (+36.4%)
    eth_price = pd.Series([1000.0] * 30 + [1500.0])
    btc_price = pd.Series([20000.0] * 30 + [22000.0])

    result = eth_btc_ratio_momentum(eth_price, btc_price, period=30)

    # First 30 should be NaN
    assert result.iloc[:30].isna().all()

    # Ratio momentum should be positive (ETH outperforming)
    assert result.iloc[-1] > 0


def test_eth_btc_ratio_momentum_btc_outperforms():
    """Test ETH/BTC ratio momentum when BTC outperforms."""
    # ETH: 1000 -> 1100 (+10%)
    # BTC: 20000 -> 30000 (+50%)
    # Ratio: 0.05 -> 0.0367 (-26.7%)
    eth_price = pd.Series([1000.0] * 30 + [1100.0])
    btc_price = pd.Series([20000.0] * 30 + [30000.0])

    result = eth_btc_ratio_momentum(eth_price, btc_price, period=30)

    # Ratio momentum should be negative (BTC outperforming)
    assert result.iloc[-1] < 0


def test_eth_btc_ratio_momentum_zero_btc():
    """Test ETH/BTC ratio momentum handles zero BTC price."""
    eth_price = pd.Series([1000.0] * 31)
    btc_price = pd.Series([0.0] * 31)

    result = eth_btc_ratio_momentum(eth_price, btc_price, period=30)

    # Division by zero should give NaN
    assert result.isna().all()


def test_eth_btc_correlation_regime_high():
    """Test ETH/BTC correlation with highly correlated returns."""
    # Create identical returns -> correlation = 1
    returns = [0.01, -0.02, 0.03, -0.01, 0.02] * 20  # 100 values
    eth_returns = pd.Series(returns)
    btc_returns = pd.Series(returns)

    result = eth_btc_correlation_regime(eth_returns, btc_returns, window=60)

    # First 59 should be NaN
    assert result.iloc[:59].isna().all()

    # Correlation should be ~1.0
    assert np.isclose(result.iloc[-1], 1.0, atol=0.01)


def test_eth_btc_correlation_regime_low():
    """Test ETH/BTC correlation with uncorrelated returns."""
    # Create uncorrelated returns
    np.random.seed(42)
    eth_returns = pd.Series(np.random.randn(100) * 0.02)
    btc_returns = pd.Series(np.random.randn(100) * 0.02)

    result = eth_btc_correlation_regime(eth_returns, btc_returns, window=60)

    # First 59 should be NaN
    assert result.iloc[:59].isna().all()

    # Correlation should be close to 0 (not perfectly 0 due to randomness)
    assert abs(result.iloc[-1]) < 0.5


def test_eth_btc_correlation_regime_negative():
    """Test ETH/BTC correlation with negatively correlated returns."""
    # Create inverse returns -> correlation = -1
    base_returns = [0.01, -0.02, 0.03, -0.01, 0.02] * 20  # 100 values
    eth_returns = pd.Series(base_returns)
    btc_returns = pd.Series([-x for x in base_returns])

    result = eth_btc_correlation_regime(eth_returns, btc_returns, window=60)

    # Correlation should be ~-1.0
    assert np.isclose(result.iloc[-1], -1.0, atol=0.01)


def test_eth_btc_correlation_regime_insufficient_data():
    """Test ETH/BTC correlation with insufficient data."""
    eth_returns = pd.Series([0.01] * 50)
    btc_returns = pd.Series([0.01] * 50)

    result = eth_btc_correlation_regime(eth_returns, btc_returns, window=60)

    # All should be NaN (need 60 values)
    assert result.isna().all()


# =============================================================================
# Edge Cases and NaN Handling
# =============================================================================


def test_all_functions_handle_empty_series():
    """Test that all functions handle empty series gracefully."""
    empty = pd.Series([], dtype=float)

    # BTC features
    assert hash_ribbon(empty).empty
    assert nvt_zscore(empty).empty
    assert mvrv_signal(empty).empty
    assert sopr_signal(empty).empty
    assert address_momentum(empty).empty
    assert volume_momentum(empty).empty
    assert nupl_regime(empty).empty
    assert puell_signal(empty).empty
    assert supply_in_profit_extreme(empty).empty

    # ETH features
    assert gas_fee_zscore(empty).empty
    assert eth_address_momentum(empty).empty
    assert eth_transfer_value_zscore(empty).empty
    assert eth_nvt_signal(empty, empty).empty
    assert eth_btc_ratio_momentum(empty, empty).empty
    assert eth_btc_correlation_regime(empty, empty).empty


def test_all_functions_preserve_index():
    """Test that all functions preserve the input index."""
    index = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.Series(range(100), index=index, dtype=float)

    # Test a few representative functions
    result = hash_ribbon(data)
    assert result.index.equals(index)

    result = nvt_zscore(data)
    assert result.index.equals(index)

    result = nupl_regime(data / 100)  # Normalize to 0-1 range
    assert result.index.equals(index)


def test_functions_with_nan_input():
    """Test that functions handle NaN values in input."""
    data_with_nan = pd.Series([100.0, np.nan, 110.0, 120.0, np.nan, 130.0])

    # These should not crash
    result = address_momentum(data_with_nan, period=1)
    assert result is not None

    result = volume_momentum(data_with_nan, period=1)
    assert result is not None

    result = mvrv_signal(data_with_nan)
    assert result is not None
