"""Tests for composite on-chain regime detection."""

import numpy as np
import pandas as pd
import pytest

from sparky.features.onchain_regime import onchain_regime_signal, onchain_regime_with_positions


@pytest.fixture
def bullish_signals():
    """All signals bullish (+1)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return {
        "mvrv": pd.Series(1.0, index=idx),
        "sopr": pd.Series(1.0, index=idx),
        "netflow": pd.Series(1.0, index=idx),
    }


@pytest.fixture
def bearish_signals():
    """All signals bearish (-1)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return {
        "mvrv": pd.Series(-1.0, index=idx),
        "sopr": pd.Series(-1.0, index=idx),
        "netflow": pd.Series(-1.0, index=idx),
    }


@pytest.fixture
def mixed_signals():
    """Mixed signals."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return {
        "mvrv": pd.Series(1.0, index=idx),
        "sopr": pd.Series(-1.0, index=idx),
        "netflow": pd.Series(0.0, index=idx),
    }


class TestOnchainRegimeSignal:
    def test_all_bullish_gives_long(self, bullish_signals):
        result = onchain_regime_signal(bullish_signals, persistence_days=1)
        assert (result == 1.0).all()

    def test_all_bearish_gives_flat(self, bearish_signals):
        result = onchain_regime_signal(bearish_signals, persistence_days=1)
        assert (result == 0.0).all()

    def test_equal_weights_default(self, mixed_signals):
        result = onchain_regime_signal(mixed_signals, persistence_days=1)
        # Weighted sum: (1 + -1 + 0) / 3 = 0.0, not > 0 -> flat
        assert (result == 0.0).all()

    def test_custom_weights(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        signals = {
            "strong": pd.Series(1.0, index=idx),
            "weak": pd.Series(-1.0, index=idx),
        }
        weights = {"strong": 0.8, "weak": 0.2}
        result = onchain_regime_signal(signals, weights=weights, persistence_days=1)
        # Score = 0.8*1 + 0.2*(-1) = 0.6 > 0 -> long
        assert (result == 1.0).all()

    def test_persistence_filter_delays_switch(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        sig = pd.Series([-1.0] * 10 + [1.0] * 10, index=idx)
        signals = {"only": sig}
        weights = {"only": 1.0}

        result = onchain_regime_signal(signals, weights=weights, persistence_days=3)
        # Should not switch immediately at day 10
        assert result.iloc[10] == 0.0
        # Should eventually switch
        assert result.iloc[-1] == 1.0

    def test_persistence_1_is_immediate(self, bullish_signals):
        result = onchain_regime_signal(bullish_signals, persistence_days=1)
        assert (result == 1.0).all()

    def test_threshold_effect(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        signals = {"a": pd.Series(0.3, index=idx)}
        weights = {"a": 1.0}
        # With threshold 0.0 -> long (0.3 > 0)
        r1 = onchain_regime_signal(signals, weights=weights, threshold=0.0, persistence_days=1)
        assert (r1 == 1.0).all()
        # With threshold 0.5 -> flat (0.3 < 0.5)
        r2 = onchain_regime_signal(signals, weights=weights, threshold=0.5, persistence_days=1)
        assert (r2 == 0.0).all()

    def test_output_is_binary(self, bullish_signals):
        result = onchain_regime_signal(bullish_signals, persistence_days=1)
        assert set(result.unique()).issubset({0.0, 1.0})


class TestOnchainRegimeWithPositions:
    def test_output_columns(self, bullish_signals):
        df = onchain_regime_with_positions(bullish_signals, persistence_days=1)
        assert "mvrv" in df.columns
        assert "sopr" in df.columns
        assert "netflow" in df.columns
        assert "composite_score" in df.columns
        assert "regime_signal" in df.columns
        assert "position" in df.columns

    def test_position_matches_regime(self, bullish_signals):
        df = onchain_regime_with_positions(bullish_signals, persistence_days=1)
        pd.testing.assert_series_equal(df["position"], df["regime_signal"], check_names=False)

    def test_composite_score_values(self, bullish_signals):
        df = onchain_regime_with_positions(bullish_signals, persistence_days=1)
        # Equal weights, all 1.0: score = 1/3 + 1/3 + 1/3 = 1.0
        assert np.isclose(df["composite_score"].iloc[0], 1.0)
