"""Tests for walk-forward validation."""

import numpy as np
import pandas as pd

from sparky.validation.walk_forward import run_walk_forward, walk_forward_summary


def _make_prices(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="D", tz="UTC")
    returns = rng.normal(0.0003, 0.02, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    return prices


def _simple_signal(prices):
    """Always long."""
    return pd.Series(1.0, index=prices.index)


class TestRunWalkForward:
    def test_returns_expected_keys(self):
        prices = _make_prices()
        result = run_walk_forward(prices, _simple_signal, n_folds=3, min_train_periods=365, cost_frac=0.0015)
        assert "folds" in result
        assert "aggregate_sharpe" in result
        assert "retention_ratio" in result
        assert "all_folds_positive" in result
        assert "n_folds" in result
        assert result["n_folds"] > 0

    def test_fold_count(self):
        prices = _make_prices()
        result = run_walk_forward(prices, _simple_signal, n_folds=4, min_train_periods=365, cost_frac=0.0015)
        assert result["n_folds"] <= 4

    def test_fold_keys(self):
        prices = _make_prices()
        result = run_walk_forward(prices, _simple_signal, n_folds=3, min_train_periods=365, cost_frac=0.0015)
        for fold in result["folds"]:
            assert "sharpe" in fold
            assert "max_drawdown" in fold
            assert "n_trades" in fold
            assert "train_end" in fold
            assert "test_start" in fold


class TestWalkForwardSummary:
    def test_returns_string(self):
        prices = _make_prices()
        result = run_walk_forward(prices, _simple_signal, n_folds=3, min_train_periods=365, cost_frac=0.0015)
        summary = walk_forward_summary(result)
        assert isinstance(summary, str)
        assert "Walk-Forward" in summary
        assert "retention" in summary
