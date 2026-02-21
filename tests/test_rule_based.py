"""Tests for rule-based backtest utilities."""

import numpy as np
import pandas as pd
import pytest

from sparky.backtest.rule_based import compute_strategy_metrics, net_ret, subperiod_analysis


@pytest.fixture
def daily_index():
    return pd.date_range("2019-01-01", periods=1800, freq="D", tz="UTC")


@pytest.fixture
def prices(daily_index):
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.02, len(daily_index))
    p = 100.0 * np.cumprod(1 + returns)
    return pd.Series(p, index=daily_index)


@pytest.fixture
def binary_positions(daily_index):
    pos = pd.Series(1.0, index=daily_index)
    pos.iloc[100:200] = 0.0
    pos.iloc[500:600] = 0.0
    pos.iloc[900:1000] = 0.0
    return pos


class TestNetRet:
    def test_basic_output(self, prices, binary_positions):
        ret = net_ret(prices, binary_positions, 0.003)
        assert isinstance(ret, pd.Series)
        assert len(ret) > 0
        assert isinstance(ret.index, pd.DatetimeIndex)

    def test_no_costs_matches_buy_and_hold(self, prices, daily_index):
        pos = pd.Series(1.0, index=daily_index)
        ret = net_ret(prices, pos, 0.0)
        bh = prices.pct_change().dropna()
        # With constant position=1, lagged pos=1 everywhere after first,
        # so net_ret should match bh (offset by 1 for the shift)
        common = ret.index.intersection(bh.index)
        pd.testing.assert_series_equal(ret[common], bh[common], atol=1e-10)

    def test_costs_reduce_returns(self, prices, binary_positions):
        ret_no_cost = net_ret(prices, binary_positions, 0.0)
        ret_with_cost = net_ret(prices, binary_positions, 0.003)
        assert ret_with_cost.sum() < ret_no_cost.sum()

    def test_continuous_positions(self, prices, daily_index):
        pos = pd.Series(0.5, index=daily_index)
        ret = net_ret(prices, pos, 0.003)
        assert len(ret) > 0
        # Half position should give roughly half the returns
        full_ret = net_ret(prices, pd.Series(1.0, index=daily_index), 0.0)
        ratio = ret.mean() / full_ret.mean()
        assert 0.3 < ratio < 0.7

    def test_negative_positions(self, prices, daily_index):
        pos = pd.Series(-1.0, index=daily_index)
        ret = net_ret(prices, pos, 0.0)
        bh = prices.pct_change().dropna()
        # Short should be negative of long (approximately)
        common = ret.index.intersection(bh.index)
        np.testing.assert_allclose(ret[common].values, -bh[common].values, atol=1e-10)

    def test_requires_datetime_index(self):
        prices = pd.Series([100, 101, 102])
        positions = pd.Series([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            net_ret(prices, positions, 0.003)

    def test_negative_cost_frac_raises(self, prices, daily_index):
        pos = pd.Series(1.0, index=daily_index)
        with pytest.raises(ValueError, match="non-negative"):
            net_ret(prices, pos, -0.001)

    def test_hand_computed_costs(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        prices = pd.Series([100, 102, 101, 103, 105], index=idx, dtype=float)
        positions = pd.Series([0, 1, 1, 0, 1], index=idx, dtype=float)
        cf = 0.01  # 1% cost for easy math

        ret = net_ret(prices, positions, cf)

        # Day 1: lagged_pos=0, pr=0.02, cost=0 -> 0
        # Day 2: lagged_pos=1, pr=-0.0098.., cost=0 -> -0.0098..
        # Day 3: lagged_pos=1, pr=0.0198.., cost=1*0.01=0.01 -> 0.0098..
        # Day 4: lagged_pos=0, pr=0.0194.., cost=1*0.01=0.01 -> -0.01
        assert len(ret) == 4


class TestSubperiodAnalysis:
    def test_output_structure(self, prices, binary_positions):
        result = subperiod_analysis(prices, binary_positions, 0.003)
        assert "full" in result
        for label, data in result.items():
            assert "sharpe" in data
            assert "max_drawdown" in data
            assert "annual_return" in data
            assert "n_trades" in data
            assert "win_rate" in data
            assert "bh_sharpe" in data

    def test_subperiods_present(self, prices, binary_positions):
        result = subperiod_analysis(prices, binary_positions, 0.003)
        assert "full" in result
        # 2020+ should be present since data starts 2019 and has 1800 days
        assert "2020+" in result


class TestComputeStrategyMetrics:
    def test_returns_none_for_few_trades(self, prices, daily_index):
        pos = pd.Series(1.0, index=daily_index)  # no trades
        result = compute_strategy_metrics(prices, pos, 0.003)
        assert result is None

    def test_returns_metrics_for_enough_trades(self, prices, binary_positions):
        result = compute_strategy_metrics(prices, binary_positions, 0.003)
        assert result is not None
        assert "sharpe" in result
        assert "dsr" in result
        assert "n_trades" in result
        assert result["n_trades"] >= 5

    def test_n_trials_passed_through(self, prices, binary_positions):
        result = compute_strategy_metrics(prices, binary_positions, 0.003, n_trials=100)
        assert result is not None
        assert result["n_trials"] == 100
