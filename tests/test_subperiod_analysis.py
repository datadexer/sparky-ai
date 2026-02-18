"""Tests for subperiod_analysis() in sweep_utils."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "infra"))
from sweep_utils import subperiod_analysis  # noqa: E402


def _make_prices(start, end, trend=0.0002):
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    returns = np.random.default_rng(42).normal(trend, 0.02, len(idx))
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=idx, name="close")
    return prices


def _make_positions(prices, long_frac=0.6):
    rng = np.random.default_rng(99)
    pos = pd.Series(
        rng.choice([1.0, 0.0], size=len(prices), p=[long_frac, 1 - long_frac]),
        index=prices.index,
    )
    return pos


EXPECTED_KEYS = {"sharpe", "max_drawdown", "annual_return", "n_trades", "win_rate", "bh_sharpe"}


class TestOutputStructure:
    def test_full_and_2020_keys(self):
        prices = _make_prices("2019-01-01", "2023-12-31")
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        assert "full" in result
        assert "2020+" in result
        for label in result:
            assert set(result[label].keys()) == EXPECTED_KEYS

    def test_2017_key_present_for_wide_range(self):
        prices = _make_prices("2016-01-01", "2023-12-31")
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        assert "2017+" in result


class TestPartialCoverage:
    def test_prices_starting_2022(self):
        prices = _make_prices("2022-01-01", "2023-12-31")
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        assert "full" in result
        assert "2020+" in result

    def test_2017_present_when_data_starts_2020(self):
        prices = _make_prices("2020-06-01", "2023-12-31")
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        assert "2017+" in result  # clips to actual start, still produces results


class TestShortData:
    def test_20_days_only_full(self):
        prices = _make_prices("2023-12-01", "2023-12-20")
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        # 20 days < 30 threshold, all sub-periods should be skipped
        assert result == {}

    def test_35_days_produces_full(self):
        prices = _make_prices("2023-11-01", "2023-12-15")
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        assert "full" in result


class TestBuyAndHold:
    def test_rising_prices_positive_bh_sharpe(self):
        idx = pd.date_range("2019-01-01", "2023-12-31", freq="D", tz="UTC")
        prices = pd.Series(
            np.linspace(100, 500, len(idx)) + np.random.default_rng(7).normal(0, 2, len(idx)),
            index=idx,
        )
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        assert result["full"]["bh_sharpe"] > 0


class TestFiniteValues:
    @pytest.mark.parametrize("start,end", [("2019-01-01", "2023-12-31"), ("2021-06-01", "2023-06-01")])
    def test_no_inf_nan(self, start, end):
        prices = _make_prices(start, end)
        positions = _make_positions(prices)
        result = subperiod_analysis(prices, positions, 0.003)
        for label, metrics in result.items():
            for key, val in metrics.items():
                assert math.isfinite(val), f"{label}.{key} = {val}"
