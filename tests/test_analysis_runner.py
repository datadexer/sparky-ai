"""Tests for analysis_runner validation and investigation functions.

Uses synthetic data via monkeypatching build_strategy to avoid real data deps.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "infra"))
sys.path.insert(0, str(ROOT / "src"))


def _synthetic_build_strategy(config):
    """Return synthetic prices/positions/df/ppy for testing."""
    rng = np.random.RandomState(42)
    n = 1000
    ppy = 2190  # 4h
    dates = pd.date_range("2019-07-01", periods=n, freq="4h", tz="UTC")
    prices = pd.Series(100 * np.cumprod(1 + rng.randn(n) * 0.005 + 0.0001), index=dates)
    positions = pd.Series(0.0, index=dates)
    # Simple alternating long/flat signal
    for i in range(0, n, 40):
        positions.iloc[i : min(i + 30, n)] = 1.0
    df = pd.DataFrame(
        {"close": prices, "open": prices * 0.999, "high": prices * 1.002, "low": prices * 0.998}, index=dates
    )
    return prices, positions, df, ppy


@pytest.fixture(autouse=True)
def _mock_build_strategy():
    with patch("analysis_runner.build_strategy", side_effect=_synthetic_build_strategy):
        yield


@pytest.fixture(autouse=True)
def _mock_load_data():
    """Mock _load_data for edge_attribution and target_vol_frontier."""

    def fake_load(asset, tf):
        rng = np.random.RandomState(42)
        n = 1000
        ppy = 2190
        dates = pd.date_range("2019-07-01", periods=n, freq="4h", tz="UTC")
        prices = pd.Series(100 * np.cumprod(1 + rng.randn(n) * 0.005 + 0.0001), index=dates)
        df = pd.DataFrame({"close": prices}, index=dates)
        return df, prices, ppy

    with patch(
        "analysis_runner.experiment_runner._load_data"
        if hasattr(sys.modules.get("analysis_runner", None), "experiment_runner")
        else "experiment_runner._load_data",
        side_effect=fake_load,
    ):
        yield


# We need to import after fixtures are defined but the module may cache
import analysis_runner as ar  # noqa: E402


SAMPLE_CONFIG = {
    "asset": "btc",
    "timeframe": "4h",
    "signal_type": "donchian",
    "signal_params": {"entry_period": 160, "exit_period": 25},
    "sizing": "inverse_vol",
    "sizing_params": {"vol_window": 30, "target_vol": 0.15},
}


class TestStressTest:
    def test_sharpe_decreases_with_cost(self):
        r = ar.stress_test(SAMPLE_CONFIG, cost_range_bps=[10, 30, 50, 100])
        assert r["status"] == "ok"
        sharpes = r["sharpe_per_cost"]
        vals = list(sharpes.values())
        # Generally decreasing (allow small non-monotonicity from noise)
        assert vals[0] >= vals[-1]

    def test_has_breakeven(self):
        r = ar.stress_test(SAMPLE_CONFIG)
        assert "breakeven_bps" in r
        assert r["breakeven_bps"] is not None


class TestBootstrapSharpe:
    def test_returns_ci(self):
        r = ar.bootstrap_sharpe(SAMPLE_CONFIG, n_samples=500, block_size=10)
        assert r["status"] == "ok"
        assert "ci_95" in r
        assert "percentiles" in r
        assert 5 in r["percentiles"]
        assert 95 in r["percentiles"]
        assert r["percentiles"][5] <= r["percentiles"][95]

    def test_distribution_has_correct_size(self):
        r = ar.bootstrap_sharpe(SAMPLE_CONFIG, n_samples=200, block_size=10)
        assert len(r["plots_data"]["sharpe_dist"]) == 200


class TestWalkForwardMulti:
    def test_correct_window_count(self):
        r = ar.walk_forward_multi(SAMPLE_CONFIG, window_sizes_days=[100])
        assert r["status"] == "ok"
        if 100 in r["results"]:
            # n~999 after dropna, windows of 100 → 9 windows
            assert r["results"][100]["n_windows"] >= 9


class TestSubsampleStability:
    def test_output_shape(self):
        rates = [0.1, 0.2, 0.3]
        r = ar.subsample_stability(SAMPLE_CONFIG, drop_rates=rates, repetitions=20)
        assert r["status"] == "ok"
        assert len(r["results"]) == len(rates)
        for rate in rates:
            assert "mean" in r["results"][rate]
            assert "std" in r["results"][rate]


class TestCpcvValidate:
    def test_basic(self):
        r = ar.cpcv_validate(SAMPLE_CONFIG, n_groups=4, purge_days=2)
        assert r["status"] == "ok"
        assert "pbo" in r
        assert 0 <= r["pbo"] <= 1
        assert r["n_paths"] == 6  # C(4,2)


class TestTradeProfile:
    def test_basic(self):
        r = ar.trade_profile(SAMPLE_CONFIG)
        assert r["status"] == "ok"
        assert r["n_trades"] > 0
        assert 0 <= r["win_rate"] <= 1
        assert r.get("avg_holding_periods", 0) > 0


class TestDrawdownAnalysis:
    def test_basic(self):
        r = ar.drawdown_analysis(SAMPLE_CONFIG)
        assert r["status"] == "ok"
        assert r["max_drawdown"] < 0


class TestRollingStability:
    def test_basic(self):
        r = ar.rolling_stability(SAMPLE_CONFIG, window_days=100)
        assert r["status"] == "ok"
        assert "mean_rolling_sharpe" in r
        assert len(r["plots_data"]["rolling_sharpe"]) > 0


class TestErrorHandling:
    def test_safe_catches_exception(self):
        def _always_fails(config):
            raise RuntimeError("test error")

        r = ar._safe(_always_fails, {})
        assert r["status"] == "error"
        assert "test error" in r["error"]


class TestFullBattery:
    def test_runs_without_crash(self):
        r = ar.run_full_validation_battery(SAMPLE_CONFIG)
        assert r["status"] == "ok"
        assert r["overall_verdict"] in ("PASS", "FAIL", "CONDITIONAL")
        assert isinstance(r["tests"], dict)
        assert len(r["tests"]) >= 10


class TestFullInvestigation:
    def test_runs_without_crash(self):
        r = ar.run_full_investigation(SAMPLE_CONFIG)
        assert r["status"] == "ok"
        assert "trade_profile" in r["results"]
        assert "edge_attribution" in r["results"]
        assert "regime_decomposition" in r["results"]
