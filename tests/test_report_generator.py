"""Smoke tests for report_generator — plots produce files, HTML renders."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "infra"))

import report_generator as rg  # noqa: E402


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def _synth_validation_results():
    return {
        "status": "ok",
        "overall_verdict": "CONDITIONAL",
        "hard_fails": [],
        "soft_fails": ["rolling_stability"],
        "tests": {
            "stress_test": {
                "status": "ok",
                "verdict": "pass",
                "breakeven_bps": 90,
                "results": {30: {"sharpe": 1.5}, 50: {"sharpe": 1.1}},
                "sharpe_per_cost": {30: 1.5, 50: 1.1, 70: 0.8, 90: 0.3},
                "plots_data": {"cost_bps": [30, 50, 70, 90], "sharpe": [1.5, 1.1, 0.8, 0.3], "breakeven": 90},
            },
            "bootstrap_sharpe": {
                "status": "ok",
                "verdict": "pass",
                "percentiles": {5: 0.9, 25: 1.2, 50: 1.5, 75: 1.8, 95: 2.1},
                "plots_data": {
                    "sharpe_dist": list(np.random.RandomState(42).randn(500) * 0.3 + 1.5),
                    "percentiles": {5: 0.9, 25: 1.2, 50: 1.5, 75: 1.8, 95: 2.1},
                },
            },
            "drawdown_analysis": {
                "status": "ok",
                "verdict": "pass",
                "max_drawdown": -0.15,
                "plots_data": {
                    "dd_series": list(-np.abs(np.random.RandomState(42).randn(200)) * 0.02),
                    "index": [f"2020-{i:03d}" for i in range(200)],
                    "max_dd": -0.15,
                },
            },
            "rolling_stability": {
                "status": "ok",
                "verdict": "soft_fail",
                "flagged_periods": [{"start": "2022-01-01", "end": "2022-06-30", "duration_days": 180}],
                "mean_rolling_sharpe": 1.2,
                "plots_data": {
                    "rolling_sharpe": list(np.random.RandomState(42).randn(300) * 0.5 + 1.0),
                    "dates": [f"2020-{i:03d}" for i in range(300)],
                    "flagged": [],
                },
            },
        },
    }


def _synth_investigation_results():
    return {
        "status": "ok",
        "results": {
            "trade_profile": {
                "status": "ok",
                "n_trades": 50,
                "win_rate": 0.55,
                "avg_win": 0.03,
                "avg_loss": -0.02,
                "profit_factor": 1.8,
                "median_holding_periods": 12,
                "plots_data": {
                    "holding_periods": list(range(5, 55)),
                    "trade_returns": list(np.random.RandomState(42).randn(50) * 0.02),
                },
            },
            "edge_attribution": {
                "status": "ok",
                "full_sharpe": 1.5,
                "random_entry_sharpe": 0.3,
                "flat_sizing_sharpe": 1.2,
                "signal_edge": 1.2,
                "sizing_edge": 0.3,
            },
            "regime_decomposition": {
                "status": "ok",
                "regime_metrics": {
                    "bull": {"sharpe": 2.0, "max_drawdown": -0.05, "win_rate": 0.6, "n_obs": 200},
                    "bear": {"sharpe": 0.5, "max_drawdown": -0.15, "win_rate": 0.45, "n_obs": 100},
                    "sideways": {"sharpe": 1.0, "max_drawdown": -0.08, "win_rate": 0.5, "n_obs": 150},
                },
                "crisis_metrics": {"COVID crash": {"total_return": -0.05, "n_obs": 20}},
                "monthly_heatmap": {2020: {1: 5.0, 2: -3.0, 3: -10.0, 4: 8.0}, 2021: {1: 15.0, 5: -20.0}},
                "plots_data": {
                    "regime_metrics": {"bull": {"sharpe": 2.0}, "bear": {"sharpe": 0.5}, "sideways": {"sharpe": 1.0}},
                    "monthly_heatmap": {2020: {1: 5.0, 2: -3.0, 3: -10.0, 4: 8.0}, 2021: {1: 15.0, 5: -20.0}},
                },
            },
        },
    }


class TestPlotFunctions:
    def test_cost_curve(self, tmp_dir):
        p = rg.plot_cost_curve(
            {"cost_bps": [30, 50, 70], "sharpe": [1.5, 1.1, 0.8], "breakeven": 90}, tmp_dir / "cost.png"
        )
        assert Path(p).exists()

    def test_bootstrap_distribution(self, tmp_dir):
        dist = list(np.random.RandomState(42).randn(200) * 0.3 + 1.5)
        p = rg.plot_bootstrap_distribution(
            {"sharpe_dist": dist, "percentiles": {5: 0.9, 50: 1.5, 95: 2.1}}, tmp_dir / "boot.png"
        )
        assert Path(p).exists()

    def test_drawdown_series(self, tmp_dir):
        dd = list(-np.abs(np.random.RandomState(42).randn(100)) * 0.02)
        p = rg.plot_drawdown_series(
            {"dd_series": dd, "index": [f"d{i}" for i in range(100)], "max_dd": min(dd)}, tmp_dir / "dd.png"
        )
        assert Path(p).exists()

    def test_regime_performance(self, tmp_dir):
        data = {"regime_metrics": {"bull": {"sharpe": 2.0}, "bear": {"sharpe": 0.5}, "sideways": {"sharpe": 1.0}}}
        p = rg.plot_regime_performance(data, tmp_dir / "regime.png")
        assert Path(p).exists()

    def test_monthly_heatmap(self, tmp_dir):
        data = {"monthly_heatmap": {2020: {1: 5.0, 2: -3.0, 6: 2.0}, 2021: {3: 10.0, 7: -5.0}}}
        p = rg.plot_monthly_heatmap(data, tmp_dir / "heatmap.png")
        assert Path(p).exists()


class TestHTMLReports:
    def test_candidate_report(self, tmp_dir):
        val = _synth_validation_results()
        inv = _synth_investigation_results()
        path = rg.generate_candidate_report("test_candidate", inv, val, str(tmp_dir))
        assert Path(path).exists()
        html = Path(path).read_text()
        assert "test_candidate" in html
        assert "CONDITIONAL" in html
        assert "Trade Profile" in html

    def test_project_summary(self, tmp_dir):
        all_results = {
            "cand_a": {"validation": _synth_validation_results(), "dsr": 0.96},
            "cand_b": {"validation": _synth_validation_results(), "dsr": 0.91},
        }
        path = rg.generate_project_summary("test_project", all_results, str(tmp_dir))
        assert Path(path).exists()
        html = Path(path).read_text()
        assert "cand_a" in html
        assert "cand_b" in html
