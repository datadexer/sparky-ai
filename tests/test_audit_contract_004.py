"""Tests for the audit_contract_004.py script.

Tests analyze_group and generate_report edge cases including None handling,
dynamic T from n_observations, and fallback behavior.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts/ to path so we can import audit_contract_004
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from audit_contract_004 import analyze_group, generate_report


class TestAnalyzeGroup:
    def test_returns_none_for_all_none_sharpes(self):
        """analyze_group returns None when all runs have sharpe=None."""
        runs = [
            {"sharpe": None, "name": "run1", "run_id": "r1"},
            {"sharpe": None, "name": "run2", "run_id": "r2"},
        ]
        result = analyze_group("sweep", runs)
        assert result is None

    def test_uses_n_observations_for_T(self):
        """analyze_group uses n_observations from runs for T when available."""
        runs = [
            {"sharpe": 1.0, "name": "run1", "run_id": "r1", "dsr": 0.8, "max_drawdown": -0.1, "n_observations": 10000},
            {"sharpe": 0.8, "name": "run2", "run_id": "r2", "dsr": 0.6, "max_drawdown": -0.2, "n_observations": 12000},
        ]
        result = analyze_group("sweep", runs)
        assert result is not None
        # Median of [10000, 12000] = 11000
        assert result["T"] == 11000
        assert "median" in result["T_source"]

    def test_falls_back_to_default_T(self):
        """analyze_group falls back to 8760*5 when n_observations absent."""
        runs = [
            {"sharpe": 1.0, "name": "run1", "run_id": "r1", "dsr": 0.8, "max_drawdown": -0.1},
            {"sharpe": 0.8, "name": "run2", "run_id": "r2", "dsr": 0.6, "max_drawdown": -0.2},
        ]
        result = analyze_group("sweep", runs)
        assert result is not None
        assert result["T"] == 8760 * 5
        assert "default" in result["T_source"]

    def test_ignores_none_n_observations(self):
        """analyze_group ignores runs with n_observations=None."""
        runs = [
            {"sharpe": 1.0, "name": "run1", "run_id": "r1", "dsr": 0.8, "max_drawdown": -0.1, "n_observations": None},
            {"sharpe": 0.8, "name": "run2", "run_id": "r2", "dsr": 0.6, "max_drawdown": -0.2, "n_observations": 20000},
        ]
        result = analyze_group("sweep", runs)
        assert result is not None
        assert result["T"] == 20000


class TestGenerateReport:
    def _make_analysis(self, **overrides):
        """Create a minimal analysis dict for testing generate_report."""
        base = {
            "group": "sweep",
            "n_runs": 5,
            "n_with_sharpe": 5,
            "best_sharpe": 1.2,
            "best_run_name": "test_run",
            "best_run_id": "r1",
            "best_dsr": 0.85,
            "best_max_dd": -0.15,
            "expected_max_sharpe": 0.5,
            "sharpe_vs_expected": 0.7,
            "mean_sharpe": 0.8,
            "std_sharpe": 0.3,
            "median_sharpe": 0.9,
            "all_sharpes": [0.5, 0.7, 0.9, 1.0, 1.2],
            "T": 43800,
            "T_source": "default (8760*5)",
        }
        base.update(overrides)
        return base

    def test_best_dsr_none_shows_na(self):
        """generate_report with best_dsr=None should show 'N/A', not crash."""
        analysis = self._make_analysis(best_dsr=None)
        report = generate_report([analysis], total_runs=5)
        assert "N/A" in report
        # Should not contain formatting error
        assert "None" not in report or "N/A" in report

    def test_best_max_dd_none_shows_na(self):
        """generate_report with best_max_dd=None should show 'N/A', not crash."""
        analysis = self._make_analysis(best_max_dd=None)
        report = generate_report([analysis], total_runs=5)
        assert "N/A" in report

    def test_report_includes_T_source(self):
        """generate_report should include T and T_source in per-step detail."""
        analysis = self._make_analysis(T=11000, T_source="median of 5 runs")
        report = generate_report([analysis], total_runs=5)
        assert "11000" in report
        assert "median of 5 runs" in report

    def test_skips_none_analyses(self):
        """generate_report should skip None analyses without crashing."""
        analysis = self._make_analysis()
        report = generate_report([None, analysis, None], total_runs=5)
        assert "sweep" in report
