"""Tests for guardrails module.

Tests all 11 individual checks, orchestrators, JSONL logging,
and has_blocking_failure logic.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sparky.tracking.guardrails import (
    _VALID_SEVERITIES,
    GuardrailResult,
    check_consistency,
    check_costs_specified,
    check_dsr_threshold,
    check_holdout_boundary,
    check_max_drawdown,
    check_minimum_samples,
    check_minimum_trades,
    check_no_lookahead,
    check_param_data_ratio,
    check_returns_distribution,
    check_sharpe_sanity,
    has_blocking_failure,
    log_results,
    run_post_checks,
    run_pre_checks,
)

# === Fixtures ===


@pytest.fixture
def training_data():
    """Generate synthetic training data within holdout boundary."""
    dates = pd.date_range("2020-01-01", periods=5000, freq="h", tz="UTC")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "feature_1": np.random.randn(5000),
            "feature_2": np.random.randn(5000),
            "feature_3": np.random.randn(5000),
            "target_1h": np.random.randint(0, 2, 5000),
        },
        index=dates,
    )
    return data


@pytest.fixture
def good_config():
    """Config that should pass all pre-checks."""
    return {
        "features": ["feature_1", "feature_2", "feature_3"],
        "target": "target_1h",
        "transaction_costs_bps": 50.0,
    }


@pytest.fixture
def good_returns():
    """Returns series that should pass post-checks."""
    np.random.seed(42)
    return np.random.randn(2000) * 0.01  # small daily returns


@pytest.fixture
def good_metrics():
    """Metrics dict that should pass post-checks."""
    return {
        "sharpe": 1.2,
        "dsr": 0.85,
        "max_drawdown": -0.15,
        "rolling_sharpe_std": 0.5,
    }


# === Pre-experiment checks ===


class TestCheckHoldoutBoundary:
    def test_passes_for_training_data(self, training_data):
        result = check_holdout_boundary(training_data, asset="btc")
        assert result.passed
        assert result.severity == "block"

    def test_fails_for_future_data(self, holdout_dates):
        # Data that extends past the holdout embargo boundary
        start = holdout_dates["max_training_ts"] - pd.Timedelta(days=30)
        dates = pd.date_range(start, periods=5000, freq="h", tz="UTC")
        data = pd.DataFrame({"x": range(5000)}, index=dates)
        result = check_holdout_boundary(data, asset="btc")
        assert not result.passed
        assert result.severity == "block"


class TestCheckMinimumSamples:
    def test_passes_with_enough_data(self, training_data):
        result = check_minimum_samples(training_data)
        assert result.passed

    def test_fails_with_too_little_data(self):
        data = pd.DataFrame({"x": range(100)})
        result = check_minimum_samples(data)
        assert not result.passed
        assert result.severity == "block"

    def test_custom_threshold(self):
        data = pd.DataFrame({"x": range(500)})
        result = check_minimum_samples(data, min_samples=1000)
        assert not result.passed
        result2 = check_minimum_samples(data, min_samples=100)
        assert result2.passed


class TestCheckNoLookahead:
    def test_passes_when_target_not_in_features(self, good_config):
        result = check_no_lookahead(pd.DataFrame(), good_config)
        assert result.passed

    def test_fails_when_target_in_features(self):
        config = {
            "features": ["feature_1", "target_1h"],
            "target": "target_1h",
        }
        result = check_no_lookahead(pd.DataFrame(), config)
        assert not result.passed
        assert "look-ahead" in result.message.lower()


class TestCheckCostsSpecified:
    def test_passes_with_valid_costs(self, good_config):
        result = check_costs_specified(good_config)
        assert result.passed

    def test_fails_when_missing(self):
        result = check_costs_specified({})
        assert not result.passed
        assert result.severity == "block"

    def test_fails_when_too_low(self):
        result = check_costs_specified({"transaction_costs_bps": 1.0})
        assert not result.passed


class TestCheckParamDataRatio:
    def test_passes_with_low_ratio(self, good_config, training_data):
        result = check_param_data_ratio(good_config, training_data)
        assert result.passed

    def test_fails_with_high_ratio(self):
        config = {"features": [f"f{i}" for i in range(500)]}
        data = pd.DataFrame({"x": range(100)})
        result = check_param_data_ratio(config, data, max_ratio=0.1)
        assert not result.passed
        assert result.severity == "warn"


# === Post-experiment checks ===


class TestCheckSharpeSanity:
    def test_passes_normal_sharpe(self, good_metrics):
        result = check_sharpe_sanity(good_metrics)
        assert result.passed

    def test_fails_extreme_sharpe(self):
        result = check_sharpe_sanity({"sharpe": 10.0})
        assert not result.passed
        assert result.severity == "block"

    def test_fails_negative_extreme(self):
        result = check_sharpe_sanity({"sharpe": -5.0})
        assert not result.passed


class TestCheckMinimumTrades:
    def test_passes_with_enough_trades(self, good_returns, good_config):
        result = check_minimum_trades(good_returns, good_config)
        assert result.passed

    def test_fails_with_few_trades(self, good_config):
        # All same sign — very few "trades"
        returns = np.ones(100) * 0.01
        result = check_minimum_trades(returns, good_config)
        assert not result.passed


class TestCheckDsrThreshold:
    def test_passes_above_threshold(self):
        result = check_dsr_threshold({"dsr": 0.95})
        assert result.passed

    def test_fails_below_threshold(self):
        result = check_dsr_threshold({"dsr": 0.50})
        assert not result.passed
        assert result.severity == "info"

    def test_fails_when_missing(self):
        result = check_dsr_threshold({})
        assert not result.passed


class TestCheckMaxDrawdown:
    def test_passes_within_limit(self, good_metrics):
        result = check_max_drawdown(good_metrics)
        assert result.passed

    def test_fails_excessive_drawdown(self):
        result = check_max_drawdown({"max_drawdown": -0.60})
        assert not result.passed
        assert result.severity == "warn"


class TestCheckReturnsDistribution:
    def test_passes_normal_distribution(self, good_returns):
        result = check_returns_distribution(good_returns)
        assert result.passed

    def test_fails_extreme_kurtosis(self):
        # Create returns with extreme kurtosis
        returns = np.zeros(1000)
        returns[0] = 100  # One extreme outlier
        returns[1] = -100
        result = check_returns_distribution(returns)
        assert not result.passed

    def test_passes_few_returns(self):
        result = check_returns_distribution(np.array([0.01, 0.02]))
        assert result.passed  # too few to check


class TestCheckConsistency:
    def test_passes_consistent(self, good_metrics):
        result = check_consistency(good_metrics)
        assert result.passed

    def test_fails_inconsistent(self):
        result = check_consistency({"rolling_sharpe_std": 2.0})
        assert not result.passed

    def test_passes_when_missing(self):
        result = check_consistency({})
        assert result.passed  # not available = pass


# === Orchestrators ===


class TestRunPreChecks:
    def test_returns_all_checks(self, training_data, good_config):
        results = run_pre_checks(training_data, good_config)
        assert len(results) == 5
        assert all(isinstance(r, GuardrailResult) for r in results)

    def test_all_pass_for_good_data(self, training_data, good_config):
        results = run_pre_checks(training_data, good_config)
        assert all(r.passed for r in results)


class TestRunPostChecks:
    def test_returns_all_checks(self, good_returns, good_metrics, good_config):
        results = run_post_checks(good_returns, good_metrics, good_config)
        assert len(results) == 6
        assert all(isinstance(r, GuardrailResult) for r in results)


class TestHasBlockingFailure:
    def test_no_failures(self):
        results = [
            GuardrailResult(passed=True, check_name="a", message="ok", severity="block"),
            GuardrailResult(passed=True, check_name="b", message="ok", severity="warn"),
        ]
        assert not has_blocking_failure(results)

    def test_warn_failure_not_blocking(self):
        results = [
            GuardrailResult(passed=False, check_name="a", message="bad", severity="warn"),
        ]
        assert not has_blocking_failure(results)

    def test_block_failure_is_blocking(self):
        results = [
            GuardrailResult(passed=False, check_name="a", message="bad", severity="block"),
        ]
        assert has_blocking_failure(results)


class TestLogResults:
    def test_writes_jsonl(self, tmp_path):
        logfile = str(tmp_path / "guardrail_log.jsonl")
        results = [
            GuardrailResult(passed=True, check_name="test", message="ok", severity="info"),
        ]
        log_results(results, run_id="test_run", logfile=logfile)

        with open(logfile) as f:
            entry = json.loads(f.readline())
        assert entry["run_id"] == "test_run"
        assert entry["summary"]["total"] == 1
        assert entry["summary"]["passed"] == 1
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, tmp_path):
        logfile = str(tmp_path / "guardrail_log.jsonl")
        results = [GuardrailResult(passed=True, check_name="t", message="ok", severity="info")]
        log_results(results, run_id="run1", logfile=logfile)
        log_results(results, run_id="run2", logfile=logfile)

        with open(logfile) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["run_id"] == "run1"
        assert json.loads(lines[1])["run_id"] == "run2"


# === C-1: Fail closed for non-DatetimeIndex ===


class TestHoldoutBoundaryNonDatetimeIndex:
    def test_range_index_fails_closed(self):
        """RangeIndex input must fail with passed=False, severity=block."""
        data = pd.DataFrame({"x": range(100)})  # RangeIndex, not DatetimeIndex
        result = check_holdout_boundary(data, asset="btc")
        assert not result.passed
        assert result.severity == "block"
        assert "not DatetimeIndex" in result.message


# === C-2: Timezone normalization ===


class TestHoldoutBoundaryTimezones:
    def test_tz_naive_datetimeindex(self):
        """tz-naive DatetimeIndex should not raise TypeError."""
        dates = pd.date_range("2020-01-01", periods=100, freq="h")  # no tz
        data = pd.DataFrame({"x": range(100)}, index=dates)
        result = check_holdout_boundary(data, asset="btc")
        # Should pass (2020 data well before holdout)
        assert result.passed
        assert result.severity == "block"

    def test_tz_aware_utc(self):
        """tz-aware UTC DatetimeIndex should work normally."""
        dates = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
        data = pd.DataFrame({"x": range(100)}, index=dates)
        result = check_holdout_boundary(data, asset="btc")
        assert result.passed

    def test_non_utc_timezone(self):
        """Non-UTC timezone (US/Eastern) should not raise TypeError."""
        dates = pd.date_range("2020-01-01", periods=100, freq="h", tz="US/Eastern")
        data = pd.DataFrame({"x": range(100)}, index=dates)
        result = check_holdout_boundary(data, asset="btc")
        assert result.passed


# === m-4: Severity validation ===


class TestGuardrailResultSeverityValidation:
    def test_invalid_severity_raises_valueerror(self):
        """Invalid severity string must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            GuardrailResult(passed=True, check_name="test", message="ok", severity="critical")

    def test_valid_severities_accepted(self):
        """All valid severities should not raise."""
        for sev in ("block", "warn", "info"):
            result = GuardrailResult(passed=True, check_name="t", message="ok", severity=sev)
            assert result.severity == sev


# === M-2: Explicit n_trades ===


class TestMinimumTradesExplicit:
    def test_explicit_n_trades_override(self):
        """Explicit n_trades arg takes priority over heuristic."""
        returns = np.ones(100) * 0.01  # All same sign, heuristic would say 1
        config = {}
        result = check_minimum_trades(returns, config, n_trades=50)
        assert result.passed
        assert "explicit" in result.message

    def test_n_trades_from_config(self):
        """n_trades in config takes priority over heuristic."""
        returns = np.ones(100) * 0.01
        config = {"n_trades": 50}
        result = check_minimum_trades(returns, config)
        assert result.passed
        assert "config" in result.message

    def test_heuristic_fallback_label(self, good_returns, good_config):
        """Heuristic fallback includes 'heuristic' in message."""
        result = check_minimum_trades(good_returns, good_config)
        assert "heuristic" in result.message

    def test_explicit_n_trades_too_low_fails(self):
        """Explicit n_trades below min_trades should fail."""
        returns = np.random.randn(200) * 0.01
        config = {}
        result = check_minimum_trades(returns, config, n_trades=5, min_trades=30)
        assert not result.passed
        assert "explicit" in result.message


# === Severity structural test ===


class TestAllCheckSeverities:
    """All built-in check functions return severity in the valid set."""

    def test_pre_check_severities(self, training_data, good_config):
        results = run_pre_checks(training_data, good_config)
        for r in results:
            assert r.severity in _VALID_SEVERITIES, f"{r.check_name} has invalid severity {r.severity}"

    def test_post_check_severities(self, good_returns, good_metrics, good_config):
        results = run_post_checks(good_returns, good_metrics, good_config)
        for r in results:
            assert r.severity in _VALID_SEVERITIES, f"{r.check_name} has invalid severity {r.severity}"
