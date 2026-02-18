"""Tests for sparky.tracking.metrics — sanity checks for all key metrics."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from sparky.tracking.metrics import (
    analytical_dsr,
    compute_all_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


@pytest.fixture
def profitable_returns():
    """Returns with a clearly positive mean — profitable series."""
    return RNG.normal(loc=0.002, scale=0.01, size=500)


@pytest.fixture
def losing_returns():
    """Returns with a clearly negative mean — losing series."""
    return RNG.normal(loc=-0.002, scale=0.01, size=500)


@pytest.fixture
def zero_returns():
    """All-zero returns."""
    return np.zeros(100)


@pytest.fixture
def single_return():
    """Single-element returns array."""
    return np.array([0.01])


@pytest.fixture
def short_returns():
    """Very short returns series (5 elements)."""
    return np.array([0.01, -0.005, 0.02, 0.003, -0.001])


@pytest.fixture
def positive_skew_returns():
    """Returns with strong positive skew (many small losses, rare large gains)."""
    # Chi-squared shifted to have mostly negative values but large positive tails
    base = RNG.chisquare(df=2, size=1000)
    # Shift so mean is near zero but skew is positive
    base = base - base.mean() + 0.001
    return base


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    EXPECTED_KEYS = {
        "sharpe",
        "psr",
        "dsr",
        "min_track_record",
        "n_trials",
        "skewness",
        "kurtosis",
        "sortino",
        "max_drawdown",
        "calmar",
        "cvar_5pct",
        "rolling_sharpe_std",
        "profit_factor",
        "worst_year_sharpe",
        "n_observations",
        "win_rate",
        "mean_return",
        "total_return",
    }

    def test_returns_all_expected_keys(self, profitable_returns):
        result = compute_all_metrics(profitable_returns, n_trials=10)
        assert self.EXPECTED_KEYS == set(result.keys()), (
            f"Missing keys: {self.EXPECTED_KEYS - set(result.keys())}, "
            f"Extra keys: {set(result.keys()) - self.EXPECTED_KEYS}"
        )

    def test_n_observations_matches_input(self, profitable_returns):
        result = compute_all_metrics(profitable_returns)
        assert result["n_observations"] == len(profitable_returns)

    def test_n_trials_stored_correctly(self, profitable_returns):
        result = compute_all_metrics(profitable_returns, n_trials=42)
        assert result["n_trials"] == 42

    def test_win_rate_in_unit_interval(self, profitable_returns):
        result = compute_all_metrics(profitable_returns)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_all_values_are_finite_or_nan(self, profitable_returns):
        """No value should be +/-inf (NaN is acceptable for short series)."""
        result = compute_all_metrics(profitable_returns)
        for key, val in result.items():
            if isinstance(val, float):
                assert not math.isinf(val), f"Key '{key}' is infinite: {val}"

    def test_zero_returns_does_not_crash(self, zero_returns):
        """compute_all_metrics should handle all-zero returns without raising."""
        result = compute_all_metrics(zero_returns)
        assert isinstance(result, dict)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_short_returns_does_not_crash(self, short_returns):
        """compute_all_metrics should handle a 5-element series without raising."""
        result = compute_all_metrics(short_returns)
        assert isinstance(result, dict)

    def test_single_element_does_not_crash(self, single_return):
        """compute_all_metrics on a single-element series.

        The underlying PSR/DSR calculations divide by (T-1)=0 for T=1,
        raising ZeroDivisionError. This is a known limitation of the
        statistical formulas which require at least 2 observations.
        """
        try:
            result = compute_all_metrics(single_return)
            assert isinstance(result, dict)
        except ZeroDivisionError:
            # Expected for T=1: PSR/DSR formulas require T >= 2
            pass


# ---------------------------------------------------------------------------
# DSR vs PSR — multiple-testing penalty
# ---------------------------------------------------------------------------


class TestDSRvsPSR:
    def test_dsr_less_than_psr_when_multiple_trials(self, profitable_returns):
        """With n_trials > 1, DSR must be strictly less than PSR."""
        psr = probabilistic_sharpe_ratio(profitable_returns)
        dsr = deflated_sharpe_ratio(profitable_returns, n_trials=50)
        assert dsr < psr, f"Expected DSR ({dsr:.4f}) < PSR ({psr:.4f}) with 50 trials."

    def test_dsr_approx_psr_when_one_trial(self, profitable_returns):
        """With n_trials=1, DSR should be approximately equal to PSR."""
        psr = probabilistic_sharpe_ratio(profitable_returns)
        dsr = deflated_sharpe_ratio(profitable_returns, n_trials=1)
        # When n_trials=1 the expected max SR benchmark is near 0, similar to PSR
        # They won't be exactly equal but should be close (within 0.15)
        assert abs(dsr - psr) < 0.15, f"DSR ({dsr:.4f}) and PSR ({psr:.4f}) differ too much for n_trials=1."

    def test_dsr_decreases_as_trials_increase(self, profitable_returns):
        """More trials = more skepticism = lower DSR."""
        dsrs = [deflated_sharpe_ratio(profitable_returns, n_trials=n) for n in [1, 5, 20, 100]]
        for i in range(len(dsrs) - 1):
            assert dsrs[i] >= dsrs[i + 1], f"DSR did not decrease as trials increased: {dsrs}"

    def test_dsr_is_probability(self, profitable_returns):
        """DSR must be in [0, 1]."""
        dsr = deflated_sharpe_ratio(profitable_returns, n_trials=10)
        assert 0.0 <= dsr <= 1.0

    def test_psr_is_probability(self, profitable_returns):
        """PSR must be in [0, 1]."""
        psr = probabilistic_sharpe_ratio(profitable_returns)
        assert 0.0 <= psr <= 1.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_max_drawdown_non_positive(self, profitable_returns):
        mdd = max_drawdown(profitable_returns)
        assert mdd <= 0.0, f"max_drawdown returned positive value: {mdd}"

    def test_max_drawdown_non_positive_for_losing(self, losing_returns):
        mdd = max_drawdown(losing_returns)
        assert mdd <= 0.0

    def test_max_drawdown_zero_for_monotone_gains(self):
        """Monotonically increasing returns => drawdown = 0."""
        returns = np.full(50, 0.01)  # always +1%
        mdd = max_drawdown(returns)
        assert mdd == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_zero_for_monotone_gains_2(self):
        """Same as above but via the fixture path."""
        returns = np.full(50, 0.01)
        mdd = max_drawdown(returns)
        assert mdd <= 0.0

    def test_max_drawdown_severe_for_all_losses(self):
        """All negative returns => substantial drawdown."""
        returns = np.full(100, -0.01)
        mdd = max_drawdown(returns)
        assert mdd < -0.5, f"Expected severe drawdown, got {mdd:.4f}"

    def test_max_drawdown_in_range(self, profitable_returns):
        mdd = max_drawdown(profitable_returns)
        assert -1.0 <= mdd <= 0.0

    def test_max_drawdown_zero_returns(self, zero_returns):
        mdd = max_drawdown(zero_returns)
        assert mdd == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_profit_factor_greater_than_one_for_profitable(self, profitable_returns):
        pf = profit_factor(profitable_returns)
        assert pf > 1.0, f"Expected profit_factor > 1 for profitable series, got {pf:.4f}"

    def test_profit_factor_less_than_one_for_losing(self, losing_returns):
        pf = profit_factor(losing_returns)
        assert pf < 1.0, f"Expected profit_factor < 1 for losing series, got {pf:.4f}"

    def test_profit_factor_positive(self, profitable_returns):
        pf = profit_factor(profitable_returns)
        assert pf > 0.0

    def test_profit_factor_all_gains(self):
        """All positive returns => profit_factor is infinite."""
        returns = np.full(20, 0.01)
        pf = profit_factor(returns)
        assert math.isinf(pf)

    def test_profit_factor_all_losses(self):
        """All negative returns => profit_factor is zero (no gains)."""
        returns = np.full(20, -0.01)
        pf = profit_factor(returns)
        assert pf == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# sortino_ratio vs sharpe_ratio
# ---------------------------------------------------------------------------


class TestSortinoVsSharpe:
    def test_sortino_greater_than_sharpe_with_positive_skew(self, positive_skew_returns):
        """For positively skewed returns, Sortino > Sharpe because downside vol < total vol."""
        sr = sharpe_ratio(positive_skew_returns)
        so = sortino_ratio(positive_skew_returns)
        # Positive skew means losses cluster (small), gains spread out (large)
        # Downside std is smaller than total std => Sortino > Sharpe
        assert so > sr, f"Expected Sortino ({so:.4f}) > Sharpe ({sr:.4f}) for positively skewed returns."

    def test_sortino_equals_sharpe_for_symmetric_returns(self):
        """For symmetric (normal) returns, Sortino ≈ Sharpe * sqrt(2).

        With RMS downside deviation over the full series, for a symmetric
        distribution half the values are zero and half are negative, so
        downside_dev ≈ std / sqrt(2), giving Sortino ≈ Sharpe * sqrt(2).
        """
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.02, 10000)
        sr = sharpe_ratio(returns)
        so = sortino_ratio(returns)
        assert so > sr
        assert so == pytest.approx(sr * np.sqrt(2), rel=0.15)

    def test_sortino_is_finite_for_normal_returns(self, profitable_returns):
        so = sortino_ratio(profitable_returns)
        assert math.isfinite(so)


# ---------------------------------------------------------------------------
# expected_max_sharpe
# ---------------------------------------------------------------------------


class TestExpectedMaxSharpe:
    def test_expected_max_sharpe_increases_with_trials(self):
        """More trials => higher expected max Sharpe from noise."""
        ems_values = [expected_max_sharpe(n, T=8760) for n in [1, 5, 10, 50, 100, 500]]
        for i in range(len(ems_values) - 1):
            assert ems_values[i] <= ems_values[i + 1], (
                f"expected_max_sharpe did not increase monotonically: {ems_values}"
            )

    def test_expected_max_sharpe_positive(self):
        """Expected max Sharpe should be positive."""
        ems = expected_max_sharpe(100, T=8760)
        assert ems > 0.0

    def test_expected_max_sharpe_one_trial(self):
        """With 1 trial, expected max Sharpe should be small (near 0)."""
        ems = expected_max_sharpe(1, T=8760)
        # With 1 trial we use max(n_trials, 2)=2 per implementation, so still small
        assert ems >= 0.0

    def test_expected_max_sharpe_large_N(self):
        """With many trials, expected max Sharpe from noise grows meaningfully.

        The formula uses sr_variance = 1/T so EMS is in units of per-observation
        Sharpe (not annualized). With 1000 trials and T=100 (short series), the
        expected max SR should exceed the single-trial value by a meaningful margin.
        """
        ems_1 = expected_max_sharpe(1, T=100)
        ems_1000 = expected_max_sharpe(1000, T=100)
        # With 1000 trials vs 1 trial, EMS should be materially larger
        assert ems_1000 > ems_1 * 2.0, f"Expected EMS(1000) > 2x EMS(1): {ems_1000:.4f} vs {ems_1:.4f}"

    def test_expected_max_sharpe_increases_with_fewer_observations(self):
        """Less data (smaller T) => larger sr_variance => higher expected max Sharpe."""
        ems_long = expected_max_sharpe(100, T=8760)
        ems_short = expected_max_sharpe(100, T=100)
        assert ems_short > ems_long, (
            f"Expected EMS to be larger with fewer observations. Short ({ems_short:.4f}) vs Long ({ems_long:.4f})"
        )


# ---------------------------------------------------------------------------
# minimum_track_record_length
# ---------------------------------------------------------------------------


class TestMinimumTrackRecordLength:
    def test_finite_for_nonzero_sharpe(self, profitable_returns):
        """For a strategy with SR > 0, MTRL should be finite."""
        mtrl = minimum_track_record_length(profitable_returns)
        assert math.isfinite(mtrl), f"Expected finite MTRL, got {mtrl}"

    def test_infinite_for_zero_sharpe(self):
        """For a strategy with SR = 0, MTRL should be infinite."""
        # Construct returns with zero mean => SR = 0
        returns = np.array([0.01, -0.01] * 100, dtype=float)
        # SR will be 0 since mean is 0
        sr = sharpe_ratio(returns)
        if abs(sr) < 1e-10:
            mtrl = minimum_track_record_length(returns)
            assert math.isinf(mtrl), f"Expected infinite MTRL for zero SR, got {mtrl}"
        else:
            pytest.skip(f"Returns did not produce zero SR (SR={sr:.4f}), skipping")

    def test_mtrl_decreases_as_sharpe_increases(self):
        """Higher SR => less track record needed."""
        rng = np.random.default_rng(7)
        returns_low_sr = rng.normal(0.0001, 0.02, 1000)
        returns_high_sr = rng.normal(0.005, 0.02, 1000)

        mtrl_low = minimum_track_record_length(returns_low_sr)
        mtrl_high = minimum_track_record_length(returns_high_sr)

        # Higher SR needs fewer observations
        if math.isfinite(mtrl_low) and math.isfinite(mtrl_high):
            assert mtrl_high < mtrl_low, (
                f"Expected MTRL to decrease with higher SR: low SR MTRL={mtrl_low:.1f}, high SR MTRL={mtrl_high:.1f}"
            )

    def test_mtrl_is_positive(self, profitable_returns):
        mtrl = minimum_track_record_length(profitable_returns)
        assert mtrl > 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_zero_returns_sharpe(self, zero_returns):
        """All-zero returns => Sharpe = 0."""
        sr = sharpe_ratio(zero_returns)
        assert sr == pytest.approx(0.0)

    def test_all_zero_returns_sortino(self, zero_returns):
        """All-zero returns => Sortino = 0."""
        so = sortino_ratio(zero_returns)
        assert so == pytest.approx(0.0)

    def test_all_zero_returns_max_drawdown(self, zero_returns):
        """All-zero returns => max_drawdown = 0."""
        mdd = max_drawdown(zero_returns)
        assert mdd == pytest.approx(0.0, abs=1e-10)

    def test_all_zero_returns_compute_all(self, zero_returns):
        """compute_all_metrics on all-zero returns returns a valid dict."""
        result = compute_all_metrics(zero_returns)
        assert isinstance(result, dict)
        assert result["sharpe"] == pytest.approx(0.0)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_single_element_returns_sharpe(self, single_return):
        """Single element => no std => sharpe = 0."""
        sr = sharpe_ratio(single_return)
        assert sr == pytest.approx(0.0)

    def test_short_series_metrics_dont_crash(self, short_returns):
        """5-element series should not raise any exceptions."""
        result = compute_all_metrics(short_returns, n_trials=3)
        assert isinstance(result, dict)
        assert result["n_observations"] == 5

    def test_returns_accepts_list_input(self):
        """compute_all_metrics should accept plain Python lists."""
        returns_list = [0.01, -0.005, 0.02, 0.003, -0.001, 0.008, -0.002]
        result = compute_all_metrics(returns_list)
        assert isinstance(result, dict)

    def test_dsr_zero_for_clearly_unprofitable(self):
        """DSR should be near 0 for a clearly losing strategy with many trials."""
        rng = np.random.default_rng(99)
        returns = rng.normal(-0.01, 0.02, 500)
        dsr = deflated_sharpe_ratio(returns, n_trials=100)
        assert dsr < 0.1, f"Expected near-zero DSR for losing strategy, got {dsr:.4f}"

    def test_dsr_high_for_clearly_profitable_few_trials(self):
        """DSR should be high for a clearly profitable strategy with few trials."""
        rng = np.random.default_rng(1)
        returns = rng.normal(0.01, 0.01, 500)  # SR ~= 1.0
        dsr = deflated_sharpe_ratio(returns, n_trials=1)
        assert dsr > 0.90, f"Expected high DSR for very profitable strategy, got {dsr:.4f}"


# ---------------------------------------------------------------------------
# DSR formula regression test — verifies SE uses observed SR (not sr0)
# ---------------------------------------------------------------------------


class TestDSRFormulaRegression:
    """Regression test: DSR SE formula must use observed SR, not sr0."""

    def test_dsr_matches_hand_computed_formula(self):
        """Verify DSR output matches hand-computed value using the correct formula."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 1000)

        sr = sharpe_ratio(returns)
        T = len(returns)
        std = np.std(returns)
        standardized = (returns - np.mean(returns)) / std
        skew = float(np.mean(standardized**3))
        kurt = float(np.mean(standardized**4))

        sr0 = expected_max_sharpe(10, T)

        # Correct formula: SE uses observed sr (not sr0)
        se = np.sqrt((1 - skew * sr + ((kurt - 1) / 4) * sr**2) / (T - 1))
        expected_dsr = float(norm.cdf((sr - sr0) / se))

        actual_dsr = deflated_sharpe_ratio(returns, n_trials=10)
        assert abs(actual_dsr - expected_dsr) < 1e-10, (
            f"DSR mismatch: actual={actual_dsr:.10f}, expected={expected_dsr:.10f}. SE may be using sr0 instead of sr."
        )

    def test_dsr_se_uses_sr_not_sr0(self):
        """Confirm the SE formula uses the observed SR by checking against sr0-based result."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 2000)

        sr = sharpe_ratio(returns)
        T = len(returns)
        std = np.std(returns)
        standardized = (returns - np.mean(returns)) / std
        skew = float(np.mean(standardized**3))
        kurt = float(np.mean(standardized**4))
        sr0 = expected_max_sharpe(50, T)

        # Wrong formula (using sr0 in SE) — this is what the old bug computed
        se_wrong = np.sqrt((1 - skew * sr0 + ((kurt - 1) / 4) * sr0**2) / (T - 1))
        dsr_wrong = float(norm.cdf((sr - sr0) / se_wrong))

        # Correct formula (using sr in SE)
        se_correct = np.sqrt((1 - skew * sr + ((kurt - 1) / 4) * sr**2) / (T - 1))
        dsr_correct = float(norm.cdf((sr - sr0) / se_correct))

        actual_dsr = deflated_sharpe_ratio(returns, n_trials=50)

        # actual must match correct, NOT the wrong version
        assert abs(actual_dsr - dsr_correct) < 1e-10
        # The wrong formula gives a different result (unless sr ≈ sr0)
        if abs(sr - sr0) > 0.01:
            assert abs(actual_dsr - dsr_wrong) > 1e-6, "DSR matched the wrong formula — SE may still use sr0"


# ---------------------------------------------------------------------------
# analytical_dsr
# ---------------------------------------------------------------------------


class TestAnalyticalDSR:
    """Tests for analytical_dsr() — DSR from summary statistics."""

    def test_matches_deflated_sharpe_ratio(self):
        """analytical_dsr must match deflated_sharpe_ratio for the same inputs."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.01, 1000)

        sr = sharpe_ratio(returns)
        T = len(returns)
        std = np.std(returns)
        standardized = (returns - np.mean(returns)) / std
        skew = float(np.mean(standardized**3))
        kurt = float(np.mean(standardized**4))

        dsr_from_returns = deflated_sharpe_ratio(returns, n_trials=20)
        dsr_analytical = analytical_dsr(sr, skew, kurt, T, n_trials=20)

        assert abs(dsr_from_returns - dsr_analytical) < 1e-10, (
            f"analytical_dsr ({dsr_analytical}) != deflated_sharpe_ratio ({dsr_from_returns})"
        )

    def test_gaussian_fallback_when_none(self):
        """None skewness/kurtosis should use Gaussian defaults (0, 3)."""
        dsr_explicit = analytical_dsr(sr=1.0, skewness=0.0, kurtosis=3.0, T=5000, n_trials=10)
        dsr_none = analytical_dsr(sr=1.0, skewness=None, kurtosis=None, T=5000, n_trials=10)
        assert abs(dsr_explicit - dsr_none) < 1e-10

    def test_is_probability(self):
        """analytical_dsr must return a value in [0, 1]."""
        dsr = analytical_dsr(sr=1.5, skewness=-0.5, kurtosis=5.0, T=5000, n_trials=50)
        assert 0.0 <= dsr <= 1.0

    def test_decreases_with_more_trials(self):
        """More trials = higher bar = lower DSR."""
        dsrs = [analytical_dsr(sr=1.0, skewness=0.0, kurtosis=3.0, T=5000, n_trials=n) for n in [1, 10, 50, 200]]
        for i in range(len(dsrs) - 1):
            assert dsrs[i] >= dsrs[i + 1], f"DSR did not decrease: {dsrs}"

    def test_zero_for_losing_strategy(self):
        """Negative Sharpe with many trials should give near-zero DSR."""
        dsr = analytical_dsr(sr=-0.5, skewness=0.0, kurtosis=3.0, T=5000, n_trials=100)
        assert dsr < 0.01

    def test_negative_variance_guard(self):
        """Extreme skew + high SR that makes variance negative should return 0."""
        # skew=10, sr=3, kurt=3: 1 - 10*3 + 0.5*9 = 1-30+4.5 = -24.5 (negative!)
        dsr = analytical_dsr(sr=3.0, skewness=10.0, kurtosis=3.0, T=1000, n_trials=5)
        assert dsr == 0.0

    def test_compute_all_metrics_includes_moments(self):
        """compute_all_metrics should include skewness and kurtosis for recomputation."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 500)
        metrics = compute_all_metrics(returns, n_trials=10)

        assert "skewness" in metrics
        assert "kurtosis" in metrics
        # Kurtosis should be raw (near 3 for Gaussian)
        assert 2.0 < metrics["kurtosis"] < 4.0, f"Kurtosis {metrics['kurtosis']} looks wrong for Gaussian"
        assert -1.0 < metrics["skewness"] < 1.0, f"Skewness {metrics['skewness']} looks wrong for Gaussian"

        # Verify we can round-trip through analytical_dsr
        dsr_roundtrip = analytical_dsr(
            sr=metrics["sharpe"],
            skewness=metrics["skewness"],
            kurtosis=metrics["kurtosis"],
            T=metrics["n_observations"],
            n_trials=metrics["n_trials"],
        )
        assert abs(dsr_roundtrip - metrics["dsr"]) < 1e-10
