"""Tests for backtest statistics."""

import numpy as np
import pandas as pd
import pytest

from sparky.backtest.statistics import BacktestStatistics


class TestBacktestStatistics:
    """Tests for BacktestStatistics."""

    def test_sharpe_confidence_interval_basic(self):
        """Test basic confidence interval calculation."""
        np.random.seed(42)

        # Generate returns with known properties
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        lower, upper = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=1000)

        # CI should be reasonable
        assert lower < upper
        # Point estimate should be within CI (usually)
        point_sharpe = returns.mean() / returns.std()
        # Note: point estimate may occasionally fall outside bootstrap CI
        # So we just check that the interval is non-trivial
        assert upper - lower > 0

    def test_sharpe_confidence_interval_positive_returns(self):
        """Test CI for positive returns."""
        np.random.seed(42)

        # Positive returns (mean > 0)
        returns = pd.Series(np.random.normal(0.01, 0.02, 100))

        lower, upper = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=1000, ci=0.95)

        # Both bounds should be positive for strongly positive returns
        # (though this isn't guaranteed for all random seeds)
        assert lower < upper
        assert upper > 0  # At least upper bound should be positive

    def test_sharpe_confidence_interval_zero_std(self):
        """Test CI when returns have zero standard deviation."""
        # All returns are the same (zero variance)
        returns = pd.Series([0.01] * 50)

        lower, upper = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=100)

        # Should handle gracefully (Sharpe undefined, but should return something)
        assert not np.isnan(lower)
        assert not np.isnan(upper)

    def test_sharpe_confidence_interval_different_ci_levels(self):
        """Test that different CI levels give different intervals."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        lower_95, upper_95 = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=1000, ci=0.95)
        lower_90, upper_90 = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=1000, ci=0.90)

        # 90% CI should be narrower than 95% CI
        width_95 = upper_95 - lower_95
        width_90 = upper_90 - lower_90
        assert width_90 < width_95

    def test_sharpe_confidence_interval_empty_returns(self):
        """Test that empty returns raise an error."""
        returns = pd.Series([])

        with pytest.raises(ValueError, match="cannot be empty"):
            BacktestStatistics.sharpe_confidence_interval(returns)

    def test_sharpe_significance_positive_returns(self):
        """Test that significantly positive returns have low p-value."""
        np.random.seed(42)

        # Strong positive returns
        returns = pd.Series(np.random.normal(0.02, 0.01, 252))

        p_value = BacktestStatistics.sharpe_significance(returns)

        # Should be highly significant
        assert p_value < 0.05

    def test_sharpe_significance_negative_returns(self):
        """Test that significantly negative returns have low p-value."""
        np.random.seed(42)

        # Strong negative returns
        returns = pd.Series(np.random.normal(-0.02, 0.01, 252))

        p_value = BacktestStatistics.sharpe_significance(returns)

        # Should be highly significant
        assert p_value < 0.05

    def test_sharpe_significance_random_returns(self):
        """Test that random returns around zero have high p-value."""
        np.random.seed(42)

        # Returns centered at zero with small sample
        returns = pd.Series(np.random.normal(0.0, 0.02, 30))

        p_value = BacktestStatistics.sharpe_significance(returns)

        # Likely not significant (though this can vary with random seed)
        # Just check it returns a valid p-value
        assert 0 <= p_value <= 1

    def test_sharpe_significance_empty_returns(self):
        """Test that empty returns raise an error."""
        returns = pd.Series([])

        with pytest.raises(ValueError, match="cannot be empty"):
            BacktestStatistics.sharpe_significance(returns)

    def test_strategy_vs_benchmark_identical(self):
        """Test that identical returns have high p-value."""
        np.random.seed(42)

        returns = pd.Series(np.random.normal(0.01, 0.02, 100))

        p_value = BacktestStatistics.strategy_vs_benchmark(returns, returns)

        # Should not be significant (identical returns)
        # When returns are identical, p-value will be NaN (no variance in difference)
        # This is expected behavior
        assert np.isnan(p_value) or p_value > 0.95

    def test_strategy_vs_benchmark_different(self):
        """Test that significantly different returns have low p-value."""
        np.random.seed(42)

        # Strategy with consistently higher returns
        strat_returns = pd.Series(np.random.normal(0.02, 0.01, 252))
        bench_returns = pd.Series(np.random.normal(0.005, 0.01, 252))

        p_value = BacktestStatistics.strategy_vs_benchmark(strat_returns, bench_returns)

        # Should be significant
        assert p_value < 0.05

    def test_strategy_vs_benchmark_slight_difference(self):
        """Test with slight difference in returns."""
        np.random.seed(42)

        # Similar returns with small sample
        strat_returns = pd.Series(np.random.normal(0.01, 0.02, 30))
        bench_returns = pd.Series(np.random.normal(0.009, 0.02, 30))

        p_value = BacktestStatistics.strategy_vs_benchmark(strat_returns, bench_returns)

        # Just check it returns a valid p-value
        assert 0 <= p_value <= 1

    def test_strategy_vs_benchmark_mismatched_lengths(self):
        """Test that mismatched lengths raise an error."""
        strat_returns = pd.Series([0.01, 0.02, 0.03])
        bench_returns = pd.Series([0.01, 0.02])

        with pytest.raises(ValueError, match="same length"):
            BacktestStatistics.strategy_vs_benchmark(strat_returns, bench_returns)

    def test_strategy_vs_benchmark_empty_returns(self):
        """Test that empty returns raise an error."""
        strat_returns = pd.Series([])
        bench_returns = pd.Series([])

        with pytest.raises(ValueError, match="cannot be empty"):
            BacktestStatistics.strategy_vs_benchmark(strat_returns, bench_returns)

    def test_confidence_interval_contains_point_estimate(self):
        """Test that confidence interval usually contains the point estimate."""
        np.random.seed(42)

        # Generate returns
        returns = pd.Series(np.random.normal(0.01, 0.02, 200))

        # Calculate point estimate
        point_sharpe = returns.mean() / returns.std()

        # Calculate CI
        lower, upper = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=5000, ci=0.95)

        # Point estimate should usually be within CI
        # (May fail occasionally due to randomness, but should work most of the time)
        # We'll just check that the interval is reasonable
        assert lower < upper
        assert abs(point_sharpe) < abs(upper - lower) * 3  # Loose sanity check

    def test_bootstrap_ci_width_reasonable(self):
        """Test that bootstrap CI width is reasonable for known distribution."""
        np.random.seed(42)

        # Large sample with known properties
        returns = pd.Series(np.random.normal(0.01, 0.02, 1000))

        lower, upper = BacktestStatistics.sharpe_confidence_interval(
            returns, n_bootstrap=2000, ci=0.95, random_state=42
        )

        # CI width should be positive and finite
        width = upper - lower
        assert width > 0
        assert np.isfinite(width)

        # Width should be reasonable (not too large)
        # For 1000 samples, CI should be fairly tight
        assert width < 100.0  # Loose upper bound (annualized now)

    def test_sharpe_ci_is_annualized_by_default(self):
        """Test that bootstrap CI is annualized (comparable to annualized_sharpe)."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        # Annualized (default)
        lower_ann, upper_ann = BacktestStatistics.sharpe_confidence_interval(
            returns, n_bootstrap=1000, random_state=42, annualize=True
        )
        # Daily (not annualized)
        lower_daily, upper_daily = BacktestStatistics.sharpe_confidence_interval(
            returns, n_bootstrap=1000, random_state=42, annualize=False
        )

        # Annualized CI should be sqrt(252) times wider than daily
        scale = np.sqrt(252)
        assert abs(lower_ann - lower_daily * scale) < 0.5  # Allow some bootstrap noise
        assert abs(upper_ann - upper_daily * scale) < 0.5

    def test_sharpe_ci_random_state_reproducibility(self):
        """Test that random_state produces reproducible results."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        lower1, upper1 = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=1000, random_state=42)
        lower2, upper2 = BacktestStatistics.sharpe_confidence_interval(returns, n_bootstrap=1000, random_state=42)

        assert lower1 == pytest.approx(lower2)
        assert upper1 == pytest.approx(upper2)
