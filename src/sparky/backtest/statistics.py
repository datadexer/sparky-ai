"""Statistical analysis tools for backtesting."""

import numpy as np
import pandas as pd
from scipy import stats


class BacktestStatistics:
    """Statistical analysis tools for evaluating backtest results.

    Provides methods for calculating confidence intervals, significance tests,
    and strategy comparisons using resampling and hypothesis testing.
    """

    @staticmethod
    def sharpe_confidence_interval(
        returns: pd.Series,
        n_bootstrap: int = 10000,
        ci: float = 0.95,
        annualize: bool = True,
        random_state: int | None = None,
    ) -> tuple[float, float]:
        """Calculate confidence interval for annualized Sharpe ratio using bootstrap.

        Uses bootstrap resampling to estimate the distribution of the Sharpe ratio
        and compute confidence intervals. By default, the Sharpe is annualized
        (multiplied by sqrt(252)) to be comparable with annualized_sharpe().

        Args:
            returns: Series of daily returns.
            n_bootstrap: Number of bootstrap samples to generate.
            ci: Confidence interval level (e.g., 0.95 for 95% CI).
            annualize: If True (default), multiply Sharpe by sqrt(252).
            random_state: Optional seed for reproducibility.

        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval.
        """
        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        returns_array = returns.values
        n = len(returns_array)
        rng = np.random.RandomState(random_state)
        annualization_factor = np.sqrt(252) if annualize else 1.0

        # Generate bootstrap samples and calculate Sharpe ratio for each
        sharpe_ratios = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = rng.choice(returns_array, size=n, replace=True)

            # Calculate Sharpe ratio for this sample
            mean_return = np.mean(bootstrap_sample)
            std_return = np.std(bootstrap_sample, ddof=1)

            # Handle case where std is zero
            if std_return > 0:
                sharpe_ratios[i] = (mean_return / std_return) * annualization_factor
            else:
                sharpe_ratios[i] = 0.0

        # Calculate confidence interval
        alpha = 1 - ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(sharpe_ratios, lower_percentile)
        upper_bound = np.percentile(sharpe_ratios, upper_percentile)

        return (lower_bound, upper_bound)

    @staticmethod
    def sharpe_significance(returns: pd.Series) -> float:
        """Test if Sharpe ratio is significantly different from zero.

        Uses a one-sample t-test to test the null hypothesis that the mean
        return is zero (i.e., Sharpe ratio is zero).

        Args:
            returns: Series of returns

        Returns:
            p-value for the test (H0: mean return = 0)
        """
        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        # One-sample t-test against zero
        t_statistic, p_value = stats.ttest_1samp(returns, 0.0)

        return p_value

    @staticmethod
    def strategy_vs_benchmark(
        strat_returns: pd.Series,
        bench_returns: pd.Series
    ) -> float:
        """Test if strategy returns are significantly different from benchmark.

        Uses a paired t-test to compare strategy returns against benchmark returns.
        Tests the null hypothesis that there is no difference between the two.

        Args:
            strat_returns: Series of strategy returns
            bench_returns: Series of benchmark returns (must have same length)

        Returns:
            p-value for the test (H0: no difference between strategy and benchmark)
        """
        if len(strat_returns) != len(bench_returns):
            raise ValueError("Strategy and benchmark returns must have the same length")

        if len(strat_returns) == 0:
            raise ValueError("Returns series cannot be empty")

        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(strat_returns, bench_returns)

        return p_value
