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
    def strategy_vs_benchmark(strat_returns: pd.Series, bench_returns: pd.Series) -> float:
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

    @staticmethod
    def block_bootstrap_monte_carlo(
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        n_simulations: int = 1000,
        block_size: int | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> dict:
        """Monte Carlo simulation using block bootstrap to preserve autocorrelation.

        Standard Monte Carlo (simple resampling) destroys time-series structure.
        Block bootstrap resamples contiguous blocks of returns, preserving
        short-term momentum and mean-reversion patterns common in crypto markets.

        This is critical for crypto where returns exhibit 2-5 day autocorrelation.
        Simple resampling underestimates variance → inflates win rates.

        Args:
            strategy_returns: Strategy return series
            market_returns: Market (Buy & Hold) return series
            n_simulations: Number of bootstrap samples (default 1000)
            block_size: Size of blocks to resample (default: sqrt(n) for optimal bias-variance)
            risk_free_rate: Annualized risk-free rate (decimal, e.g., 0.045 for 4.5%)
            periods_per_year: Trading periods per year (252 for daily, 365 for crypto)

        Returns:
            Dict with:
                - win_rate: Fraction of simulations where strategy beats market
                - wins/ties/losses: Count of each outcome
                - baseline_strategy_sharpe: Strategy Sharpe on original data
                - baseline_market_sharpe: Market Sharpe on original data
                - block_size: Block size used
                - n_simulations: Number of simulations run

        References:
            Politis & Romano (1994): "The Stationary Bootstrap"
            Ledoit & Wolf (2008): "Robust performance hypothesis testing with the Sharpe ratio"
        """
        from sparky.features.returns import annualized_sharpe

        if len(strategy_returns) == 0 or len(market_returns) == 0:
            raise ValueError("Returns series cannot be empty")

        # Convert risk-free rate from annual to per-period
        rf_per_period = risk_free_rate / periods_per_year

        # Auto-select block size using sqrt(n) rule if not specified
        # This balances bias (small blocks) vs variance (large blocks)
        if block_size is None:
            block_size = int(np.sqrt(len(strategy_returns)))
            block_size = max(5, min(block_size, 50))  # Clamp to reasonable range

        # Align series to same length
        min_len = min(len(strategy_returns), len(market_returns))
        strategy_array = strategy_returns.values[:min_len]
        market_array = market_returns.values[:min_len]
        n = len(strategy_array)

        # Number of blocks needed to reconstruct full series
        n_blocks = int(np.ceil(n / block_size))

        wins = 0
        ties = 0
        losses = 0

        for _ in range(n_simulations):
            # Resample blocks for strategy
            strategy_blocks = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n - block_size + 1)
                block = strategy_array[start_idx : start_idx + block_size]
                strategy_blocks.append(block)

            # Concatenate blocks and trim to original length
            strategy_resampled = np.concatenate(strategy_blocks)[:n]

            # Resample blocks for market (independent resampling)
            market_blocks = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n - block_size + 1)
                block = market_array[start_idx : start_idx + block_size]
                market_blocks.append(block)

            market_resampled = np.concatenate(market_blocks)[:n]

            # Compute Sharpe ratios on resampled data
            strategy_sharpe = annualized_sharpe(
                pd.Series(strategy_resampled), risk_free_rate=rf_per_period, periods_per_year=periods_per_year
            )
            market_sharpe = annualized_sharpe(
                pd.Series(market_resampled), risk_free_rate=rf_per_period, periods_per_year=periods_per_year
            )

            # Compare
            if strategy_sharpe > market_sharpe + 0.001:  # Small epsilon for numerical stability
                wins += 1
            elif abs(strategy_sharpe - market_sharpe) <= 0.001:
                ties += 1
            else:
                losses += 1

        # Compute baseline Sharpe ratios on original data
        baseline_strategy_sharpe = annualized_sharpe(
            pd.Series(strategy_array), risk_free_rate=rf_per_period, periods_per_year=periods_per_year
        )
        baseline_market_sharpe = annualized_sharpe(
            pd.Series(market_array), risk_free_rate=rf_per_period, periods_per_year=periods_per_year
        )

        return {
            "win_rate": wins / n_simulations,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "baseline_strategy_sharpe": float(baseline_strategy_sharpe),
            "baseline_market_sharpe": float(baseline_market_sharpe),
            "block_size": block_size,
            "n_simulations": n_simulations,
        }
