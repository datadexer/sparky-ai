"""Financial returns calculations.

Pure functions for computing returns, risk metrics, and performance statistics.
All functions are type-hinted with docstrings containing the formula.

These are the foundation calculations that everything else depends on.
Validated against hand-calculated expected values in tests/test_returns.py.
"""

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Compute simple (arithmetic) returns from a price series.

    Formula: r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1

    Args:
        prices: Series of prices with DatetimeIndex or sequential index.

    Returns:
        Series of simple returns. First value is NaN.
    """
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute logarithmic returns from a price series.

    Formula: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Log returns are additive across time and approximately equal to
    simple returns for small values.

    Args:
        prices: Series of prices (must be positive).

    Returns:
        Series of log returns. First value is NaN.
    """
    return np.log(prices / prices.shift(1))


def annualized_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Formula: Sharpe = (mean(r) - rf) / std(r) * sqrt(N)

    where N = periods_per_year (252 for daily, 52 for weekly, 12 for monthly).

    Args:
        returns: Series of period returns (not prices).
        risk_free_rate: Per-period risk-free rate (default 0).
        periods_per_year: Annualization factor (252 for daily).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero or insufficient data.
    """
    excess = returns.dropna() - risk_free_rate
    if len(excess) < 2:
        return 0.0
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown from a price series.

    Formula: MDD = max_over_t( (peak_t - P_t) / peak_t )

    where peak_t = max(P_0, P_1, ..., P_t) is the running maximum.

    Args:
        prices: Series of prices or equity values.

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.333 for 33.3% drawdown).
        Returns 0.0 if prices are empty or monotonically increasing.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0
    running_max = prices.cummax()
    drawdowns = (running_max - prices) / running_max
    result = drawdowns.max()
    return float(result) if not np.isnan(result) else 0.0


def realized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized realized volatility.

    Formula: vol = std(r) * sqrt(N)

    where N = periods_per_year.

    Args:
        returns: Series of period returns (not prices).
        periods_per_year: Annualization factor (252 for daily).

    Returns:
        Annualized volatility. Returns 0.0 if insufficient data.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))
