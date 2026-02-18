"""Strategy evaluation metrics beyond Sharpe ratio."""

import numpy as np
from scipy.stats import norm


def sharpe_ratio(returns, risk_free=0.0):
    excess = returns - risk_free
    return float(np.mean(excess) / np.std(excess, ddof=1)) if np.std(excess) > 0 else 0.0


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0):
    """Probability that true SR exceeds sr_benchmark, accounting for
    non-normal returns (skew, kurtosis) and sample length.
    Bailey & Lopez de Prado (2012)."""
    sr = sharpe_ratio(returns)
    T = len(returns)
    skew = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
    kurt = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4))

    se = np.sqrt((1 - skew * sr + ((kurt - 1) / 4) * sr**2) / (T - 1))
    if se == 0:
        return 0.0
    return float(norm.cdf((sr - sr_benchmark) / se))


def deflated_sharpe_ratio(returns, n_trials, sr_variance=None):
    """Sharpe ratio corrected for multiple testing.
    Bailey & Lopez de Prado (2014).

    This is THE key metric. A DSR > 0.95 means <5% chance the result is a fluke
    after accounting for all n_trials strategies tested.
    """
    sr = sharpe_ratio(returns)
    T = len(returns)
    skew = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
    kurt = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4))

    if sr_variance is None:
        sr_variance = 1.0 / T  # default estimate

    # Expected maximum SR under null (False Strategy Theorem)
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    e = np.exp(1)
    sr0 = np.sqrt(sr_variance) * (
        (1 - gamma) * norm.ppf(1 - 1 / max(n_trials, 2))
        + gamma * norm.ppf(1 - 1 / (max(n_trials, 2) * e))
    )

    se = np.sqrt((1 - skew * sr0 + ((kurt - 1) / 4) * sr0**2) / (T - 1))
    if se == 0:
        return 0.0
    return float(norm.cdf((sr - sr0) / se))


def expected_max_sharpe(n_trials, T, sr_variance=None):
    """Expected maximum Sharpe from noise alone given n_trials.
    False Strategy Theorem — Bailey & Lopez de Prado (2014)."""
    if sr_variance is None:
        sr_variance = 1.0 / T
    gamma = 0.5772156649015329
    e = np.exp(1)
    return float(
        np.sqrt(sr_variance)
        * (
            (1 - gamma) * norm.ppf(1 - 1 / max(n_trials, 2))
            + gamma * norm.ppf(1 - 1 / (max(n_trials, 2) * e))
        )
    )


def minimum_track_record_length(returns, sr_benchmark=0.0, confidence=0.95):
    """Minimum number of observations needed to conclude SR > benchmark
    at given confidence level."""
    sr = sharpe_ratio(returns)
    skew = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
    kurt = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4))
    z = norm.ppf(confidence)

    if sr - sr_benchmark == 0:
        return float("inf")

    return float(
        (1 - skew * sr + ((kurt - 1) / 4) * sr**2) * (z / (sr - sr_benchmark)) ** 2
    )


# === RISK METRICS ===


def sortino_ratio(returns, risk_free=0.0):
    """Like Sharpe but only penalizes downside volatility."""
    excess = returns - risk_free
    downside = excess[excess < 0]
    downside_std = (
        np.std(downside, ddof=1) if len(downside) > 1 else np.std(excess, ddof=1)
    )
    return float(np.mean(excess) / downside_std) if downside_std > 0 else 0.0


def max_drawdown(returns):
    """Maximum peak-to-trough decline."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def calmar_ratio(returns, periods_per_year=252):
    """Annualized return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ann_return = np.mean(returns) * periods_per_year
    return float(ann_return / abs(mdd))


def conditional_var(returns, alpha=0.05):
    """Expected loss in the worst alpha% of periods (CVaR/Expected Shortfall)."""
    cutoff = np.percentile(returns, alpha * 100)
    return float(np.mean(returns[returns <= cutoff]))


# === CONSISTENCY METRICS ===


def rolling_sharpe_std(returns, window=126):
    """Std of rolling Sharpe -- lower = more consistent."""
    if len(returns) < window * 2:
        return float("nan")
    rolling_srs = []
    for i in range(len(returns) - window):
        chunk = returns[i : i + window]
        rolling_srs.append(
            np.mean(chunk) / np.std(chunk, ddof=1) if np.std(chunk) > 0 else 0
        )
    return float(np.std(rolling_srs))


def profit_factor(returns):
    """Gross profits / gross losses. >1 = profitable, >2 = strong."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


def worst_year_sharpe(returns, periods_per_year=252):
    """Sharpe of the worst calendar year."""
    n_years = len(returns) // periods_per_year
    if n_years < 2:
        return sharpe_ratio(returns)
    yearly_sharpes = []
    for i in range(n_years):
        chunk = returns[i * periods_per_year : (i + 1) * periods_per_year]
        yearly_sharpes.append(sharpe_ratio(chunk))
    return float(min(yearly_sharpes))


# === COMBINED ===


def compute_all_metrics(returns, n_trials=1, risk_free=0.0, periods_per_year=252):
    """Compute comprehensive strategy metrics from a returns series.

    Args:
        returns: array of period returns (daily or hourly)
        n_trials: total number of strategy configurations tested (for DSR)
        risk_free: risk-free rate per period
        periods_per_year: number of trading periods per year

    Returns:
        dict of metrics suitable for wandb.log()
    """
    returns = np.asarray(returns, dtype=float)
    return {
        # Statistical significance
        "sharpe": sharpe_ratio(returns, risk_free),
        "psr": probabilistic_sharpe_ratio(returns),
        "dsr": deflated_sharpe_ratio(returns, n_trials),
        "min_track_record": minimum_track_record_length(returns),
        "n_trials": n_trials,
        # Risk
        "sortino": sortino_ratio(returns, risk_free),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "cvar_5pct": conditional_var(returns, 0.05),
        # Consistency
        "rolling_sharpe_std": rolling_sharpe_std(returns),
        "profit_factor": profit_factor(returns),
        "worst_year_sharpe": worst_year_sharpe(returns, periods_per_year),
        # Practical
        "n_observations": len(returns),
        "win_rate": float(np.mean(returns > 0)),
        "mean_return": float(np.mean(returns)),
        "total_return": float(np.prod(1 + returns) - 1),
    }
