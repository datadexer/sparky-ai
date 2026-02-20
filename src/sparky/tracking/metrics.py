"""Strategy evaluation metrics beyond Sharpe ratio."""

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

_VALID_PPY = {365, 1095, 2190, 4380, 8760}


def sharpe_ratio(returns, risk_free=0.0):
    """Per-period (non-annualized) Sharpe ratio.

    Returns raw mean/std — callers must multiply by sqrt(periods_per_year) to
    annualize (e.g. sqrt(365) for daily crypto). Used internally by PSR/DSR
    which require the per-period form. Do not compare directly against
    annualized benchmarks without applying the annualization factor.
    """
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

    variance = (1 - skew * sr + ((kurt - 1) / 4) * sr**2) / (T - 1)
    if variance <= 0:
        return 0.0
    return float(norm.cdf((sr - sr_benchmark) / np.sqrt(variance)))


def deflated_sharpe_ratio(returns, n_trials, sr_variance=None):
    """Sharpe ratio corrected for multiple testing.
    Bailey & Lopez de Prado (2014).

    This is THE key metric. A DSR > 0.95 means <5% chance the result is a fluke
    after accounting for all n_trials strategies tested.

    The SE uses the observed SR (not sr0) per the paper's formula:
      DSR = Phi( (SR_hat - SR_0) / SE(SR_hat) )
    where SE(SR_hat) = sqrt((1 - γ₃·SR_hat + (γ₄-1)/4·SR_hat²) / (T-1))
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
        (1 - gamma) * norm.ppf(1 - 1 / max(n_trials, 2)) + gamma * norm.ppf(1 - 1 / (max(n_trials, 2) * e))
    )

    # SE of the observed SR estimator (Mertens 2002, non-normality adjusted)
    variance = (1 - skew * sr + ((kurt - 1) / 4) * sr**2) / (T - 1)
    if variance <= 0:
        return 0.0
    return float(norm.cdf((sr - sr0) / np.sqrt(variance)))


def expected_max_sharpe(n_trials, T, sr_variance=None):
    """Expected maximum Sharpe from noise alone given n_trials.
    False Strategy Theorem — Bailey & Lopez de Prado (2014)."""
    if sr_variance is None:
        sr_variance = 1.0 / T
    gamma = 0.5772156649015329
    e = np.exp(1)
    return float(
        np.sqrt(sr_variance)
        * ((1 - gamma) * norm.ppf(1 - 1 / max(n_trials, 2)) + gamma * norm.ppf(1 - 1 / (max(n_trials, 2) * e)))
    )


def analytical_dsr(sr, skewness, kurtosis, T, n_trials, sr_variance=None):
    """DSR from summary statistics only — no return series needed.

    Computes the Deflated Sharpe Ratio analytically from five summary
    statistics. Mathematically identical to deflated_sharpe_ratio() but
    does not require the raw return array.

    Args:
        sr: Observed (sample) Sharpe ratio.
        skewness: Sample skewness (3rd standardized moment). None → 0 (Gaussian).
        kurtosis: Sample raw/Pearson kurtosis (4th standardized moment).
            Normal distribution = 3. NOT excess kurtosis (scipy default).
            None → 3 (Gaussian).
        T: Number of return observations.
        n_trials: Number of independent strategies tested.
        sr_variance: Variance of SR across trials. Default: 1/T.

    Returns:
        DSR in [0, 1]. >0.95 means <5% chance the result is a fluke.

    Note:
        When skewness/kurtosis are None (Gaussian fallback), DSR may be
        overstated for fat-tailed returns (typical of crypto). Results
        should be labeled as approximate.
    """
    import logging as _log

    if skewness is None or kurtosis is None:
        _log.getLogger(__name__).warning(
            "analytical_dsr: skewness/kurtosis not provided, using Gaussian "
            "assumption (skew=0, kurt=3). DSR may be overstated for fat-tailed returns."
        )
        skewness = skewness if skewness is not None else 0.0
        kurtosis = kurtosis if kurtosis is not None else 3.0

    if sr_variance is None:
        sr_variance = 1.0 / T

    # Expected maximum SR under null (False Strategy Theorem)
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    e = np.exp(1)
    N = max(n_trials, 2)
    sr0 = np.sqrt(sr_variance) * ((1 - gamma) * norm.ppf(1 - 1.0 / N) + gamma * norm.ppf(1 - 1.0 / (N * e)))

    # SE of the observed SR estimator (Mertens 2002)
    variance = (1 - skewness * sr + ((kurtosis - 1) / 4) * sr**2) / (T - 1)
    if variance <= 0:
        return 0.0
    return float(norm.cdf((sr - sr0) / np.sqrt(variance)))


def minimum_track_record_length(returns, sr_benchmark=0.0, confidence=0.95):
    """Minimum number of observations needed to conclude SR > benchmark
    at given confidence level."""
    sr = sharpe_ratio(returns)
    skew = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
    kurt = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4))
    z = norm.ppf(confidence)

    if sr - sr_benchmark == 0:
        return float("inf")

    return float((1 - skew * sr + ((kurt - 1) / 4) * sr**2) * (z / (sr - sr_benchmark)) ** 2)


# === RISK METRICS ===


def sortino_ratio(returns, risk_free=0.0):
    """Like Sharpe but only penalizes downside volatility.

    Uses full-series downside deviation: sqrt(mean(min(excess, 0)^2)),
    which includes zeros for positive returns (correct RMS formulation).
    """
    excess = returns - risk_free
    downside = np.minimum(excess, 0.0)  # zeros for positive returns
    downside_dev = np.sqrt(np.mean(downside**2))  # RMS over full series
    return float(np.mean(excess) / downside_dev) if downside_dev > 0 else 0.0


def max_drawdown(returns):
    """Maximum peak-to-trough decline."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def calmar_ratio(returns, periods_per_year=365):
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
        rolling_srs.append(np.mean(chunk) / np.std(chunk, ddof=1) if np.std(chunk) > 0 else 0)
    return float(np.std(rolling_srs))


def profit_factor(returns):
    """Gross profits / gross losses. >1 = profitable, >2 = strong."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


def worst_year_sharpe(returns, periods_per_year=365):
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


def validate_periods_per_year(returns, periods_per_year, strict=True):
    """Validate periods_per_year against the data. Raises on gross mismatch."""
    if periods_per_year not in _VALID_PPY:
        raise ValueError(
            f"periods_per_year={periods_per_year} is not a standard crypto frequency. "
            f"Valid values: {sorted(_VALID_PPY)}"
        )
    # If returns has a DatetimeIndex, cross-check against actual data frequency
    idx = getattr(returns, "index", None)
    has_datetime_index = idx is not None and hasattr(idx, "dtype") and "datetime" in str(idx.dtype)

    if has_datetime_index:
        try:
            span_days = (idx[-1] - idx[0]).total_seconds() / 86400
            if span_days > 30:
                actual_ppy = len(idx) / (span_days / 365.25)
                ratio = periods_per_year / actual_ppy
                if ratio > 1.6 or ratio < 0.6:
                    msg = (
                        f"periods_per_year={periods_per_year} but data implies ~{actual_ppy:.0f} "
                        f"obs/year (ratio={ratio:.2f}). Likely wrong ppy — would inflate Sharpe "
                        f"by sqrt({ratio:.2f})={ratio**0.5:.2f}x."
                    )
                    if strict:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
        except (TypeError, IndexError):
            pass
    elif periods_per_year > 1100:
        # numpy array (no DatetimeIndex) with high-frequency ppy — dangerous
        # This catches .values + ppy=2190 on 8h data (the exact failure mode)
        if strict:
            raise ValueError(
                f"periods_per_year={periods_per_year} with no DatetimeIndex. "
                f"High-frequency ppy requires a DatetimeIndex for validation to prevent "
                f"Sharpe inflation. Pass a pandas Series (not .values) or set strict_ppy=False."
            )
        else:
            logger.warning(
                "periods_per_year=%d with no DatetimeIndex — cannot validate. "
                "Sharpe may be inflated if data resolution doesn't match ppy.",
                periods_per_year,
            )


def compute_all_metrics(returns, n_trials=1, risk_free=0.0, periods_per_year=365, strict_ppy=True):
    """Compute comprehensive strategy metrics from a returns series.

    Args:
        returns: array of period returns (daily or hourly)
        n_trials: total number of strategy configurations tested (for DSR)
        risk_free: risk-free rate per period
        periods_per_year: number of trading periods per year (must be in _VALID_PPY)
        strict_ppy: if True (default), reject numpy arrays with ppy > 1100
    """
    validate_periods_per_year(returns, periods_per_year, strict=strict_ppy)
    returns = np.asarray(returns, dtype=float)

    # Pre-compute distribution moments for logging (raw Pearson kurtosis, normal=3)
    std = np.std(returns)
    if std > 0:
        standardized = (returns - np.mean(returns)) / std
        _skewness = float(np.mean(standardized**3))
        _kurtosis = float(np.mean(standardized**4))
    else:
        _skewness = 0.0
        _kurtosis = 3.0

    sr_per_period = sharpe_ratio(returns, risk_free)
    sr_annualized = sr_per_period * np.sqrt(periods_per_year)

    return {
        # Statistical significance
        # sharpe_per_period: raw mean/std (no annualization) — used by PSR/DSR formulas
        # sharpe: annualized (multiply per-period by sqrt(periods_per_year)) — compare to benchmarks
        "sharpe": sr_annualized,
        "sharpe_per_period": sr_per_period,
        "psr": probabilistic_sharpe_ratio(returns),
        "dsr": deflated_sharpe_ratio(returns, n_trials),
        "min_track_record": minimum_track_record_length(returns),
        "n_trials": n_trials,
        # Distribution moments (for analytical_dsr recomputation)
        "skewness": _skewness,
        "kurtosis": _kurtosis,  # raw Pearson kurtosis, normal=3
        # Risk (annualized, matching sharpe convention)
        "sortino": sortino_ratio(returns, risk_free) * np.sqrt(periods_per_year),
        "sortino_per_period": sortino_ratio(returns, risk_free),
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
