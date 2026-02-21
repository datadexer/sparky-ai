"""Funding rate derived features for carry trading signals."""

import pandas as pd

__all__ = [
    "funding_rate_rolling_avg",
    "funding_rate_zscore",
    "funding_rate_regime",
    "funding_rate_carry_signal",
]


def funding_rate_rolling_avg(
    funding_rates: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling average of funding rates over multiple windows.

    Default windows: [8, 24, 168] (8h=1day, 24h=3days, 168h=1week for 8h data).
    """
    if windows is None:
        windows = [8, 24, 168]
    result = pd.DataFrame(index=funding_rates.index)
    for w in windows:
        result[f"fr_avg_{w}"] = funding_rates.rolling(window=w, min_periods=w).mean()
    return result


def funding_rate_zscore(
    funding_rates: pd.Series,
    window: int = 720,
) -> pd.Series:
    """Z-score of funding rate vs trailing window."""
    rolling_mean = funding_rates.rolling(window=window, min_periods=window).mean()
    rolling_std = funding_rates.rolling(window=window, min_periods=window).std()
    return (funding_rates - rolling_mean) / rolling_std.replace(0, float("nan"))


def funding_rate_regime(
    funding_rates: pd.Series,
    positive_threshold: float = 0.0001,
    window: int = 56,
) -> pd.Series:
    """Classify funding rate regime as positive/negative/neutral."""
    smoothed = funding_rates.rolling(window=window, min_periods=window).mean()
    result = pd.Series("neutral", index=funding_rates.index)
    result[smoothed > positive_threshold] = "positive"
    result[smoothed < -positive_threshold] = "negative"
    return result


def funding_rate_carry_signal(
    funding_rates: pd.Series,
    cost_threshold_pct: float = 0.065,
    window: int = 56,
) -> pd.Series:
    """Binary carry signal: 1 when smoothed funding rate > cost threshold.

    Threshold is in per-period units (e.g., 0.065% per 8h for Binance).
    Signal = 1 when shorting earns carry, 0 otherwise.
    """
    smoothed = funding_rates.rolling(window=window, min_periods=window).mean()
    threshold = cost_threshold_pct / 100
    return (smoothed > threshold).astype(float)
