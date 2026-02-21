"""Fear & Greed Index derived features for sentiment signals."""

import pandas as pd

__all__ = ["fgi_extreme_signal", "fgi_exposure_adjustment"]


def fgi_extreme_signal(
    fgi: pd.Series,
    fear_threshold: int = 20,
    greed_threshold: int = 80,
) -> pd.Series:
    """Generate a contrarian signal from FGI extremes.

    Returns +1.0 when previous day's FGI <= fear_threshold (extreme fear = buy),
    -1.0 when previous day's FGI >= greed_threshold (extreme greed = sell),
    0.0 otherwise. Uses a 1-bar lag to prevent look-ahead bias.

    Parameters
    ----------
    fgi : pd.Series
        Fear & Greed Index values (0-100).
    fear_threshold : int
        Values at or below this trigger a buy signal.
    greed_threshold : int
        Values at or above this trigger a sell signal.

    Returns
    -------
    pd.Series
        Signal values: +1.0, -1.0, or 0.0.
    """
    if fear_threshold >= greed_threshold:
        raise ValueError(f"fear_threshold ({fear_threshold}) must be < greed_threshold ({greed_threshold})")
    lagged = fgi.shift(1)
    signal = pd.Series(0.0, index=fgi.index)
    signal[lagged <= fear_threshold] = 1.0
    signal[lagged >= greed_threshold] = -1.0
    signal[lagged.isna()] = float("nan")
    return signal


def fgi_exposure_adjustment(
    fgi: pd.Series,
    fear_threshold: int = 20,
    greed_threshold: int = 80,
    adjustment: float = 0.25,
) -> pd.Series:
    """Adjust position exposure based on FGI extremes.

    Returns a multiplier based on previous day's FGI: 1.0 + adjustment during
    extreme fear (increase exposure), 1.0 - adjustment during extreme greed
    (reduce exposure), 1.0 otherwise. Uses a 1-bar lag to prevent look-ahead bias.

    Parameters
    ----------
    fgi : pd.Series
        Fear & Greed Index values (0-100).
    fear_threshold : int
        Values at or below this increase exposure.
    greed_threshold : int
        Values at or above this reduce exposure.
    adjustment : float
        How much to adjust exposure by (default 0.25 = +/- 25%).

    Returns
    -------
    pd.Series
        Exposure multiplier values.
    """
    if fear_threshold >= greed_threshold:
        raise ValueError(f"fear_threshold ({fear_threshold}) must be < greed_threshold ({greed_threshold})")
    if not (0.0 < adjustment < 1.0):
        raise ValueError(f"adjustment must be in (0, 1), got {adjustment}")
    lagged = fgi.shift(1)
    adj = pd.Series(1.0, index=fgi.index)
    adj[lagged <= fear_threshold] = 1.0 + adjustment
    adj[lagged >= greed_threshold] = 1.0 - adjustment
    adj[lagged.isna()] = float("nan")
    return adj
