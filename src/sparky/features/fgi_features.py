"""Fear & Greed Index derived features for sentiment signals."""

import pandas as pd

__all__ = ["fgi_extreme_signal", "fgi_exposure_adjustment"]


def fgi_extreme_signal(
    fgi: pd.Series,
    fear_threshold: int = 20,
    greed_threshold: int = 80,
) -> pd.Series:
    """Generate a contrarian signal from FGI extremes.

    Returns +1.0 when FGI <= fear_threshold (extreme fear = buy signal),
    -1.0 when FGI >= greed_threshold (extreme greed = sell signal),
    0.0 otherwise.

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
    signal = pd.Series(0.0, index=fgi.index)
    signal[fgi <= fear_threshold] = 1.0
    signal[fgi >= greed_threshold] = -1.0
    return signal


def fgi_exposure_adjustment(
    fgi: pd.Series,
    fear_threshold: int = 20,
    greed_threshold: int = 80,
    adjustment: float = 0.25,
) -> pd.Series:
    """Adjust position exposure based on FGI extremes.

    Returns a multiplier: 1.0 + adjustment during extreme fear (increase
    exposure), 1.0 - adjustment during extreme greed (reduce exposure),
    1.0 otherwise (no change).

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
    adj = pd.Series(1.0, index=fgi.index)
    adj[fgi <= fear_threshold] = 1.0 + adjustment
    adj[fgi >= greed_threshold] = 1.0 - adjustment
    return adj
