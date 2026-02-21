"""Volatility targeting position sizing."""

import pandas as pd

__all__ = ["vol_target_position_size", "apply_vol_targeting"]


def vol_target_position_size(
    forecast_vol: pd.Series,
    target_vol: float = 0.20,
    max_leverage: float = 1.0,
    min_position: float = 0.0,
    smoothing: int = 1,
) -> pd.Series:
    """Compute position size from forecast vol and target vol.

    position = clip(target_vol / forecast_vol, min_position, max_leverage)
    Optional rolling mean smoothing to reduce turnover.
    """
    raw = (target_vol / forecast_vol).clip(lower=min_position, upper=max_leverage)
    if smoothing > 1:
        raw = raw.rolling(window=smoothing, min_periods=1).mean()
    return raw


def apply_vol_targeting(
    returns: pd.Series,
    forecast_vol: pd.Series,
    target_vol: float = 0.20,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Apply vol targeting to returns.

    Returns position-sized returns: returns * position_size.
    No look-ahead: forecast_vol is already a forward forecast.
    """
    pos_size = vol_target_position_size(forecast_vol, target_vol, max_leverage)
    return returns * pos_size
