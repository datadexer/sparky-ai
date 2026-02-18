"""Multi-Resolution Features

Compute same indicators at multiple timeframes (4h, 12h, 24h, 168h).
Captures patterns at different timescales.
"""

import numpy as np
import pandas as pd


def rsi_multi_resolution(close: pd.Series, windows: list[int] = None) -> dict[str, pd.Series]:
    """Compute RSI at multiple timeframes.

    Default windows: [4, 12, 14, 24, 168] hours

    Args:
        close: Close price series
        windows: List of window sizes in hours

    Returns:
        Dict mapping "rsi_{N}h" -> Series
    """
    if windows is None:
        windows = [4, 12, 14, 24, 168]

    from sparky.features.technical import rsi

    result = {}
    for window in windows:
        result[f"rsi_{window}h"] = rsi(close, period=window)

    return result


def momentum_multi_resolution(close: pd.Series, windows: list[int] = None) -> dict[str, pd.Series]:
    """Compute momentum at multiple timeframes.

    Default windows: [4, 12, 24, 72, 168] hours

    Args:
        close: Close price series
        windows: List of window sizes in hours

    Returns:
        Dict mapping "momentum_{N}h" -> Series
    """
    if windows is None:
        windows = [4, 12, 24, 72, 168]

    from sparky.features.technical import momentum

    result = {}
    for window in windows:
        result[f"momentum_{window}h"] = momentum(close, period=window)

    return result


def ema_cross_multi_resolution(
    close: pd.Series, fast: int = 9, slow: int = 21, resolutions: list[int] = None
) -> dict[str, pd.Series]:
    """Compute EMA crossover at multiple timeframes.

    Formula: (EMA_fast - EMA_slow) / EMA_slow

    Args:
        close: Close price series
        fast: Fast EMA period (default 9)
        slow: Slow EMA period (default 21)
        resolutions: Resampling windows in hours (default [4, 24])

    Returns:
        Dict mapping "ema_cross_{N}h" -> Series
    """
    if resolutions is None:
        resolutions = [4, 24]

    from sparky.features.technical import ema

    result = {}
    for window in resolutions:
        # Resample to resolution
        resampled = close.resample(f"{window}h").last()

        # Compute EMAs
        ema_fast = ema(resampled, period=fast)
        ema_slow = ema(resampled, period=slow)

        # Crossover strength
        cross = (ema_fast - ema_slow) / ema_slow

        # Upsample back to original frequency
        cross_upsampled = cross.reindex(close.index, method="ffill")

        result[f"ema_cross_{window}h"] = cross_upsampled

    return result


def bb_squeeze_multi_resolution(
    df: pd.DataFrame, window: int = 20, resolutions: list[int] = None
) -> dict[str, pd.Series]:
    """Detect Bollinger Band squeeze at multiple timeframes.

    Squeeze = BB bandwidth < 20th percentile over lookback window.
    Indicates consolidation before breakout.

    Args:
        df: OHLCV DataFrame
        window: BB period (default 20)
        resolutions: Resampling windows in hours (default [4, 24])

    Returns:
        Dict mapping "bb_squeeze_{N}h" -> Binary series (1 = squeeze)
    """
    if resolutions is None:
        resolutions = [4, 24]

    from sparky.features.advanced import bollinger_bandwidth

    result = {}
    for res in resolutions:
        # Resample to resolution
        resampled = df.resample(f"{res}h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )

        # Compute bandwidth
        bandwidth = bollinger_bandwidth(resampled["close"], period=window, num_std=2)

        # Percentile threshold (20th percentile = squeeze)
        threshold = bandwidth.rolling(window=100, min_periods=50).quantile(0.2)
        squeeze = (bandwidth < threshold).astype(int)

        # Upsample back
        squeeze_upsampled = squeeze.reindex(df.index, method="ffill")

        result[f"bb_squeeze_{res}h"] = squeeze_upsampled

    return result


def volatility_regime_multi_resolution(returns: pd.Series, resolutions: list[int] = None) -> dict[str, pd.Series]:
    """Compute volatility regime at multiple timeframes.

    Formula: realized_vol_{N}h / realized_vol_{7d}

    High ratio (>1.5): Short-term volatility spike
    Low ratio (<0.5): Short-term calm

    Args:
        returns: Returns series
        resolutions: Short-term windows in hours (default [4, 24])

    Returns:
        Dict mapping "vol_regime_{N}h" -> Series
    """
    if resolutions is None:
        resolutions = [4, 24]

    # Baseline: 7-day volatility
    baseline_vol = returns.rolling(window=168, min_periods=168).std() * np.sqrt(168)

    result = {}
    for window in resolutions:
        short_vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(window)
        regime = pd.Series(1.0, index=returns.index)  # Neutral default
        mask = baseline_vol > 0
        regime[mask] = short_vol[mask] / baseline_vol[mask]

        result[f"vol_regime_{window}h"] = regime

    return result


def trend_alignment(close: pd.Series, windows: list[int] = None) -> pd.Series:
    """Detect when short/medium/long-term trends agree.

    Formula: sign(momentum_4h) == sign(momentum_24h) == sign(momentum_168h)

    Value 1: All trends aligned (strong conviction)
    Value 0: Divergent trends (conflicting signals)

    Args:
        close: Close price series
        windows: Momentum windows (default [4, 24, 168])

    Returns:
        Binary series (0 or 1)
    """
    if windows is None:
        windows = [4, 24, 168]

    from sparky.features.technical import momentum

    # Compute momentums
    momentums = [momentum(close, period=w) for w in windows]

    # Check alignment
    signs = [np.sign(m) for m in momentums]
    aligned = pd.Series(1, index=close.index)

    for i in range(1, len(signs)):
        aligned &= signs[i] == signs[0]

    return aligned.astype(int)
