"""Market Regime Detection Features

Features that identify market regimes (trending/choppy, high/low vol, bull/bear).
Different features work better in different regimes.
"""

import numpy as np
import pandas as pd


def drawdown_from_high(close: pd.Series, window: int = 20) -> pd.Series:
    """Current drawdown from rolling high.

    Formula: (high_N - close) / high_N

    High value (>10%): Deep pullback, potential support test
    Low value (<2%): Near highs, potential breakout

    Args:
        close: Close price series
        window: Lookback period in hours

    Returns:
        Series with drawdown as fraction (0 to 1)
    """
    rolling_high = close.rolling(window=window, min_periods=window).max()
    drawdown = (rolling_high - close) / rolling_high

    # First N candles have no drawdown
    drawdown.iloc[: window - 1] = 0

    return drawdown


def recovery_from_low(close: pd.Series, window: int = 20) -> pd.Series:
    """Current recovery from rolling low.

    Formula: (close - low_N) / low_N

    High value (>10%): Strong recovery from lows
    Low value (<2%): Near lows, potential breakdown

    Args:
        close: Close price series
        window: Lookback period in hours

    Returns:
        Series with recovery as fraction (0 to inf)
    """
    rolling_low = close.rolling(window=window, min_periods=window).min()
    recovery = (close - rolling_low) / rolling_low

    # First N candles have no recovery
    recovery.iloc[: window - 1] = 0

    return recovery


def volatility_regime(returns: pd.Series, window: int = 168) -> pd.Series:
    """Percentile rank of current volatility vs recent history.

    Formula: percentile_rank(realized_vol, window=168h)

    High (>80%): High volatility regime
    Low (<20%): Low volatility regime

    Args:
        returns: Log returns series
        window: Lookback for percentile calculation (default 7 days)

    Returns:
        Series with percentile rank [0, 1]
    """
    realized_vol = returns.rolling(window=24, min_periods=24).std() * np.sqrt(24)

    # Compute percentile rank
    def percentile_rank(series, lookback):
        """Rolling percentile rank."""
        result = pd.Series(np.nan, index=series.index)
        for i in range(lookback, len(series)):
            window_vals = series.iloc[i - lookback : i]
            current_val = series.iloc[i]
            pct_rank = (window_vals < current_val).sum() / lookback
            result.iloc[i] = pct_rank
        return result

    return percentile_rank(realized_vol, window)


def volume_regime(volume: pd.Series, window: int = 168) -> pd.Series:
    """Percentile rank of current volume vs recent history.

    High (>80%): High volume regime (breakout likely)
    Low (<20%): Low volume regime (consolidation)

    Args:
        volume: Volume series
        window: Lookback for percentile calculation (default 7 days)

    Returns:
        Series with percentile rank [0, 1]
    """

    def percentile_rank(series, lookback):
        """Rolling percentile rank."""
        result = pd.Series(np.nan, index=series.index)
        for i in range(lookback, len(series)):
            window_vals = series.iloc[i - lookback : i]
            current_val = series.iloc[i]
            pct_rank = (window_vals < current_val).sum() / lookback
            result.iloc[i] = pct_rank
        return result

    return percentile_rank(volume, window)


def trend_strength_adx_proxy(momentum: pd.Series, volatility: pd.Series) -> pd.Series:
    """Proxy for ADX (Average Directional Index) using momentum and volatility.

    Formula: abs(momentum) / volatility

    High value: Strong trend (momentum >> volatility)
    Low value: Choppy market (momentum ~ volatility)

    Args:
        momentum: Momentum indicator (e.g., 24h momentum)
        volatility: Realized volatility (e.g., 24h std)

    Returns:
        Series with trend strength (0 to inf, typically 0-3)
    """
    strength = pd.Series(0.0, index=momentum.index)
    mask = volatility > 0
    strength[mask] = momentum[mask].abs() / volatility[mask]

    return strength


def choppiness_index(returns: pd.Series, window: int = 168) -> pd.Series:
    """Measures market choppiness vs trendiness.

    Formula: 1 - abs(total_move) / sum(abs(individual_moves))

    High (>0.7): Choppy, mean-reverting
    Low (<0.3): Trending, directional

    Args:
        returns: Returns series
        window: Lookback period (default 7 days)

    Returns:
        Series with choppiness [0, 1]
    """
    total_move = returns.rolling(window=window, min_periods=window).sum().abs()
    sum_of_moves = returns.abs().rolling(window=window, min_periods=window).sum()

    choppiness = pd.Series(0.5, index=returns.index)  # Neutral default
    mask = sum_of_moves > 0
    choppiness[mask] = 1 - (total_move[mask] / sum_of_moves[mask])

    return choppiness


def breakout_proximity_upper(close: pd.Series, window: int = 200) -> pd.Series:
    """Proximity to upper breakout level.

    Formula: (close - sma_N) / (max_N - sma_N)

    High (>0.8): Near resistance breakout
    Low (<0.2): Far from resistance

    Args:
        close: Close price series
        window: Lookback period for SMA and max

    Returns:
        Series with proximity [0, 1]
    """
    sma = close.rolling(window=window, min_periods=window).mean()
    rolling_max = close.rolling(window=window, min_periods=window).max()

    proximity = pd.Series(0.5, index=close.index)
    denominator = rolling_max - sma
    mask = denominator > 0
    proximity[mask] = (close[mask] - sma[mask]) / denominator[mask]

    # Clip to [0, 1]
    proximity = proximity.clip(0, 1)

    return proximity


def breakout_proximity_lower(close: pd.Series, window: int = 200) -> pd.Series:
    """Proximity to lower breakout level.

    Formula: (sma_N - close) / (sma_N - min_N)

    High (>0.8): Near support breakdown
    Low (<0.2): Far from support

    Args:
        close: Close price series
        window: Lookback period for SMA and min

    Returns:
        Series with proximity [0, 1]
    """
    sma = close.rolling(window=window, min_periods=window).mean()
    rolling_min = close.rolling(window=window, min_periods=window).min()

    proximity = pd.Series(0.5, index=close.index)
    denominator = sma - rolling_min
    mask = denominator > 0
    proximity[mask] = (sma[mask] - close[mask]) / denominator[mask]

    # Clip to [0, 1]
    proximity = proximity.clip(0, 1)

    return proximity
