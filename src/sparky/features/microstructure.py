"""Market Microstructure Features

Order flow and intraday pattern features for hourly crypto data.
Reveals patterns invisible in daily aggregations.
"""

import numpy as np
import pandas as pd


def tick_direction_ratio(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Percentage of up-ticks (close > open) in rolling window.

    High ratio (>60%) suggests sustained buying pressure.
    Low ratio (<40%) suggests sustained selling pressure.

    Args:
        df: OHLCV DataFrame
        window: Lookback period in hours (default 24h)

    Returns:
        Series with values [0, 1] indicating % of bullish candles
    """
    up_ticks = (df["close"] > df["open"]).astype(int)
    return up_ticks.rolling(window=window, min_periods=window).mean()


def candle_body_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of candle body to total range.

    Formula: abs(close - open) / (high - low)

    High ratio (>0.7): Strong directional move
    Low ratio (<0.3): Indecision, potential reversal

    Args:
        df: OHLCV DataFrame

    Returns:
        Series with values [0, 1]
    """
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]

    # Avoid division by zero
    ratio = pd.Series(0.0, index=df.index)
    mask = total_range > 0
    ratio[mask] = body[mask] / total_range[mask]

    return ratio


def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper wick size relative to total range.

    Formula: (high - max(open, close)) / (high - low)

    High ratio: Strong rejection of higher prices (bearish)

    Args:
        df: OHLCV DataFrame

    Returns:
        Series with values [0, 1]
    """
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    total_range = df["high"] - df["low"]

    ratio = pd.Series(0.0, index=df.index)
    mask = total_range > 0
    ratio[mask] = upper_wick[mask] / total_range[mask]

    return ratio


def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Lower wick size relative to total range.

    Formula: (min(open, close) - low) / (high - low)

    High ratio: Strong rejection of lower prices (bullish)

    Args:
        df: OHLCV DataFrame

    Returns:
        Series with values [0, 1]
    """
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    total_range = df["high"] - df["low"]

    ratio = pd.Series(0.0, index=df.index)
    mask = total_range > 0
    ratio[mask] = lower_wick[mask] / total_range[mask]

    return ratio


def consecutive_candles(df: pd.DataFrame, direction: str = "green") -> pd.Series:
    """Count of consecutive bullish/bearish candles.

    Args:
        df: OHLCV DataFrame
        direction: "green" (close > open) or "red" (close < open)

    Returns:
        Series with count of consecutive candles in same direction
    """
    if direction == "green":
        condition = df["close"] > df["open"]
    elif direction == "red":
        condition = df["close"] < df["open"]
    else:
        raise ValueError(f"direction must be 'green' or 'red', got {direction}")

    # Reset counter when condition changes
    counter = pd.Series(0, index=df.index)
    current_count = 0

    for i, is_match in enumerate(condition):
        if is_match:
            current_count += 1
        else:
            current_count = 0
        counter.iloc[i] = current_count

    return counter


def high_low_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling average of high/low ratio.

    Tracks range expansion/contraction.
    Rising ratio: Increasing volatility
    Falling ratio: Consolidation

    Args:
        df: OHLCV DataFrame
        window: Lookback period in hours

    Returns:
        Series with ratio values (typically 1.01-1.10 for hourly BTC)
    """
    ratio = df["high"] / df["low"]
    return ratio.rolling(window=window, min_periods=window).mean()


def bid_ask_imbalance_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy for sell pressure without order book data.

    Formula: (high - close) / (high - low)

    High value: Close near low → selling pressure
    Low value: Close near high → buying pressure

    Args:
        df: OHLCV DataFrame

    Returns:
        Series with values [0, 1]
    """
    numerator = df["high"] - df["close"]
    denominator = df["high"] - df["low"]

    imbalance = pd.Series(0.5, index=df.index)  # Neutral default
    mask = denominator > 0
    imbalance[mask] = numerator[mask] / denominator[mask]

    return imbalance


def intraday_momentum_reversal(df: pd.DataFrame) -> pd.Series:
    """Detect when intraday direction reverses from previous candle.

    Formula: sign(close - open) != sign(close_{t-1} - open_{t-1})

    Value 1: Reversal detected
    Value 0: Continuation

    Args:
        df: OHLCV DataFrame

    Returns:
        Binary series (0 or 1)
    """
    current_direction = np.sign(df["close"] - df["open"])
    prev_direction = current_direction.shift(1)

    reversal = (current_direction != prev_direction).astype(int)
    reversal.iloc[0] = 0  # No reversal on first candle

    return reversal


def overnight_gap(df: pd.DataFrame) -> pd.Series:
    """Gap between current open and previous close.

    Formula: (open_t - close_{t-1}) / close_{t-1}

    Positive: Gap up
    Negative: Gap down

    Note: Crypto trades 24/7, but exchanges can have brief maintenance.

    Args:
        df: OHLCV DataFrame

    Returns:
        Series with gap size as fraction of price
    """
    prev_close = df["close"].shift(1)
    gap = (df["open"] - prev_close) / prev_close
    gap.iloc[0] = 0  # No gap on first candle

    return gap
