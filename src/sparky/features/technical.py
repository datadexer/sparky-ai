"""Technical indicator calculations.

Implements RSI (Wilder's smoothing), EMA, MACD, and momentum.
All implementations are validated against pandas_ta in tests/test_cross_validation.py.

CRITICAL: RSI uses Wilder's smoothing (exponential), NOT simple moving average.
CRITICAL: Momentum sign convention — positive momentum = price went UP = bullish.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average with SMA seed.

    Uses SMA of the first `span` values as the initial seed,
    then applies exponential smoothing. This matches pandas_ta behavior.

    Formula: EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
    where alpha = 2 / (span + 1), and EMA_0 = SMA(first span values).

    Args:
        series: Input series.
        span: EMA span (window size).

    Returns:
        EMA series. First (span-1) values are NaN.
    """
    alpha = 2.0 / (span + 1)

    result = pd.Series(np.nan, index=series.index, dtype=float)

    # Seed with SMA of first `span` values
    if len(series) < span:
        return result

    sma_seed = series.iloc[:span].mean()
    result.iloc[span - 1] = sma_seed

    # Apply EMA from span onwards
    for i in range(span, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]

    return result


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index using Wilder's smoothing.

    Matches pandas_ta implementation:
    1. Compute price changes (deltas)
    2. Separate positive (gains) and negative (losses) changes
    3. Apply Wilder's Moving Average (RMA) to both: ewm(alpha=1/period, adjust=False)
    4. RSI = 100 * avg_gain / (avg_gain + abs(avg_loss))

    This is equivalent to Wilder's original formula but uses pandas ewm
    for numerical stability and performance.

    Args:
        prices: Price series (typically close prices).
        period: RSI lookback period (default 14).

    Returns:
        RSI series (0-100 range). First `period` values are NaN.
    """
    delta = prices.diff()

    positive = delta.copy()
    negative = delta.copy()
    positive[positive < 0] = 0.0
    negative[negative > 0] = 0.0

    # Wilder's MA (RMA): EWM with alpha=1/period, no adjustment
    avg_gain = positive.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = negative.abs().ewm(alpha=1.0 / period, adjust=False).mean()

    result = 100.0 * avg_gain / (avg_gain + avg_loss)

    # Set first `period` values to NaN (insufficient warmup data)
    result.iloc[:period] = np.nan

    return result


def macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD (Moving Average Convergence Divergence).

    Formula:
        MACD line = EMA(fast) - EMA(slow)
        Signal line = EMA(MACD line, signal_period)
        Histogram = MACD line - Signal line

    Uses SMA-seeded EMA matching pandas_ta behavior.

    Args:
        prices: Price series (typically close prices).
        fast_period: Fast EMA span (default 12).
        slow_period: Slow EMA span (default 26).
        signal_period: Signal line EMA span (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = ema(prices, fast_period)
    ema_slow = ema(prices, slow_period)

    macd_line = ema_fast - ema_slow

    # Signal line EMA: use only the valid (non-NaN) portion of MACD line
    # pandas_ta computes signal EMA starting from where MACD line becomes valid
    signal_line = ema(macd_line.dropna().reset_index(drop=True), signal_period)
    # Re-index to match original
    valid_start = macd_line.first_valid_index()
    if valid_start is not None:
        valid_pos = macd_line.index.get_loc(valid_start)
        full_signal = pd.Series(np.nan, index=macd_line.index, dtype=float)
        for i, val in enumerate(signal_line):
            if not np.isnan(val):
                full_signal.iloc[valid_pos + i] = val
        signal_line = full_signal
    else:
        signal_line = pd.Series(np.nan, index=macd_line.index, dtype=float)

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def momentum(prices: pd.Series, period: int = 30) -> pd.Series:
    """Compute price momentum (rate of change).

    Formula: momentum_t = (P_t - P_{t-period}) / P_{t-period}

    Sign convention (CRITICAL — v1 had a sign bug here):
        Positive momentum = price went UP = bullish signal
        Negative momentum = price went DOWN = bearish signal

    Args:
        prices: Price series.
        period: Lookback period (default 30 days).

    Returns:
        Momentum series as fractional change. First `period` values are NaN.
    """
    return (prices - prices.shift(period)) / prices.shift(period)
