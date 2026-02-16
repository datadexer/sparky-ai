"""Advanced technical features for high-frequency data.

Specialized features for hourly/sub-daily data that capture:
1. Market microstructure (intraday patterns, volatility clustering)
2. Multi-timeframe dynamics (trend alignment across scales)
3. Volume-price relationships (liquidity, order flow)
4. Temporal patterns (session effects, time-of-day)

All features are designed to be theoretically grounded and avoid data mining.
"""

import numpy as np
import pandas as pd


def bollinger_bands(
    prices: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (volatility envelope).

    Formula:
        Middle band = SMA(price, period)
        Upper band = Middle + (num_std * rolling_std)
        Lower band = Middle - (num_std * rolling_std)

    Args:
        prices: Price series
        period: Rolling window (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return middle, upper, lower


def bollinger_bandwidth(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Compute Bollinger Band bandwidth (volatility measure).

    Formula: bandwidth = (upper_band - lower_band) / middle_band

    High bandwidth = high volatility (bands wide)
    Low bandwidth = low volatility (bands narrow, potential breakout)

    Args:
        prices: Price series
        period: Rolling window
        num_std: Number of standard deviations

    Returns:
        Bandwidth series (normalized volatility)
    """
    middle, upper, lower = bollinger_bands(prices, period, num_std)
    bandwidth = (upper - lower) / middle
    return bandwidth


def bollinger_position(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Compute price position within Bollinger Bands.

    Formula: position = (price - lower_band) / (upper_band - lower_band)

    Values:
        0.0 = at lower band (oversold)
        0.5 = at middle band (neutral)
        1.0 = at upper band (overbought)
        >1.0 = above upper band (breakout)
        <0.0 = below lower band (breakdown)

    Args:
        prices: Price series
        period: Rolling window
        num_std: Number of standard deviations

    Returns:
        Position series (0-1 range, can exceed)
    """
    middle, upper, lower = bollinger_bands(prices, period, num_std)
    position = (prices - lower) / (upper - lower)
    return position


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) — volatility measure.

    Formula:
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA(True Range, period)

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: EMA smoothing period

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr_series


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute normalized intraday range.

    Formula: range = (high - low) / close

    Measures intraday volatility normalized by price level.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Normalized range series
    """
    return (high - low) / close


def volume_momentum(volume: pd.Series, period: int = 30) -> pd.Series:
    """Compute volume momentum (rate of change).

    Formula: vol_momentum = (vol_t - vol_{t-period}) / vol_{t-period}

    Positive = increasing volume (accumulation/distribution)
    Negative = decreasing volume (quiet period)

    Args:
        volume: Volume series
        period: Lookback period

    Returns:
        Volume momentum series
    """
    return (volume - volume.shift(period)) / volume.shift(period)


def volume_ma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute volume relative to moving average.

    Formula: ratio = volume / SMA(volume, period)

    >1.0 = above-average volume
    <1.0 = below-average volume

    Args:
        volume: Volume series
        period: MA period

    Returns:
        Volume ratio series
    """
    vol_ma = volume.rolling(window=period).mean()
    return volume / vol_ma


def vwap_deviation(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 24
) -> pd.Series:
    """Compute price deviation from VWAP (Volume Weighted Average Price).

    Formula:
        VWAP = sum(price * volume) / sum(volume) over rolling window
        deviation = (close - VWAP) / VWAP

    Used for mean-reversion signals.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Rolling window (24 hours for daily VWAP)

    Returns:
        VWAP deviation series
    """
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    deviation = (close - vwap) / vwap
    return deviation


def higher_highs_lower_lows(high: pd.Series, low: pd.Series, period: int = 5) -> pd.Series:
    """Detect higher highs and lower lows (trend strength).

    Formula:
        +1 if high > max(high_{t-period:t-1}) (higher high, uptrend)
        -1 if low < min(low_{t-period:t-1}) (lower low, downtrend)
         0 otherwise (consolidation)

    Args:
        high: High price series
        low: Low price series
        period: Lookback period

    Returns:
        Trend direction series (-1, 0, +1)
    """
    max_high = high.rolling(window=period).max().shift(1)
    min_low = low.rolling(window=period).min().shift(1)

    higher_high = (high > max_high).astype(int)
    lower_low = (low < min_low).astype(int)

    return higher_high - lower_low


def volatility_clustering(returns: pd.Series, period: int = 24) -> pd.Series:
    """Compute volatility clustering measure (ARCH effect).

    Formula:
        realized_vol = rolling_std(returns, period)
        vol_ratio = realized_vol / mean(realized_vol, long_period)

    High vol_ratio = volatility regime (clustering)
    Low vol_ratio = calm regime

    Args:
        returns: Return series
        period: Short window for realized vol

    Returns:
        Volatility ratio series
    """
    realized_vol = returns.rolling(window=period).std()
    long_period = period * 5
    mean_vol = realized_vol.rolling(window=long_period).mean()
    vol_ratio = realized_vol / mean_vol
    return vol_ratio


def price_distance_from_sma(prices: pd.Series, period: int = 200) -> pd.Series:
    """Compute price distance from long-term SMA (trend filter).

    Formula: distance = (price - SMA) / SMA

    Positive = above trend (bullish)
    Negative = below trend (bearish)

    Args:
        prices: Price series
        period: SMA period (200 for long-term trend)

    Returns:
        Distance series
    """
    sma = prices.rolling(window=period).mean()
    distance = (prices - sma) / sma
    return distance


def momentum_quality(prices: pd.Series, period: int = 30) -> pd.Series:
    """Compute momentum quality (% of days up during momentum period).

    Formula: quality = count(close > open) / period

    High quality (>0.6) = strong, consistent momentum
    Low quality (<0.4) = choppy, weak momentum

    Args:
        prices: Price series
        period: Lookback period

    Returns:
        Momentum quality series (0-1 range)
    """
    returns = prices.pct_change()
    positive_days = (returns > 0).astype(int)
    quality = positive_days.rolling(window=period).mean()
    return quality


def session_hour(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Extract hour of day (0-23) for session analysis.

    Crypto market sessions (approximate):
    - Asian: 0-8 UTC
    - European: 8-16 UTC
    - US: 16-24 UTC

    Args:
        timestamps: DatetimeIndex (UTC)

    Returns:
        Hour series (0-23)
    """
    return pd.Series(timestamps.hour, index=timestamps, dtype=int)


def day_of_week(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Extract day of week (0=Monday, 6=Sunday).

    Crypto weekend effect: Lower volume Sat/Sun.

    Args:
        timestamps: DatetimeIndex (UTC)

    Returns:
        Day of week series (0-6)
    """
    return pd.Series(timestamps.dayofweek, index=timestamps, dtype=int)


def price_acceleration(prices: pd.Series, period: int = 10) -> pd.Series:
    """Compute price acceleration (second derivative).

    Formula:
        velocity = momentum(price, period)
        acceleration = velocity - velocity_{t-period}

    Positive acceleration = momentum increasing (trend strengthening)
    Negative acceleration = momentum decreasing (trend weakening)

    Args:
        prices: Price series
        period: Lookback period

    Returns:
        Acceleration series
    """
    velocity = (prices - prices.shift(period)) / prices.shift(period)
    acceleration = velocity - velocity.shift(period)
    return acceleration


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Helper: Compute EMA using pandas ewm.

    Args:
        series: Input series
        span: EMA span

    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()
