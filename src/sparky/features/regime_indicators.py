"""Regime detection indicators for adaptive trading strategies.

Market regimes (volatility-based classification):
- LOW: <30% annualized volatility (calm, trending markets)
- MEDIUM: 30-60% annualized (normal crypto volatility)
- HIGH: >60% annualized (chaotic, crisis periods)

Research shows Bitcoin has "distinct volatility regimes more persistent than
S&P 500" and regime-switching models significantly improve prediction accuracy.

Usage:
    regime = compute_volatility_regime(prices, window=30*24)
    position_size = get_regime_position_size(regime)
    threshold = get_regime_threshold(regime)
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeType = Literal["low", "medium", "high"]
TrendType = Literal["uptrend", "downtrend", "sideways"]


def compute_volatility_regime(
    prices: pd.Series,
    window: int = 30 * 24,
    frequency: str = "1h",
) -> pd.Series:
    """Compute volatility regime classification.

    Classifies market into LOW/MEDIUM/HIGH volatility regimes based on
    rolling realized volatility. Regime boundaries:
    - LOW: annualized vol < 30%
    - MEDIUM: 30% <= vol < 60%
    - HIGH: vol >= 60%

    Args:
        prices: Close prices (hourly or daily frequency).
        window: Rolling window in periods (default 30 days * 24 hours = 720).
        frequency: Data frequency ("1h" or "1d") for annualization factor.

    Returns:
        Series with regime labels: "low", "medium", "high".
        Same index as input prices.
    """
    # Compute returns
    returns = prices.pct_change()

    # Annualization factor
    if frequency == "1h":
        ann_factor = np.sqrt(24 * 365)  # Hours per year
    elif frequency == "1d":
        ann_factor = np.sqrt(365)  # Days per year
    else:
        raise ValueError(f"Unknown frequency: {frequency}. Use '1h' or '1d'.")

    # Rolling realized volatility (annualized)
    vol = returns.rolling(window).std() * ann_factor

    # Classify regimes
    regime = pd.Series(index=prices.index, dtype=str)
    regime[vol < 0.30] = "low"
    regime[(vol >= 0.30) & (vol < 0.60)] = "medium"
    regime[vol >= 0.60] = "high"

    # Forward-fill initial NaN values until we have enough data
    regime = regime.bfill()

    n_low = (regime == "low").sum()
    n_medium = (regime == "medium").sum()
    n_high = (regime == "high").sum()

    logger.info(
        f"Computed volatility regimes: "
        f"low={n_low} ({n_low/len(regime)*100:.1f}%), "
        f"medium={n_medium} ({n_medium/len(regime)*100:.1f}%), "
        f"high={n_high} ({n_high/len(regime)*100:.1f}%)"
    )

    return regime


def get_regime_position_size(regime: RegimeType) -> float:
    """Get position size for a given regime.

    Regime-aware position sizing reduces exposure during chaotic periods:
    - HIGH regime: 50% position (reduce risk)
    - MEDIUM regime: 75% position (moderate caution)
    - LOW regime: 100% position (full exposure in calm markets)

    Args:
        regime: Regime label ("low", "medium", or "high").

    Returns:
        Position size as fraction (0.5, 0.75, or 1.0).
    """
    position_sizes = {
        "high": 0.50,
        "medium": 0.75,
        "low": 1.00,
    }

    if regime not in position_sizes:
        raise ValueError(f"Unknown regime: {regime}. Expected 'low', 'medium', or 'high'.")

    return position_sizes[regime]


def get_regime_threshold(regime: RegimeType) -> float:
    """Get probability threshold for a given regime.

    Higher thresholds in volatile regimes require stronger conviction:
    - HIGH regime: threshold 0.55 (require high confidence)
    - MEDIUM regime: threshold 0.52 (slight caution)
    - LOW regime: threshold 0.50 (standard)

    Args:
        regime: Regime label ("low", "medium", or "high").

    Returns:
        Probability threshold (0.50, 0.52, or 0.55).
    """
    thresholds = {
        "high": 0.55,
        "medium": 0.52,
        "low": 0.50,
    }

    if regime not in thresholds:
        raise ValueError(f"Unknown regime: {regime}. Expected 'low', 'medium', or 'high'.")

    return thresholds[regime]


def compute_regime_adjusted_signals(
    probabilities: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Compute trading signals with regime-aware thresholds and position sizes.

    Args:
        probabilities: P(up) predictions, indexed by timestamp.
        regimes: Regime labels ("low", "medium", "high"), same index as probabilities.

    Returns:
        DataFrame with columns: ["probability", "regime", "threshold", "signal", "position_size"].
        signal is 1 (LONG) if probability > regime_threshold, else 0 (FLAT).
        position_size is regime-dependent (0.5, 0.75, or 1.0) when signal=1, else 0.
    """
    # Align indices
    aligned_regimes = regimes.reindex(probabilities.index, method="ffill")

    results = []
    for timestamp, prob in probabilities.items():
        regime = aligned_regimes.loc[timestamp]

        # Get regime-specific threshold and position size
        threshold = get_regime_threshold(regime)
        position_size_full = get_regime_position_size(regime)

        # Generate signal
        signal = 1 if prob > threshold else 0
        position_size = position_size_full if signal == 1 else 0.0

        results.append({
            "timestamp": timestamp,
            "probability": prob,
            "regime": regime,
            "threshold": threshold,
            "signal": signal,
            "position_size": position_size,
        })

    df = pd.DataFrame(results).set_index("timestamp")

    n_long = (df["signal"] == 1).sum()
    logger.info(
        f"Generated regime-adjusted signals: "
        f"{n_long} LONG ({n_long/len(df)*100:.1f}%), "
        f"{len(df) - n_long} FLAT ({(len(df) - n_long)/len(df)*100:.1f}%)"
    )

    return df


def detect_trend(
    prices: pd.Series,
    sma_period: int = 200,
    trend_threshold: float = 0.02,
) -> pd.Series:
    """Detect price trend relative to moving average.

    Classifies market into UPTREND/DOWNTREND/SIDEWAYS based on distance from SMA.

    Args:
        prices: Close prices.
        sma_period: SMA lookback period (default 200 for daily, 200*24=4800 for hourly).
        trend_threshold: Threshold for sideways detection (default 2%).
            If |price - SMA| / SMA < threshold → SIDEWAYS.

    Returns:
        Series with trend labels: "uptrend", "downtrend", "sideways".
    """
    sma = prices.rolling(window=sma_period).mean()

    # Distance from SMA (as percentage)
    distance_pct = (prices - sma) / sma

    # Classify trends
    trend = pd.Series(index=prices.index, dtype=str)
    trend[distance_pct > trend_threshold] = "uptrend"
    trend[distance_pct < -trend_threshold] = "downtrend"
    trend[(distance_pct >= -trend_threshold) & (distance_pct <= trend_threshold)] = "sideways"

    # Forward-fill initial NaN values
    trend = trend.bfill()

    n_up = (trend == "uptrend").sum()
    n_down = (trend == "downtrend").sum()
    n_side = (trend == "sideways").sum()

    logger.info(
        f"Detected trends: "
        f"uptrend={n_up} ({n_up/len(trend)*100:.1f}%), "
        f"downtrend={n_down} ({n_down/len(trend)*100:.1f}%), "
        f"sideways={n_side} ({n_side/len(trend)*100:.1f}%)"
    )

    return trend


def compute_combined_regime(
    volatility_regime: pd.Series,
    trend: pd.Series,
) -> pd.Series:
    """Combine volatility regime with trend direction.

    Creates combined regime labels like "high_uptrend", "medium_sideways", etc.

    Args:
        volatility_regime: Series with "low"/"medium"/"high" labels.
        trend: Series with "uptrend"/"downtrend"/"sideways" labels.

    Returns:
        Series with combined regime labels (e.g., "high_uptrend").
    """
    # Combine regimes
    combined = volatility_regime + "_" + trend

    regime_counts = combined.value_counts()

    logger.info("Combined regime distribution:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} ({count/len(combined)*100:.1f}%)")

    return combined


def get_trend_aware_position_size(
    volatility_regime: RegimeType,
    trend: TrendType,
) -> float:
    """Get position size based on volatility regime AND trend direction.

    Research insight: High volatility can be good (volatile uptrend) or bad (volatile downtrend).
    Position sizing should account for BOTH volatility AND trend.

    Rules (corrected from Phase 2A failure):
    - HIGH vol + UPTREND: 125% position (capture volatile bull - use leverage if available)
    - HIGH vol + DOWNTREND: 25% position (reduce exposure to volatile bear)
    - HIGH vol + SIDEWAYS: 50% position (avoid whipsaws)
    - MEDIUM vol + UPTREND: 100% position (standard bull exposure)
    - MEDIUM vol + DOWNTREND: 50% position (reduce bear exposure)
    - MEDIUM vol + SIDEWAYS: 75% position (normal sideways)
    - LOW vol + UPTREND: 100% position (calm bull)
    - LOW vol + DOWNTREND: 75% position (calm bear)
    - LOW vol + SIDEWAYS: 75% position (calm sideways)

    Args:
        volatility_regime: "low", "medium", or "high".
        trend: "uptrend", "downtrend", or "sideways".

    Returns:
        Position size as fraction (0.25 to 1.25).
    """
    position_sizes = {
        ("high", "uptrend"): 1.25,      # Capture volatile bull (max exposure)
        ("high", "downtrend"): 0.25,    # Avoid volatile bear (min exposure)
        ("high", "sideways"): 0.50,     # Reduce exposure in choppy high vol
        ("medium", "uptrend"): 1.00,    # Standard bull position
        ("medium", "downtrend"): 0.50,  # Reduce bear exposure
        ("medium", "sideways"): 0.75,   # Normal sideways
        ("low", "uptrend"): 1.00,       # Calm bull (full exposure)
        ("low", "downtrend"): 0.75,     # Calm bear (some exposure)
        ("low", "sideways"): 0.75,      # Calm sideways
    }

    key = (volatility_regime, trend)
    if key not in position_sizes:
        raise ValueError(f"Unknown regime combination: {volatility_regime}, {trend}")

    return position_sizes[key]


def get_trend_aware_threshold(
    volatility_regime: RegimeType,
    trend: TrendType,
    base_threshold: float = 0.5,
) -> float:
    """Get probability threshold based on volatility regime AND trend.

    Adjust threshold based on market conditions:
    - In favorable conditions (uptrend), lower threshold (more aggressive)
    - In unfavorable conditions (downtrend, high vol), raise threshold (more conservative)

    Args:
        volatility_regime: "low", "medium", or "high".
        trend: "uptrend", "downtrend", or "sideways".
        base_threshold: Base probability threshold (default 0.5).

    Returns:
        Adjusted probability threshold.
    """
    # Threshold adjustments
    adjustments = {
        ("high", "uptrend"): -0.02,     # More aggressive in volatile bull (0.48)
        ("high", "downtrend"): +0.10,   # Very conservative in volatile bear (0.60)
        ("high", "sideways"): +0.05,    # Conservative in choppy high vol (0.55)
        ("medium", "uptrend"): 0.00,    # Standard in normal bull (0.50)
        ("medium", "downtrend"): +0.05, # Somewhat conservative in bear (0.55)
        ("medium", "sideways"): +0.02,  # Slightly conservative sideways (0.52)
        ("low", "uptrend"): -0.01,      # Slightly aggressive in calm bull (0.49)
        ("low", "downtrend"): +0.03,    # Conservative in calm bear (0.53)
        ("low", "sideways"): +0.01,     # Slightly conservative calm sideways (0.51)
    }

    key = (volatility_regime, trend)
    if key not in adjustments:
        raise ValueError(f"Unknown regime combination: {volatility_regime}, {trend}")

    return base_threshold + adjustments[key]
