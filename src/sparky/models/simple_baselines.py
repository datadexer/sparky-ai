"""Simple baseline trading strategies.

Research shows simple trend-following strategies can achieve Sharpe 1.5+ on Bitcoin.
These baselines establish a "complexity floor" - any ML strategy must beat these.

Strategies implemented:
1. SMA Crossover: LONG if price > SMA(200), FLAT otherwise
2. Donchian Channel Breakout: LONG on 20-day high, EXIT on 10-day low
3. ATR-Filtered Momentum: LONG if momentum > 0 AND ATR > median

Research:
- Grayscale: "The Trend is Your Friend" (momentum + volatility)
- QuantifiedStrategies: Donchian + ATR → Sharpe 1.5+, alpha 10.8% vs BTC
- Simple moving averages work well for Bitcoin's wild swings
"""

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def sma_crossover_strategy(
    prices: pd.Series,
    sma_period: int = 200,
) -> pd.Series:
    """Simple Moving Average crossover strategy.

    Signal: LONG (1) if price > SMA(period), FLAT (0) otherwise.

    This is one of the simplest trend-following strategies.
    Research shows it works well for Bitcoin due to persistent trends.

    Args:
        prices: Close prices (daily or hourly frequency).
        sma_period: SMA lookback period (default 200 for daily, 200*24=4800 for hourly).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    sma = prices.rolling(window=sma_period).mean()

    # LONG if price > SMA, FLAT otherwise
    signals = (prices > sma).astype(int)

    # Fill initial NaN values (before SMA is computed)
    signals = signals.fillna(0)

    n_long = signals.sum()
    n_total = len(signals)

    logger.info(
        f"SMA({sma_period}) Crossover: {n_long} LONG ({n_long / n_total * 100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long) / n_total * 100:.1f}%)"
    )

    return signals


def donchian_channel_strategy(
    prices: pd.Series,
    entry_period: int = 20,
    exit_period: int = 10,
) -> pd.Series:
    """Donchian Channel breakout strategy.

    Signal:
    - LONG (1): Price breaks above highest high of last `entry_period` days
    - EXIT (0): Price breaks below lowest low of last `exit_period` days
    - Hold position until exit signal

    Classic trend-following strategy. Research shows Donchian + ATR position sizing
    achieves Sharpe 1.5+ on Bitcoin.

    Args:
        prices: Close prices (daily frequency recommended).
        entry_period: Lookback for entry breakout (default 20 days).
        exit_period: Lookback for exit breakout (default 10 days).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute Donchian channels
    upper_channel = prices.rolling(window=entry_period).max()
    lower_channel = prices.rolling(window=exit_period).min()

    # Initialize signals
    signals = pd.Series(0, index=prices.index, dtype=int)

    # Track position state
    in_position = False

    for i in range(len(prices)):
        if i < entry_period:
            # Not enough data yet
            signals.iloc[i] = 0
            continue

        current_price = prices.iloc[i]

        if not in_position:
            # Check for entry: price breaks above upper channel
            if current_price >= upper_channel.iloc[i - 1]:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            # In position: check for exit
            if i >= exit_period and current_price <= lower_channel.iloc[i - 1]:
                in_position = False
                signals.iloc[i] = 0
            else:
                # Hold position
                signals.iloc[i] = 1

    n_long = signals.sum()
    n_total = len(signals)

    logger.info(
        f"Donchian({entry_period}/{exit_period}): {n_long} LONG ({n_long / n_total * 100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long) / n_total * 100:.1f}%)"
    )

    return signals


def atr_filtered_momentum_strategy(
    prices: pd.Series,
    momentum_period: int = 30,
    atr_period: int = 14,
    use_median_filter: bool = True,
) -> pd.Series:
    """ATR-filtered momentum strategy.

    Signal: LONG if momentum > 0 AND ATR > median(ATR), FLAT otherwise.

    Research shows volatility is beneficial for trend followers.
    ATR filter ensures we only trade during volatile periods (when trends are strong).

    Args:
        prices: Close prices (daily frequency recommended).
        momentum_period: Lookback for momentum (default 30 days).
        atr_period: Lookback for ATR computation (default 14 days).
        use_median_filter: If True, require ATR > median. If False, always trade.

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute momentum (rate of change)
    momentum = prices.pct_change(momentum_period)

    # Compute ATR (Average True Range) - using close-to-close range as proxy
    # For proper ATR, we'd need high/low, but close-to-close is acceptable approximation
    returns = prices.pct_change()
    true_range = returns.abs()
    atr = true_range.rolling(window=atr_period).mean()

    # Volatility filter
    if use_median_filter:
        atr_median = atr.median()
        volatility_filter = atr > atr_median
    else:
        volatility_filter = pd.Series(True, index=prices.index)

    # Momentum filter
    momentum_filter = momentum > 0

    # Combined signal: LONG if momentum positive AND volatility high
    signals = (momentum_filter & volatility_filter).astype(int)

    # Fill initial NaN values
    signals = signals.fillna(0)

    n_long = signals.sum()
    n_total = len(signals)

    logger.info(
        f"ATR-Filtered Momentum({momentum_period}/{atr_period}): "
        f"{n_long} LONG ({n_long / n_total * 100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long) / n_total * 100:.1f}%)"
    )

    return signals


def compute_baseline_signals(
    prices: pd.Series,
    strategy: Literal["sma", "donchian", "atr_momentum"] = "sma",
    **kwargs,
) -> pd.Series:
    """Compute signals for a simple baseline strategy.

    Args:
        prices: Close prices.
        strategy: Strategy type ("sma", "donchian", or "atr_momentum").
        **kwargs: Strategy-specific parameters.

    Returns:
        Series of signals (1 = LONG, 0 = FLAT).
    """
    if strategy == "sma":
        return sma_crossover_strategy(prices, **kwargs)
    elif strategy == "donchian":
        return donchian_channel_strategy(prices, **kwargs)
    elif strategy == "atr_momentum":
        return atr_filtered_momentum_strategy(prices, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'sma', 'donchian', or 'atr_momentum'.")
