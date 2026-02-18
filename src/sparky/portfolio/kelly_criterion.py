"""Kelly Criterion position sizing.

The Kelly Criterion determines optimal position size to maximize long-term growth.

Formula:
    f* = (p*b - q) / b

Where:
    f* = Optimal fraction of bankroll to bet
    p  = Probability of win
    b  = Ratio of average win to average loss
    q  = Probability of loss (1 - p)

For trading:
    - p = win rate (fraction of profitable trades)
    - b = avg_win / avg_loss
    - f* = optimal position size (0.0 to 1.0+)

Fractional Kelly (e.g., 0.25 * f* or 0.5 * f*) is safer and reduces volatility
while still capturing most of the growth benefits.

References:
    - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
    - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_kelly_parameters(
    returns: pd.Series,
    signals: pd.Series,
) -> Tuple[float, float, float]:
    """Calculate Kelly Criterion parameters from historical returns.

    Args:
        returns: Historical returns (daily or hourly).
        signals: Trading signals (1 = LONG, 0 = FLAT).

    Returns:
        Tuple of (win_rate, win_loss_ratio, kelly_fraction):
            - win_rate: Fraction of profitable trades (p)
            - win_loss_ratio: Average win / average loss (b)
            - kelly_fraction: Optimal Kelly fraction (f*)
    """
    # Get returns only when we have a position
    position_returns = returns[signals.shift(1) == 1].dropna()

    if len(position_returns) == 0:
        logger.warning("No trades found in signals")
        return 0.0, 0.0, 0.0

    # Separate wins and losses
    wins = position_returns[position_returns > 0]
    losses = position_returns[position_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        logger.warning(f"Insufficient trades: {len(wins)} wins, {len(losses)} losses")
        return 0.0, 0.0, 0.0

    # Calculate Kelly parameters
    win_rate = len(wins) / len(position_returns)  # p
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0  # b

    # Kelly formula: f* = (p*b - q) / b
    q = 1 - win_rate
    kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio if win_loss_ratio > 0 else 0.0

    logger.info(
        f"Kelly parameters: win_rate={win_rate:.3f}, "
        f"avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}, "
        f"win_loss_ratio={win_loss_ratio:.3f}, kelly_fraction={kelly_fraction:.3f}"
    )

    return win_rate, win_loss_ratio, kelly_fraction


def apply_kelly_sizing(
    signals: pd.Series,
    returns: pd.Series,
    fraction: float = 0.25,
    max_leverage: float = 2.0,
    lookback: int = 252,
) -> pd.Series:
    """Apply Kelly Criterion position sizing to trading signals.

    Uses a rolling window to calculate Kelly parameters and adjust position sizes.
    Fractional Kelly (default 0.25) provides safer leverage while capturing most growth.

    Args:
        signals: Binary signals (1 = LONG, 0 = FLAT).
        returns: Historical returns for calculating Kelly parameters.
        fraction: Fractional Kelly multiplier (0.25 = quarter Kelly, 0.5 = half Kelly).
                  Default 0.25 is conservative and recommended.
        max_leverage: Maximum allowed position size (default 2.0 = 200% or 2x leverage).
        lookback: Rolling window for Kelly calculation (default 252 = ~1 year daily).

    Returns:
        Series of position sizes (0.0 to max_leverage), same index as signals.
    """
    position_sizes = pd.Series(0.0, index=signals.index)

    for i in range(lookback, len(signals)):
        if signals.iloc[i] == 0:
            # No signal, no position
            position_sizes.iloc[i] = 0.0
            continue

        # Get historical window
        hist_returns = returns.iloc[i - lookback : i]
        hist_signals = signals.iloc[i - lookback : i]

        # Calculate Kelly parameters on historical data
        win_rate, win_loss_ratio, kelly_frac = calculate_kelly_parameters(hist_returns, hist_signals)

        if kelly_frac <= 0:
            # Negative Kelly = negative edge, don't trade
            position_sizes.iloc[i] = 0.0
        else:
            # Apply fractional Kelly and cap at max leverage
            size = min(fraction * kelly_frac, max_leverage)
            position_sizes.iloc[i] = size

    logger.info(
        f"Kelly sizing: fraction={fraction}, max_leverage={max_leverage}, "
        f"mean_size={position_sizes[position_sizes > 0].mean():.3f}, "
        f"max_size={position_sizes.max():.3f}"
    )

    return position_sizes


def apply_fixed_sizing(
    signals: pd.Series,
    fixed_size: float = 1.0,
) -> pd.Series:
    """Apply fixed position sizing (baseline for comparison).

    Args:
        signals: Binary signals (1 = LONG, 0 = FLAT).
        fixed_size: Fixed position size (default 1.0 = 100%).

    Returns:
        Series of position sizes (0.0 or fixed_size), same index as signals.
    """
    position_sizes = signals.astype(float) * fixed_size
    return position_sizes
