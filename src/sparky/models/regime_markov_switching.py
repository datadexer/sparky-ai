"""Markov-Switching Donchian Strategy.

Research Basis:
- "The risks of trading on cryptocurrencies: A regime-switching approach" (Taylor & Francis 2023)
- "Regime switching forecasting for cryptocurrencies" (Springer Digital Finance 2024)
- State-dependent approaches outperform pure time-dependent models
- Train separate strategies for each regime, switch based on current state

Strategy Logic:
- Detect volatility regime (LOW/MEDIUM/HIGH) via threshold-based classification
- Train/optimize 3 regime-specific Donchian variants:
  - AGGRESSIVE (LOW vol): Short periods (15/5) for fast entries
  - STANDARD (MEDIUM vol): Medium periods (20/10) balanced
  - CONSERVATIVE (HIGH vol): Long periods (40/20) avoid whipsaws
- Switch strategies based on current regime

Why This Works:
- Different market conditions require different strategies
- Low-vol markets: trends are slow, need aggressive entry
- High-vol markets: whipsaws are common, need conservative exit
- Research shows regime-switching models outperform static models

Target:
- Sharpe ≥0.85 (vs baseline 0.772)
- Reduce drawdowns in high-vol periods via conservative parameters
- Capture trends faster in low-vol periods via aggressive parameters
"""

import logging
import pandas as pd
import numpy as np

from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.features.regime_indicators import compute_volatility_regime

logger = logging.getLogger(__name__)


def markov_switching_donchian(
    prices: pd.Series,
    vol_window: int = 30,
    aggressive_params: tuple[int, int] = (15, 5),
    standard_params: tuple[int, int] = (20, 10),
    conservative_params: tuple[int, int] = (40, 20),
) -> pd.Series:
    """Markov-Switching Donchian with regime-specific parameters.

    Switches between 3 Donchian strategies based on volatility regime:
    - LOW vol → AGGRESSIVE (15/5): Fast entries, tight stops
    - MEDIUM vol → STANDARD (20/10): Balanced
    - HIGH vol → CONSERVATIVE (40/20): Slow entries, wide stops

    Args:
        prices: Close prices (daily frequency).
        vol_window: Volatility regime window (default 30 days).
        aggressive_params: (entry, exit) for LOW vol (default 15/5).
        standard_params: (entry, exit) for MEDIUM vol (default 20/10).
        conservative_params: (entry, exit) for HIGH vol (default 40/20).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute volatility regime
    regime = compute_volatility_regime(prices, window=vol_window, frequency="1d")

    # Initialize signals
    signals = pd.Series(0, index=prices.index, dtype=int)

    # Track position state
    in_position = False
    current_params = standard_params

    for i in range(len(prices)):
        if i < vol_window:
            # Not enough data yet
            signals.iloc[i] = 0
            continue

        # Get regime-specific parameters
        current_regime = regime.iloc[i]
        if current_regime == "low":
            current_params = aggressive_params
        elif current_regime == "high":
            current_params = conservative_params
        else:  # medium
            current_params = standard_params

        entry_period, exit_period = current_params

        # Compute Donchian channels with regime-specific periods
        if i >= entry_period:
            upper_channel = prices.iloc[i - entry_period:i].max()
        else:
            upper_channel = prices.iloc[:i].max()

        if i >= exit_period:
            lower_channel = prices.iloc[i - exit_period:i].min()
        else:
            lower_channel = prices.iloc[:i].min()

        current_price = prices.iloc[i]

        if not in_position:
            # Check for entry: price breaks above upper channel
            if i > 0 and current_price >= upper_channel:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            # In position: check for exit
            if current_price <= lower_channel:
                in_position = False
                signals.iloc[i] = 0
            else:
                # Hold position
                signals.iloc[i] = 1

    n_long = signals.sum()
    n_total = len(signals)

    # Analyze regime distribution
    n_low = (regime == "low").sum()
    n_medium = (regime == "medium").sum()
    n_high = (regime == "high").sum()

    logger.info(
        f"Markov-Switching Donchian: {n_long} LONG ({n_long/n_total*100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long)/n_total*100:.1f}%)"
    )
    logger.info(
        f"Regime distribution: "
        f"LOW={n_low} ({n_low/n_total*100:.1f}%, params={aggressive_params}), "
        f"MEDIUM={n_medium} ({n_medium/n_total*100:.1f}%, params={standard_params}), "
        f"HIGH={n_high} ({n_high/n_total*100:.1f}%, params={conservative_params})"
    )

    return signals


def markov_switching_ensemble(
    prices: pd.Series,
    vol_window: int = 30,
) -> pd.Series:
    """Multi-timeframe ensemble with Markov-Switching parameters.

    Combines 3 Donchian channels with regime-dependent periods:
    - LOW vol: (15/5, 25/10, 35/15) - aggressive
    - MEDIUM vol: (20/10, 40/20, 60/30) - standard
    - HIGH vol: (40/20, 80/40, 120/60) - conservative

    Uses majority voting (LONG if 2+ agree).

    Args:
        prices: Close prices (daily frequency).
        vol_window: Volatility regime window (default 30 days).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute volatility regime
    regime = compute_volatility_regime(prices, window=vol_window, frequency="1d")

    # Initialize signals for 3 timeframes
    signals_short = pd.Series(0, index=prices.index, dtype=int)
    signals_medium = pd.Series(0, index=prices.index, dtype=int)
    signals_long = pd.Series(0, index=prices.index, dtype=int)

    # Track position states
    in_position_short = False
    in_position_medium = False
    in_position_long = False

    for i in range(len(prices)):
        if i < vol_window:
            # Not enough data yet
            signals_short.iloc[i] = 0
            signals_medium.iloc[i] = 0
            signals_long.iloc[i] = 0
            continue

        # Get regime-specific periods
        current_regime = regime.iloc[i]
        if current_regime == "low":
            # Aggressive: fast entries
            short_entry, short_exit = 15, 5
            medium_entry, medium_exit = 25, 10
            long_entry, long_exit = 35, 15
        elif current_regime == "high":
            # Conservative: slow entries, avoid whipsaws
            short_entry, short_exit = 40, 20
            medium_entry, medium_exit = 80, 40
            long_entry, long_exit = 120, 60
        else:  # medium
            # Standard: balanced
            short_entry, short_exit = 20, 10
            medium_entry, medium_exit = 40, 20
            long_entry, long_exit = 60, 30

        current_price = prices.iloc[i]

        # Short timeframe
        if i >= short_entry:
            upper_short = prices.iloc[i - short_entry:i].max()
            lower_short = prices.iloc[i - short_exit:i].min() if i >= short_exit else prices.iloc[:i].min()

            if not in_position_short:
                if i > 0 and current_price >= upper_short:
                    in_position_short = True
                    signals_short.iloc[i] = 1
                else:
                    signals_short.iloc[i] = 0
            else:
                if current_price <= lower_short:
                    in_position_short = False
                    signals_short.iloc[i] = 0
                else:
                    signals_short.iloc[i] = 1

        # Medium timeframe
        if i >= medium_entry:
            upper_medium = prices.iloc[i - medium_entry:i].max()
            lower_medium = prices.iloc[i - medium_exit:i].min() if i >= medium_exit else prices.iloc[:i].min()

            if not in_position_medium:
                if i > 0 and current_price >= upper_medium:
                    in_position_medium = True
                    signals_medium.iloc[i] = 1
                else:
                    signals_medium.iloc[i] = 0
            else:
                if current_price <= lower_medium:
                    in_position_medium = False
                    signals_medium.iloc[i] = 0
                else:
                    signals_medium.iloc[i] = 1

        # Long timeframe
        if i >= long_entry:
            upper_long = prices.iloc[i - long_entry:i].max()
            lower_long = prices.iloc[i - long_exit:i].min() if i >= long_exit else prices.iloc[:i].min()

            if not in_position_long:
                if i > 0 and current_price >= upper_long:
                    in_position_long = True
                    signals_long.iloc[i] = 1
                else:
                    signals_long.iloc[i] = 0
            else:
                if current_price <= lower_long:
                    in_position_long = False
                    signals_long.iloc[i] = 0
                else:
                    signals_long.iloc[i] = 1

    # Majority voting (LONG if 2+ agree)
    ensemble = (signals_short + signals_medium + signals_long) >= 2
    ensemble = ensemble.astype(int)

    n_long = ensemble.sum()
    n_total = len(ensemble)

    logger.info(
        f"Markov-Switching Ensemble: {n_long} LONG ({n_long/n_total*100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long)/n_total*100:.1f}%)"
    )

    return ensemble


def regime_transition_probability(
    regime_history: pd.Series,
) -> pd.DataFrame:
    """Compute Markov transition probabilities between regimes.

    Analyzes historical regime transitions to estimate:
    - P(low → low), P(low → medium), P(low → high)
    - P(medium → low), P(medium → medium), P(medium → high)
    - P(high → low), P(high → medium), P(high → high)

    Useful for understanding regime persistence.

    Args:
        regime_history: Series with regime labels ("low", "medium", "high").

    Returns:
        DataFrame with transition probabilities (3x3 matrix).
        Rows = current state, Columns = next state.
    """
    # Count transitions
    transitions = {
        ("low", "low"): 0, ("low", "medium"): 0, ("low", "high"): 0,
        ("medium", "low"): 0, ("medium", "medium"): 0, ("medium", "high"): 0,
        ("high", "low"): 0, ("high", "medium"): 0, ("high", "high"): 0,
    }

    for i in range(len(regime_history) - 1):
        current_regime = regime_history.iloc[i]
        next_regime = regime_history.iloc[i + 1]
        key = (current_regime, next_regime)
        if key in transitions:
            transitions[key] += 1

    # Compute probabilities
    transition_matrix = pd.DataFrame(
        index=["low", "medium", "high"],
        columns=["low", "medium", "high"],
        dtype=float,
    )

    for current_state in ["low", "medium", "high"]:
        total = sum(transitions[(current_state, next_state)] for next_state in ["low", "medium", "high"])
        if total > 0:
            for next_state in ["low", "medium", "high"]:
                transition_matrix.loc[current_state, next_state] = transitions[(current_state, next_state)] / total
        else:
            # No transitions observed, assume uniform
            transition_matrix.loc[current_state, :] = 1/3

    logger.info("Markov Transition Probabilities:")
    logger.info(f"\n{transition_matrix}")

    # Log persistence (diagonal elements)
    logger.info(
        f"Regime persistence: "
        f"LOW={transition_matrix.loc['low', 'low']:.3f}, "
        f"MEDIUM={transition_matrix.loc['medium', 'medium']:.3f}, "
        f"HIGH={transition_matrix.loc['high', 'high']:.3f}"
    )

    return transition_matrix
