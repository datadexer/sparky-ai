"""Adaptive Lookback Windows for Donchian Channels.

Research Basis:
- "Donchian breakout adaptive parameters regime-dependent crypto trading" (2024)
- High-volatility markets require shorter periods for quicker signals
- Stable markets require longer periods to avoid false breakouts
- Bitcoin exhibits persistent volatility regimes (Fidelity 2024)

Strategy Logic:
- Compute volatility regime (LOW/MEDIUM/HIGH based on 30-day realized vol)
- Adjust Donchian lookback periods based on regime:
  - HIGH vol (>60%): SHORT windows (10/20/30) to react faster
  - MEDIUM vol (30-60%): STANDARD windows (20/40/60)
  - LOW vol (<30%): LONG windows (30/60/90) to avoid whipsaws

Target:
- Sharpe ≥0.85 (vs baseline 0.772)
- Reduce whipsaws in low-vol periods
- Capture breakouts faster in high-vol periods
"""

import logging
import pandas as pd
import numpy as np

from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.features.regime_indicators import compute_volatility_regime

logger = logging.getLogger(__name__)


def adaptive_lookback_donchian(
    prices: pd.Series,
    vol_window: int = 30,
    aggressive_periods: tuple[int, int] = (10, 5),
    standard_periods: tuple[int, int] = (20, 10),
    conservative_periods: tuple[int, int] = (30, 15),
) -> pd.Series:
    """Adaptive Lookback Donchian with regime-dependent periods.

    Args:
        prices: Close prices (daily frequency).
        vol_window: Volatility regime window (default 30 days).
        aggressive_periods: (entry, exit) for HIGH vol regime (default 10/5).
        standard_periods: (entry, exit) for MEDIUM vol regime (default 20/10).
        conservative_periods: (entry, exit) for LOW vol regime (default 30/15).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute volatility regime
    regime = compute_volatility_regime(prices, window=vol_window, frequency="1d")

    # Initialize signals
    signals = pd.Series(0, index=prices.index, dtype=int)

    # Track position state and current periods
    in_position = False
    current_entry_period = standard_periods[0]
    current_exit_period = standard_periods[1]

    for i in range(len(prices)):
        if i < vol_window:
            # Not enough data yet
            signals.iloc[i] = 0
            continue

        # Get regime-specific periods
        current_regime = regime.iloc[i]
        if current_regime == "high":
            current_entry_period, current_exit_period = aggressive_periods
        elif current_regime == "low":
            current_entry_period, current_exit_period = conservative_periods
        else:  # medium
            current_entry_period, current_exit_period = standard_periods

        # Compute Donchian channels with adaptive periods
        if i >= current_entry_period:
            upper_channel = prices.iloc[i - current_entry_period:i].max()
        else:
            upper_channel = prices.iloc[:i].max()

        if i >= current_exit_period:
            lower_channel = prices.iloc[i - current_exit_period:i].min()
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

    logger.info(
        f"Adaptive Lookback Donchian: {n_long} LONG ({n_long/n_total*100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long)/n_total*100:.1f}%)"
    )

    return signals


def adaptive_lookback_ensemble(
    prices: pd.Series,
    vol_window: int = 30,
) -> pd.Series:
    """Multi-timeframe ensemble with adaptive lookback windows.

    Combines 3 Donchian channels with regime-dependent periods:
    - HIGH vol: (10/20/30) - faster reaction
    - MEDIUM vol: (20/40/60) - standard
    - LOW vol: (30/60/90) - avoid whipsaws

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
        if current_regime == "high":
            short_entry, short_exit = 10, 5
            medium_entry, medium_exit = 20, 10
            long_entry, long_exit = 30, 15
        elif current_regime == "low":
            short_entry, short_exit = 30, 15
            medium_entry, medium_exit = 60, 30
            long_entry, long_exit = 90, 45
        else:  # medium
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
        f"Adaptive Lookback Ensemble: {n_long} LONG ({n_long/n_total*100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long)/n_total*100:.1f}%)"
    )

    return ensemble
