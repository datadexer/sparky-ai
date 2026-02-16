"""Regime-Filtered Donchian Channel Strategy.

Addresses catastrophic failure in high-volatility bear markets (2022Q2: Sharpe -3.534).

Strategy Logic:
- Compute Donchian breakout signals normally (20/10, 40/20, 60/30)
- Filter signals by volatility regime:
  - HIGH vol (>60%): FLAT only (avoid whipsaws)
  - MEDIUM/LOW vol: Normal Donchian signals

Hypothesis:
- Multi-Timeframe Ensemble works great in trending markets (2019-2020, 2023)
- Fails catastrophically in choppy high-vol periods (2022Q2-Q3)
- Filtering out high-vol periods should improve robustness

Target:
- Fix 2022 bear market (Sharpe -1.902 → closer to 0)
- Maintain performance in bull markets
- Walk-forward mean Sharpe ≥ 1.0
"""

import logging
import pandas as pd

from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.features.regime_indicators import compute_volatility_regime

logger = logging.getLogger(__name__)


def regime_filtered_donchian(
    prices: pd.Series,
    entry_period: int = 20,
    exit_period: int = 10,
    vol_window: int = 30,
    filter_high_vol: bool = True,
) -> pd.Series:
    """Regime-filtered Donchian channel strategy.

    Args:
        prices: Close prices (daily frequency).
        entry_period: Donchian entry period (default 20 days).
        exit_period: Donchian exit period (default 10 days).
        vol_window: Volatility regime window (default 30 days).
        filter_high_vol: If True, force FLAT in HIGH volatility (default True).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute base Donchian signals
    base_signals = donchian_channel_strategy(prices, entry_period=entry_period, exit_period=exit_period)

    # Compute volatility regime
    regime = compute_volatility_regime(prices, window=vol_window, frequency="1d")

    # Filter signals by regime
    filtered_signals = base_signals.copy()

    if filter_high_vol:
        # Force FLAT in HIGH volatility periods
        high_vol_mask = (regime == "high")
        filtered_signals[high_vol_mask] = 0

        n_filtered = high_vol_mask.sum()
        n_total = len(base_signals)
        pct_filtered = n_filtered / n_total * 100

        logger.info(f"Regime filter: {n_filtered}/{n_total} days ({pct_filtered:.1f}%) forced FLAT (HIGH vol)")

    return filtered_signals


def regime_filtered_ensemble(
    prices: pd.Series,
    vol_window: int = 30,
    filter_high_vol: bool = True,
) -> pd.Series:
    """Multi-timeframe ensemble with regime filtering.

    Combines 3 Donchian channels (20/40/60 day) with majority voting,
    then filters out HIGH volatility periods.

    Args:
        prices: Close prices (daily frequency).
        vol_window: Volatility regime window (default 30 days).
        filter_high_vol: If True, force FLAT in HIGH volatility (default True).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Compute individual Donchian signals
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)

    # Majority voting (LONG if 2+ agree)
    base_ensemble = (signals_20 + signals_40 + signals_60) >= 2
    base_ensemble = base_ensemble.astype(int)

    # Compute volatility regime
    regime = compute_volatility_regime(prices, window=vol_window, frequency="1d")

    # Filter by regime
    filtered_ensemble = base_ensemble.copy()

    if filter_high_vol:
        high_vol_mask = (regime == "high")
        filtered_ensemble[high_vol_mask] = 0

        n_filtered = high_vol_mask.sum()
        n_total = len(base_ensemble)
        pct_filtered = n_filtered / n_total * 100

        logger.info(f"Ensemble regime filter: {n_filtered}/{n_total} days ({pct_filtered:.1f}%) forced FLAT (HIGH vol)")

    return filtered_ensemble
