"""Multi-Horizon Volatility Term Structure for Regime Detection.

Research Basis:
- "Utilizing Volatility Term Structure Changes to Spot Regime Shifts" (Amberdata 2024)
- Implied volatility varies across maturities (options term structure)
- For spot trading, use realized vol at multiple horizons (7/30/90 day)
- Slope and curvature of term structure signal regime changes

Strategy Logic:
- Calculate volatility at 3 horizons: 7-day, 30-day, 90-day
- Analyze term structure:
  - CONTANGO (long > short vol): Stable regime, normal position
  - BACKWARDATION (short > long vol): Unstable regime, reduce position
  - FLAT (similar across horizons): Low uncertainty, increase position
- Regime classification by slope: (vol_90d - vol_7d) / 83 days

Target:
- Sharpe ≥0.85 (vs baseline 0.772)
- Early detection of regime shifts via term structure changes
- Avoid large drawdowns by detecting instability early
"""

import logging
import pandas as pd
import numpy as np

from sparky.models.simple_baselines import donchian_channel_strategy

logger = logging.getLogger(__name__)


def compute_volatility_term_structure(
    prices: pd.Series,
    short_window: int = 7,
    medium_window: int = 30,
    long_window: int = 90,
) -> pd.DataFrame:
    """Compute multi-horizon volatility term structure.

    Args:
        prices: Close prices (daily frequency).
        short_window: Short volatility window (default 7 days).
        medium_window: Medium volatility window (default 30 days).
        long_window: Long volatility window (default 90 days).

    Returns:
        DataFrame with columns: ["vol_7d", "vol_30d", "vol_90d", "slope", "curvature", "regime"].
        All volatilities are annualized (%).
        Slope = (vol_90d - vol_7d) / 83 (annualized vol change per day).
        Curvature = vol_30d - (vol_7d + vol_90d) / 2.
    """
    returns = prices.pct_change()

    # Compute volatilities at 3 horizons (annualized)
    vol_7d = returns.rolling(short_window).std() * np.sqrt(365)
    vol_30d = returns.rolling(medium_window).std() * np.sqrt(365)
    vol_90d = returns.rolling(long_window).std() * np.sqrt(365)

    # Compute term structure slope (change per day)
    slope = (vol_90d - vol_7d) / 83  # 90 - 7 = 83 days

    # Compute term structure curvature
    curvature = vol_30d - (vol_7d + vol_90d) / 2

    # Classify regimes based on slope
    regime = pd.Series(index=prices.index, dtype=str)
    regime[slope > 0.002] = "contango"      # Stable (long > short vol)
    regime[slope < -0.002] = "backwardation"  # Unstable (short > long vol)
    regime[(slope >= -0.002) & (slope <= 0.002)] = "flat"  # Low uncertainty

    # Forward-fill initial NaN values
    regime = regime.bfill()

    df = pd.DataFrame({
        "vol_7d": vol_7d,
        "vol_30d": vol_30d,
        "vol_90d": vol_90d,
        "slope": slope,
        "curvature": curvature,
        "regime": regime,
    }, index=prices.index)

    n_contango = (regime == "contango").sum()
    n_backwardation = (regime == "backwardation").sum()
    n_flat = (regime == "flat").sum()

    logger.info(
        f"Volatility term structure regimes: "
        f"contango={n_contango} ({n_contango/len(regime)*100:.1f}%), "
        f"backwardation={n_backwardation} ({n_backwardation/len(regime)*100:.1f}%), "
        f"flat={n_flat} ({n_flat/len(regime)*100:.1f}%)"
    )

    return df


def volatility_term_structure_donchian(
    prices: pd.Series,
    entry_period: int = 20,
    exit_period: int = 10,
    short_window: int = 7,
    medium_window: int = 30,
    long_window: int = 90,
) -> pd.Series:
    """Donchian strategy with volatility term structure position sizing.

    Position sizing based on term structure regime:
    - CONTANGO (stable): 100% position
    - FLAT (low uncertainty): 125% position (leverage if available)
    - BACKWARDATION (unstable): 50% position

    Args:
        prices: Close prices (daily frequency).
        entry_period: Donchian entry period (default 20 days).
        exit_period: Donchian exit period (default 10 days).
        short_window: Short volatility window (default 7 days).
        medium_window: Medium volatility window (default 30 days).
        long_window: Long volatility window (default 90 days).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
        Note: Position sizing is binary (0 or 1) for simplicity.
        Actual implementation would use fractional positions (0.5, 1.0, 1.25).
    """
    # Compute base Donchian signals
    base_signals = donchian_channel_strategy(prices, entry_period=entry_period, exit_period=exit_period)

    # Compute volatility term structure
    term_structure = compute_volatility_term_structure(prices, short_window, medium_window, long_window)

    # Adjust signals based on term structure regime
    adjusted_signals = base_signals.copy()

    # In BACKWARDATION (unstable), reduce exposure (50% → binary: 50% chance of FLAT)
    # For simplicity, we'll filter out half of LONG signals in backwardation
    backwardation_mask = (term_structure["regime"] == "backwardation") & (base_signals == 1)

    # Keep only signals where 7-day vol < 30-day vol (less unstable)
    # This is a proxy for "keep the less risky signals"
    unstable_long_mask = backwardation_mask & (term_structure["vol_7d"] > term_structure["vol_30d"])
    adjusted_signals[unstable_long_mask] = 0

    # In FLAT (low uncertainty), keep all signals (would be 125% in real implementation)
    # No change needed for binary signals

    n_filtered = unstable_long_mask.sum()
    n_total = len(base_signals)
    n_long_adj = adjusted_signals.sum()

    logger.info(
        f"Term Structure Donchian: {n_filtered} signals filtered (backwardation + high uncertainty)"
    )
    logger.info(
        f"Final signals: {n_long_adj} LONG ({n_long_adj/n_total*100:.1f}%), "
        f"{n_total - n_long_adj} FLAT ({(n_total - n_long_adj)/n_total*100:.1f}%)"
    )

    return adjusted_signals


def volatility_term_structure_ensemble(
    prices: pd.Series,
    short_window: int = 7,
    medium_window: int = 30,
    long_window: int = 90,
) -> pd.Series:
    """Multi-timeframe ensemble with volatility term structure filtering.

    Combines 3 Donchian channels (20/40/60) with majority voting,
    then adjusts based on volatility term structure regime.

    Args:
        prices: Close prices (daily frequency).
        short_window: Short volatility window (default 7 days).
        medium_window: Medium volatility window (default 30 days).
        long_window: Long volatility window (default 90 days).

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

    # Compute volatility term structure
    term_structure = compute_volatility_term_structure(prices, short_window, medium_window, long_window)

    # Adjust ensemble based on term structure
    adjusted_ensemble = base_ensemble.copy()

    # In BACKWARDATION, filter out unstable signals
    backwardation_mask = (term_structure["regime"] == "backwardation") & (base_ensemble == 1)
    unstable_mask = backwardation_mask & (term_structure["vol_7d"] > term_structure["vol_30d"])
    adjusted_ensemble[unstable_mask] = 0

    n_filtered = unstable_mask.sum()
    n_total = len(base_ensemble)
    n_long_adj = adjusted_ensemble.sum()

    logger.info(
        f"Term Structure Ensemble: {n_filtered} signals filtered (backwardation + high uncertainty)"
    )
    logger.info(
        f"Final ensemble: {n_long_adj} LONG ({n_long_adj/n_total*100:.1f}%), "
        f"{n_total - n_long_adj} FLAT ({(n_total - n_long_adj)/n_total*100:.1f}%)"
    )

    return adjusted_ensemble


def early_warning_indicator(
    prices: pd.Series,
    short_window: int = 7,
    medium_window: int = 30,
    long_window: int = 90,
    slope_threshold: float = 0.005,
) -> pd.Series:
    """Early warning indicator for regime shifts via term structure steepening.

    Detects when term structure slope changes rapidly, signaling potential regime shift.

    Args:
        prices: Close prices (daily frequency).
        short_window: Short volatility window (default 7 days).
        medium_window: Medium volatility window (default 30 days).
        long_window: Long volatility window (default 90 days).
        slope_threshold: Threshold for steep slope change (default 0.005).

    Returns:
        Series of warnings (1 = WARNING, 0 = NORMAL), same index as prices.
        WARNING = slope increased by >threshold in past 5 days (regime shift ahead).
    """
    term_structure = compute_volatility_term_structure(prices, short_window, medium_window, long_window)

    # Compute slope change (5-day rolling delta)
    slope_change = term_structure["slope"].diff(5)

    # Warning if slope steepened rapidly
    warnings = (slope_change.abs() > slope_threshold).astype(int)

    n_warnings = warnings.sum()
    n_total = len(warnings)

    logger.info(
        f"Early Warning Indicator: {n_warnings} warnings ({n_warnings/n_total*100:.1f}%) "
        f"for potential regime shifts"
    )

    return warnings
