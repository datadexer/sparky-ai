"""Composite on-chain regime detection.

Combines multiple on-chain signals into a single regime indicator.
"""

import pandas as pd

__all__ = ["onchain_regime_signal", "onchain_regime_with_positions"]


def onchain_regime_signal(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    threshold: float = 0.0,
    persistence_days: int = 3,
) -> pd.Series:
    """Combine multiple on-chain signals into regime signal.

    Args:
        signals: Dict mapping names to Series in [-1, 0, +1].
        weights: Dict mapping names to weights. Default: equal weights.
        threshold: Weighted sum above this -> long (1), else flat (0).
        persistence_days: Signal must hold N consecutive days before state change.

    Returns:
        Series of 1 (long) or 0 (flat).
    """
    if weights is None:
        weights = {k: 1.0 / len(signals) for k in signals}

    aligned = pd.DataFrame(signals)
    weighted_sum = sum(aligned[k] * weights[k] for k in signals)

    raw = (weighted_sum > threshold).astype(float)

    if persistence_days <= 1:
        return raw

    result = pd.Series(0.0, index=raw.index)
    current_state = 0.0
    consecutive = 0

    for i in range(len(raw)):
        if raw.iloc[i] == current_state:
            consecutive = 0
            result.iloc[i] = current_state
        else:
            consecutive += 1
            if consecutive >= persistence_days:
                current_state = raw.iloc[i]
                consecutive = 0
            result.iloc[i] = current_state

    return result


def onchain_regime_with_positions(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    threshold: float = 0.0,
    persistence_days: int = 3,
) -> pd.DataFrame:
    """Regime signal with component breakdown.

    Returns DataFrame with one column per signal, composite_score,
    regime_signal, and position.
    """
    if weights is None:
        weights = {k: 1.0 / len(signals) for k in signals}

    df = pd.DataFrame(signals)
    df["composite_score"] = sum(df[k] * weights[k] for k in signals)
    df["regime_signal"] = onchain_regime_signal(signals, weights, threshold, persistence_days)
    df["position"] = df["regime_signal"]
    return df
