"""Baseline strategies that all ML models must beat.

Three simple baselines run through the walk-forward backtester:
1. BuyAndHold — 100% in asset at all times
2. SimpleMomentum — long if 30d momentum > 0, cash otherwise
3. EqualWeight — 50/50 BTC+ETH, rebalanced monthly

These establish hurdle rates for Phase 3 ML models.
"""

import numpy as np
import pandas as pd


class BuyAndHold:
    """Always long — 100% in asset.

    This is the simplest benchmark. Any ML model that can't beat
    buy-and-hold after costs is worthless.

    Implements the model protocol (fit/predict) for backtester compatibility.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op — buy and hold doesn't learn."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Always predict 1 (long)."""
        return np.ones(len(X), dtype=int)


class SimpleMomentum:
    """Long if N-day momentum > 0, cash otherwise.

    Uses the 'momentum' column from the feature matrix if available,
    otherwise computes from 'close' column.
    """

    def __init__(self, period: int = 30, momentum_col: str = "momentum"):
        self.period = period
        self.momentum_col = momentum_col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op — momentum doesn't learn from labels."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Long (1) if momentum > 0, flat (0) otherwise."""
        if self.momentum_col in X.columns:
            momentum = X[self.momentum_col]
        elif "close" in X.columns:
            momentum = X["close"].pct_change(periods=self.period)
        else:
            # Fallback: use first column
            momentum = X.iloc[:, 0].pct_change(periods=self.period)

        signals = (momentum > 0).astype(int).values.copy()
        # NaN momentum → stay flat
        signals[momentum.isna().values] = 0
        return signals


class EqualWeight:
    """50/50 BTC+ETH, always long both.

    For backtesting: this is equivalent to always-long with 50% exposure,
    since the backtester runs per-asset. The combined equity is computed
    externally by averaging the per-asset equity curves.

    When used per-asset in the backtester, it's identical to BuyAndHold
    but with 50% position size (represented as always predict 1, scaled externally).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Always predict 1 (long at half weight — scaling done externally)."""
        return np.ones(len(X), dtype=int)


def compute_equal_weight_equity(btc_equity: pd.Series, eth_equity: pd.Series) -> pd.Series:
    """Combine per-asset equity curves into 50/50 portfolio.

    Args:
        btc_equity: BTC equity curve (starting at 1.0).
        eth_equity: ETH equity curve (starting at 1.0).

    Returns:
        Combined portfolio equity curve (50% each, starting at 1.0).
    """
    # Align dates
    combined = pd.DataFrame(
        {
            "btc": btc_equity,
            "eth": eth_equity,
        }
    ).dropna()

    # 50/50 allocation
    combined["portfolio"] = 0.5 * combined["btc"] + 0.5 * combined["eth"]

    return combined["portfolio"]
