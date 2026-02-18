"""Strategy protocol for trading signal generation."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol for trading strategies.

    Any class that implements generate_signals(), get_params(), and
    has a name attribute satisfies this protocol.
    """

    name: str

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data.

        Args:
            data: DataFrame with OHLCV and feature columns.

        Returns:
            Series of signals (1=long, 0=flat, -1=short).
        """
        ...

    def get_params(self) -> dict:
        """Return strategy parameters for logging/reproducibility.

        Returns:
            Dict of parameter names to values.
        """
        ...
