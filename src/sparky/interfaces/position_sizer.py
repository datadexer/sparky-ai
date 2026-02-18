"""Position sizing protocol."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """Protocol for position sizing.

    Determines position size based on signal strength, market data,
    and portfolio value.
    """

    def size_position(
        self,
        signal: float,
        data: pd.DataFrame,
        portfolio_value: float,
    ) -> float:
        """Determine position size.

        Args:
            signal: Trading signal strength (-1 to 1).
            data: Current market data for context.
            portfolio_value: Current portfolio value.

        Returns:
            Position size as fraction of portfolio (0.0 to 1.0).
        """
        ...
