"""Backtester protocol for strategy evaluation."""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class BacktesterProtocol(Protocol):
    """Protocol for backtesting engines.

    Defines the interface for running a model through historical data
    and producing backtest results.
    """

    def run(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series,
        **kwargs: Any,
    ) -> Any:
        """Run a backtest.

        Args:
            model: Model implementing fit/predict.
            X: Feature matrix with DatetimeIndex.
            y: Target labels with DatetimeIndex.
            returns: Period returns for equity calculation.
            **kwargs: Additional arguments (e.g., cost_model, asset).

        Returns:
            BacktestResult or equivalent result object.
        """
        ...
