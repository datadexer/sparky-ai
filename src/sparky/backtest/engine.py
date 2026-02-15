"""Walk-forward backtesting engine.

This module implements expanding window walk-forward backtesting with embargo
periods to prevent information leakage. The backtester simulates realistic
trading conditions by:

1. Using expanding windows (not sliding) to mimic real-world training
2. Adding embargo periods between train and test to prevent lookahead bias
3. Computing returns based on next-day execution (T+1 open)
4. Supporting transaction costs via a pluggable cost model

Critical timing convention (from CODEBASE_PLAN.md):
- Day T close → features computed
- Day T+1 open → trade EXECUTES
- Target: did price go UP from T+1 open to T+1+N close?

The backtester expects:
- X: DataFrame with DatetimeIndex, features as columns
- y: Series with DatetimeIndex, binary labels (1=long, 0=flat)
- Model: duck-typed with .fit(X_train, y_train) and .predict(X_test) -> array

The backtester returns metrics per fold and an overall equity curve.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from sparky.features.returns import annualized_sharpe, max_drawdown, realized_volatility


class ModelProtocol(Protocol):
    """Duck-typed model interface expected by the backtester."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on features X and labels y."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict signals (1=long, 0=flat) for features X."""
        ...


class CostModelProtocol(Protocol):
    """Duck-typed cost model interface for transaction costs."""

    def compute_cost(self, position_change: int, asset: str) -> float:
        """Compute fractional cost for a position change.

        Args:
            position_change: Change in position (e.g., 0->1, 1->0).
            asset: Asset symbol (e.g., "BTC", "ETH").

        Returns:
            Fractional cost (e.g., 0.001 for 0.1%).
        """
        ...


@dataclass
class BacktestResult:
    """Results from a walk-forward backtest run.

    Attributes:
        trades: List of trade dicts with keys: date, signal, position_before, position_after.
        equity_curve: Series indexed by date, starting at 1.0.
        per_fold_metrics: List of dicts with keys: fold, train_start, train_end, test_start,
                         test_end, sharpe, total_return, accuracy, num_trades.
        fold_count: Number of test folds completed.
    """

    trades: list[dict]
    equity_curve: pd.Series
    per_fold_metrics: list[dict]
    fold_count: int


class WalkForwardBacktester:
    """Expanding window walk-forward backtester with embargo period.

    Configuration:
        train_min_length: Minimum number of training rows required.
        embargo_days: Gap between train end and test start (prevents leakage).
        test_length_days: Length of each test period.
        step_days: How far to advance the test window each fold.

    The backtester requires at least 5 test folds to produce results.
    """

    def __init__(
        self,
        train_min_length: int,
        embargo_days: int = 7,
        test_length_days: int = 30,
        step_days: int = 30,
    ):
        """Initialize the walk-forward backtester.

        Args:
            train_min_length: Minimum number of training rows.
            embargo_days: Days between train end and test start.
            test_length_days: Length of each test period in days.
            step_days: Step size for advancing test window.
        """
        self.train_min_length = train_min_length
        self.embargo_days = embargo_days
        self.test_length_days = test_length_days
        self.step_days = step_days

    def run(
        self,
        model: ModelProtocol,
        X: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series,
        cost_model: CostModelProtocol | None = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            model: Model implementing .fit() and .predict().
            X: Feature matrix with DatetimeIndex.
            y: Target labels (1=long, 0=flat) with DatetimeIndex.
            returns: Daily returns series with DatetimeIndex (for computing equity).
            cost_model: Optional cost model for transaction costs.

        Returns:
            BacktestResult with trades, equity curve, metrics, and fold count.

        Raises:
            ValueError: If less than 5 test folds can be generated.
        """
        # Ensure all inputs are aligned and sorted
        X = X.sort_index()
        y = y.sort_index()
        returns = returns.sort_index()

        # Align all data to common dates
        common_dates = X.index.intersection(y.index).intersection(returns.index)
        X = X.loc[common_dates]
        y = y.loc[common_dates]
        returns = returns.loc[common_dates]

        # Generate folds
        folds = self._generate_folds(X.index)
        if len(folds) < 5:
            raise ValueError(
                f"Insufficient data for 5 test folds. Got {len(folds)} folds. "
                f"Need more historical data or reduce test_length_days/step_days."
            )

        # Run backtest across all folds
        all_trades = []
        all_predictions = pd.Series(index=X.index, dtype=float)
        per_fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]

            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Store predictions
            all_predictions.loc[test_idx] = predictions

            # Compute fold metrics
            accuracy = np.mean(predictions == y_test.values)
            fold_returns = returns.loc[test_idx]
            position_series = pd.Series(predictions, index=test_idx)

            # Compute fold equity curve (for sharpe/return calculation)
            fold_equity = self._compute_equity_curve(
                fold_returns, position_series, cost_model=None  # No costs for fold metrics
            )
            fold_total_return = fold_equity.iloc[-1] - 1.0
            fold_pnl_returns = fold_equity.pct_change().dropna()
            fold_sharpe = annualized_sharpe(fold_pnl_returns) if len(fold_pnl_returns) > 1 else 0.0

            # Count trades (position changes)
            position_changes = position_series.diff().fillna(position_series.iloc[0])
            num_trades = (position_changes != 0).sum()

            per_fold_metrics.append({
                "fold": fold_idx,
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "test_start": test_idx[0],
                "test_end": test_idx[-1],
                "sharpe": fold_sharpe,
                "total_return": fold_total_return,
                "accuracy": accuracy,
                "num_trades": int(num_trades),
            })

        # Compute overall equity curve with costs
        test_dates = all_predictions.dropna().index
        position_series = pd.Series(all_predictions.loc[test_dates], index=test_dates)
        equity_curve = self._compute_equity_curve(
            returns.loc[test_dates], position_series, cost_model
        )

        # Generate trade log
        all_trades = self._generate_trade_log(position_series)

        return BacktestResult(
            trades=all_trades,
            equity_curve=equity_curve,
            per_fold_metrics=per_fold_metrics,
            fold_count=len(folds),
        )

    def _generate_folds(self, dates: pd.DatetimeIndex) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate expanding window folds with embargo period.

        Args:
            dates: DatetimeIndex of all available dates.

        Returns:
            List of (train_dates, test_dates) tuples.
        """
        folds = []
        test_start_offset = self.train_min_length + self.embargo_days

        while True:
            # Define test window
            test_end_offset = test_start_offset + self.test_length_days
            if test_end_offset > len(dates):
                break

            test_dates = dates[test_start_offset:test_end_offset]
            if len(test_dates) == 0:
                break

            # Define train window (expanding: from start to embargo boundary)
            train_end_offset = test_start_offset - self.embargo_days
            train_dates = dates[:train_end_offset]

            if len(train_dates) >= self.train_min_length:
                folds.append((train_dates, test_dates))

            # Step forward
            test_start_offset += self.step_days

        return folds

    def _compute_equity_curve(
        self,
        returns: pd.Series,
        positions: pd.Series,
        cost_model: CostModelProtocol | None,
    ) -> pd.Series:
        """Compute equity curve from returns and positions.

        The equity curve starts at 1.0 and compounds:
        - When position=1 (long), apply the daily return
        - When position=0 (flat), no return
        - On position changes, apply transaction cost if cost_model provided

        Args:
            returns: Daily returns series.
            positions: Position series (1=long, 0=flat).
            cost_model: Optional cost model.

        Returns:
            Equity curve starting at 1.0 with index shifted to include day before first position.
        """
        # Build equity curve starting at 1.0
        equity_values = [1.0]  # Initial equity
        dates = [positions.index[0] - pd.Timedelta(days=1)]  # Day before first position

        current_equity = 1.0
        prev_position = 0

        for date in positions.index:
            current_position = int(positions.loc[date])
            daily_return = returns.loc[date]

            # Apply transaction cost if position changed
            if current_position != prev_position and cost_model is not None:
                cost = cost_model.compute_cost(
                    position_change=abs(current_position - prev_position),
                    asset="BTC",  # Assume BTC for now
                )
                current_equity *= (1 - cost)

            # Apply return if in position
            if current_position == 1:
                current_equity *= (1 + daily_return)

            equity_values.append(current_equity)
            dates.append(date)
            prev_position = current_position

        return pd.Series(equity_values, index=pd.DatetimeIndex(dates))

    def _generate_trade_log(self, positions: pd.Series) -> list[dict]:
        """Generate trade log from position series.

        Args:
            positions: Position series (1=long, 0=flat).

        Returns:
            List of trade dicts.
        """
        trades = []
        prev_position = 0

        for date, signal in positions.items():
            current_position = int(signal)
            if current_position != prev_position:
                trades.append({
                    "date": date,
                    "signal": current_position,
                    "position_before": prev_position,
                    "position_after": current_position,
                })
            prev_position = current_position

        return trades
