"""Tests for transaction cost model."""

import pandas as pd
import pytest

from sparky.backtest.costs import TransactionCostModel


class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_init_defaults(self):
        """Test that default initialization works."""
        model = TransactionCostModel()
        assert model.fee_pct == 0.001
        assert model.slippage_pct == 0.0002
        assert model.spread_pct == 0.0001
        assert abs(model.total_cost_pct - 0.0013) < 1e-10

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        model = TransactionCostModel(fee_pct=0.002, slippage_pct=0.0005, spread_pct=0.0003)
        assert model.fee_pct == 0.002
        assert model.slippage_pct == 0.0005
        assert model.spread_pct == 0.0003
        assert model.total_cost_pct == 0.0028

    def test_for_btc(self):
        """Test BTC pre-configured instance."""
        model = TransactionCostModel.for_btc()
        assert model.fee_pct == 0.001
        assert model.slippage_pct == 0.0002
        assert model.spread_pct == 0.0001
        # Round trip cost should be ~0.26%
        round_trip_cost = 2 * model.total_cost_pct
        assert abs(round_trip_cost - 0.0026) < 1e-6

    def test_for_eth(self):
        """Test ETH pre-configured instance."""
        model = TransactionCostModel.for_eth()
        assert model.fee_pct == 0.001
        assert model.slippage_pct == 0.0003
        assert model.spread_pct == 0.0001
        # Round trip cost should be ~0.28%
        round_trip_cost = 2 * model.total_cost_pct
        assert abs(round_trip_cost - 0.0028) < 1e-6

    def test_no_trade_no_cost(self):
        """Test that holding position incurs no costs."""
        model = TransactionCostModel()

        # Constant position (no changes)
        returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        positions = pd.Series([1.0, 1.0, 1.0, 1.0])

        returns_after = model.apply(returns, positions)

        # Only first position change should have cost
        assert returns_after.iloc[0] < returns.iloc[0]  # Entry cost
        assert returns_after.iloc[1] == returns.iloc[1]  # No change
        assert returns_after.iloc[2] == returns.iloc[2]  # No change
        assert returns_after.iloc[3] == returns.iloc[3]  # No change

    def test_position_change_incurs_cost(self):
        """Test that changing position reduces returns."""
        model = TransactionCostModel()

        # Position changes from 0 to 1 to 0
        returns = pd.Series([0.01, 0.02, 0.015])
        positions = pd.Series([1.0, 1.0, 0.0])

        returns_after = model.apply(returns, positions)

        # Entry and exit should have costs
        assert returns_after.iloc[0] < returns.iloc[0]  # Entry cost
        assert returns_after.iloc[1] == returns.iloc[1]  # No change
        assert returns_after.iloc[2] < returns.iloc[2]  # Exit cost

    def test_round_trip_cost(self):
        """Test that round trip cost is approximately correct."""
        model = TransactionCostModel.for_btc()

        # Enter and exit position
        returns = pd.Series([0.0, 0.0, 0.0])
        positions = pd.Series([1.0, 1.0, 0.0])

        returns_after = model.apply(returns, positions)

        # Total cost should be approximately 0.26% (2 * 0.0013)
        total_cost = returns.sum() - returns_after.sum()
        expected_round_trip = 2 * model.total_cost_pct

        assert abs(total_cost - expected_round_trip) < 1e-6

    def test_multiple_position_changes(self):
        """Test costs with multiple position changes."""
        model = TransactionCostModel()

        # Multiple position changes
        returns = pd.Series([0.01] * 5)
        positions = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0])

        returns_after = model.apply(returns, positions)

        # Position changes at indices 0 (entry to 1), 2 (change to -1), 4 (exit to 0)
        # Change from 1 to -1 is a change of 2
        assert returns_after.iloc[0] < returns.iloc[0]  # Entry to 1
        assert returns_after.iloc[1] == returns.iloc[1]  # No change
        assert returns_after.iloc[2] < returns.iloc[2]  # Change to -1 (cost = 2 * total_cost_pct)
        assert returns_after.iloc[3] == returns.iloc[3]  # No change
        assert returns_after.iloc[4] < returns.iloc[4]  # Exit to 0

    def test_partial_positions(self):
        """Test costs with fractional position sizes."""
        model = TransactionCostModel()

        # Fractional positions
        returns = pd.Series([0.01, 0.02, 0.015])
        positions = pd.Series([0.5, 0.5, 0.75])

        returns_after = model.apply(returns, positions)

        # Entry of 0.5 should have cost
        assert returns_after.iloc[0] < returns.iloc[0]
        # No change at index 1
        assert returns_after.iloc[1] == returns.iloc[1]
        # Increase by 0.25 should have cost
        assert returns_after.iloc[2] < returns.iloc[2]

        # Cost at index 2 should be 0.25 * total_cost_pct
        cost_at_2 = returns.iloc[2] - returns_after.iloc[2]
        expected_cost = 0.25 * model.total_cost_pct
        assert abs(cost_at_2 - expected_cost) < 1e-6

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise an error."""
        model = TransactionCostModel()

        returns = pd.Series([0.01, 0.02, 0.015])
        positions = pd.Series([1.0, 1.0])

        with pytest.raises(ValueError, match="same length"):
            model.apply(returns, positions)

    def test_empty_series(self):
        """Test behavior with empty series."""
        model = TransactionCostModel()

        returns = pd.Series([], dtype=float)
        positions = pd.Series([], dtype=float)

        returns_after = model.apply(returns, positions)
        assert len(returns_after) == 0

    def test_zero_initial_position(self):
        """Test that starting from zero position has no initial cost."""
        model = TransactionCostModel()

        # Start with zero position
        returns = pd.Series([0.01, 0.02])
        positions = pd.Series([0.0, 1.0])

        returns_after = model.apply(returns, positions)

        # No cost at index 0 (position is 0)
        assert returns_after.iloc[0] == returns.iloc[0]
        # Cost at index 1 (entering position)
        assert returns_after.iloc[1] < returns.iloc[1]
