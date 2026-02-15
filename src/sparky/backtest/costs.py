"""Transaction cost model for backtesting."""

import pandas as pd


class TransactionCostModel:
    """Model transaction costs including fees, slippage, and spread.

    Position changes incur costs on both sides of the trade (entry and exit).
    Total round trip cost = 2 * (fee + slippage + spread).

    Args:
        fee_pct: Trading fee as a fraction (e.g., 0.001 = 0.1%)
        slippage_pct: Slippage as a fraction (e.g., 0.0002 = 0.02%)
        spread_pct: Bid-ask spread cost as a fraction (e.g., 0.0001 = 0.01%)
    """

    def __init__(self, fee_pct: float = 0.001, slippage_pct: float = 0.0002, spread_pct: float = 0.0001):
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct
        self.total_cost_pct = fee_pct + slippage_pct + spread_pct

    @classmethod
    def for_btc(cls) -> "TransactionCostModel":
        """Return a pre-configured instance for Bitcoin.

        Returns:
            TransactionCostModel with BTC defaults:
            - fee=0.001 (0.1%)
            - slippage=0.0002 (0.02%)
            - spread=0.0001 (0.01%)
            - Round trip cost: ~0.26%
        """
        return cls(fee_pct=0.001, slippage_pct=0.0002, spread_pct=0.0001)

    @classmethod
    def for_eth(cls) -> "TransactionCostModel":
        """Return a pre-configured instance for Ethereum.

        Returns:
            TransactionCostModel with ETH defaults:
            - fee=0.001 (0.1%)
            - slippage=0.0003 (0.03%)
            - spread=0.0001 (0.01%)
            - Round trip cost: ~0.28%
        """
        return cls(fee_pct=0.001, slippage_pct=0.0003, spread_pct=0.0001)

    def apply(self, returns: pd.Series, positions: pd.Series) -> pd.Series:
        """Apply transaction costs to returns based on position changes.

        Costs are applied when positions change. The cost is proportional to
        the absolute value of the position change, as both increasing and
        decreasing positions incur costs.

        Args:
            returns: Series of returns (e.g., log returns or simple returns)
            positions: Series of positions (e.g., 1 for long, 0 for flat, -1 for short)
                Must have the same index as returns

        Returns:
            Series of returns after transaction costs
        """
        if len(returns) != len(positions):
            raise ValueError("Returns and positions must have the same length")

        # Calculate position changes (absolute value)
        position_changes = positions.diff().abs()

        # First position counts as a change (entry into position)
        if len(position_changes) > 0:
            position_changes.iloc[0] = abs(positions.iloc[0])

        # Calculate costs: each position change incurs the total cost percentage
        costs = position_changes * self.total_cost_pct

        # Subtract costs from returns
        returns_after_costs = returns - costs

        return returns_after_costs
