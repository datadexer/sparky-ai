"""Transaction cost model for backtesting."""

import pandas as pd


class TransactionCostModel:
    """Model transaction costs including fees, slippage, and spread.

    total_cost_pct is the one-way (per-change) cost applied to each position
    change. A round trip (enter + exit) incurs 2 position changes, so the
    total round trip cost is 2 × total_cost_pct.

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
        self.round_trip_cost = 2 * self.total_cost_pct

    @classmethod
    def standard(cls) -> "TransactionCostModel":
        """Return the standard cost model: 30 bps (0.30%) per side.

        Reflects Coinbase Advanced Trade with limit orders at modest volume,
        or DEX on L2 (Base/Arbitrum). Realistic without assuming discounts.
        A round trip (enter + exit) costs 60 bps (0.60%).

        Returns:
            TransactionCostModel with 30 bps per side.
        """
        return cls(fee_pct=0.0015, slippage_pct=0.001, spread_pct=0.0005)

    @classmethod
    def stress_test(cls) -> "TransactionCostModel":
        """Return the stress-test cost model: 50 bps (0.50%) per side.

        Worst case: Coinbase market orders at lowest tier.
        A round trip (enter + exit) costs 100 bps (1.0%).
        Research agents should run winners at both standard and stress-test.

        Returns:
            TransactionCostModel with 50 bps per side.
        """
        return cls(fee_pct=0.001, slippage_pct=0.003, spread_pct=0.001)

    @classmethod
    def futures(cls) -> "TransactionCostModel":
        """Futures cost model: 7.5 bps (0.075%) per side — Coinbase perps."""
        return cls(fee_pct=0.00075, slippage_pct=0.0, spread_pct=0.0)

    @classmethod
    def spot_maker(cls) -> "TransactionCostModel":
        """Spot maker cost model: 15 bps (0.15%) per side — Coinbase limit orders."""
        return cls(fee_pct=0.0015, slippage_pct=0.0, spread_pct=0.0)

    @classmethod
    def for_btc(cls) -> "TransactionCostModel":
        """Return the default cost model for Bitcoin (15 bps spot maker)."""
        return cls.spot_maker()

    @classmethod
    def for_eth(cls) -> "TransactionCostModel":
        """Return the default cost model for Ethereum (15 bps spot maker)."""
        return cls.spot_maker()

    def compute_cost(self, position_change: int, asset: str) -> float:
        """Compute fractional cost for a position change.

        Used by WalkForwardBacktester._compute_equity_curve() via CostModelProtocol.

        Args:
            position_change: Absolute size of position change (e.g., 1 for 0->1 or 1->0).
            asset: Asset symbol (unused — cost params are set at construction).

        Returns:
            Fractional cost (e.g., 0.0013 for 0.13%).
        """
        return abs(position_change) * self.total_cost_pct

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
