"""Pydantic models for portfolio state and trade orders.

These types represent the portfolio management layer — positions,
trade orders, and portfolio snapshots.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Position(BaseModel):
    """A single asset position."""

    asset: str = Field(..., pattern=r"^(BTC|ETH)$")
    quantity: float  # Can be 0 (no position)
    entry_price: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    value_usd: float

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in USD."""
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of entry value."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class PortfolioState(BaseModel):
    """Snapshot of portfolio at a point in time."""

    timestamp: datetime
    cash_usd: float = Field(..., ge=0)
    positions: dict[str, Position] = Field(default_factory=dict)
    total_value_usd: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_allocation(self) -> "PortfolioState":
        """Position values + cash should approximately equal total value."""
        position_value = sum(p.value_usd for p in self.positions.values())
        computed_total = position_value + self.cash_usd
        if abs(computed_total - self.total_value_usd) > 1.0:
            raise ValueError(
                f"Position values ({position_value}) + cash ({self.cash_usd}) "
                f"= {computed_total}, but total_value_usd = {self.total_value_usd}"
            )
        return self


class TradeOrder(BaseModel):
    """A trade order with cost modeling."""

    timestamp: datetime
    asset: str = Field(..., pattern=r"^(BTC|ETH)$")
    side: OrderSide
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    fee_pct: float = Field(default=0.001, ge=0)  # 0.10% default
    slippage_pct: float = Field(default=0.0002, ge=0)  # 0.02% default

    @property
    def gross_value(self) -> float:
        """Order value before costs."""
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        """Total transaction cost (fees + slippage)."""
        return self.gross_value * (self.fee_pct + self.slippage_pct)

    @property
    def net_value(self) -> float:
        """Order value after costs."""
        if self.side == OrderSide.BUY:
            return self.gross_value + self.total_cost
        return self.gross_value - self.total_cost
