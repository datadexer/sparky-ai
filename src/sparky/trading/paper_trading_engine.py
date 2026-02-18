"""Paper trading engine for simulated strategy execution.

Manages portfolio state, executes trades based on signals, tracks P&L,
and enforces trading rules (position limits, halt conditions).

Usage:
    engine = PaperTradingEngine(start_capital=100_000)
    engine.process_signal(timestamp, asset="BTC", target_weight=1.0, price=67000.0)
    state = engine.get_state()
    print(f"Total value: ${state.total_value_usd:,.2f}")
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sparky.types.config_types import PaperTradingConfig
from sparky.types.portfolio_types import (
    OrderSide,
    PortfolioState,
    Position,
    TradeOrder,
)

logger = logging.getLogger(__name__)


class HaltCondition(Exception):
    """Raised when an auto-halt condition is triggered."""

    def __init__(self, reason: str, metric_value: float, threshold: float):
        self.reason = reason
        self.metric_value = metric_value
        self.threshold = threshold
        super().__init__(f"HALT: {reason} (value={metric_value:.2f}, threshold={threshold:.2f})")


class PaperTradingEngine:
    """Simulated trading engine with full portfolio management.

    Features:
    - Portfolio state tracking (cash, positions, equity curve)
    - Signal-to-order conversion with position sizing
    - Transaction cost modeling (fees + slippage)
    - Auto-halt on drawdown, daily loss, or rule violations
    - Trade logging to JSONL for audit trail
    - Daily P&L snapshots
    """

    def __init__(
        self,
        start_capital: float = 100_000.0,
        config: Optional[PaperTradingConfig] = None,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0002,
        log_path: Optional[Path] = None,
    ):
        """Initialize paper trading engine.

        Args:
            start_capital: Initial cash in USD.
            config: Paper trading config (from trading_rules.yaml).
            fee_pct: Exchange fee per trade (default 0.1%).
            slippage_pct: Estimated slippage per trade (default 0.02%).
            log_path: Path for trade log JSONL file.
        """
        self.start_capital = start_capital
        self.config = config or PaperTradingConfig(start_capital=start_capital)
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        # Portfolio state
        self.cash = start_capital
        self.positions: dict[str, float] = {}  # asset -> quantity
        self.entry_prices: dict[str, float] = {}  # asset -> avg entry price

        # Tracking
        self.equity_history: list[dict] = []  # timestamp, total_value
        self.trade_history: list[dict] = []
        self.daily_pnl: list[dict] = []
        self.peak_equity = start_capital
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.trades_today: int = 0
        self.current_date: Optional[datetime] = None

        # Logging
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PaperTradingEngine initialized: ${start_capital:,.0f} capital, "
            f"fee={fee_pct * 100:.2f}%, slippage={slippage_pct * 100:.3f}%"
        )

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value at current prices.

        Args:
            current_prices: Dict of asset -> current price.

        Returns:
            Total portfolio value in USD.
        """
        position_value = sum(qty * current_prices.get(asset, 0.0) for asset, qty in self.positions.items() if qty > 0)
        return self.cash + position_value

    def get_state(self, current_prices: dict[str, float]) -> PortfolioState:
        """Get current portfolio state snapshot.

        Args:
            current_prices: Dict of asset -> current price.

        Returns:
            PortfolioState with all positions and values.
        """
        positions = {}
        for asset, qty in self.positions.items():
            if qty > 0 and asset in current_prices:
                price = current_prices[asset]
                positions[asset] = Position(
                    asset=asset,
                    quantity=qty,
                    entry_price=self.entry_prices.get(asset, price),
                    current_price=price,
                    value_usd=qty * price,
                )

        total_value = self.get_portfolio_value(current_prices)

        return PortfolioState(
            timestamp=datetime.now(timezone.utc),
            cash_usd=self.cash,
            positions=positions,
            total_value_usd=total_value,
        )

    def get_drawdown(self, current_prices: dict[str, float]) -> float:
        """Calculate current drawdown from peak.

        Returns:
            Drawdown as positive percentage (e.g., 10.0 means 10% drawdown).
        """
        current_value = self.get_portfolio_value(current_prices)
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - current_value) / self.peak_equity * 100)

    def _check_halt_conditions(self, current_prices: dict[str, float]) -> None:
        """Check if any auto-halt conditions are triggered.

        Raises:
            HaltCondition if a halt trigger is met.
        """
        if self.halted:
            return

        # Drawdown check
        drawdown = self.get_drawdown(current_prices)
        max_dd = getattr(self.config, "max_drawdown_pct", 25) if hasattr(self.config, "max_drawdown_pct") else 25
        if drawdown >= max_dd:
            self.halted = True
            self.halt_reason = f"Max drawdown exceeded: {drawdown:.1f}% >= {max_dd}%"
            raise HaltCondition("max_drawdown", drawdown, max_dd)

        # Daily trade limit
        max_trades = self.config.max_daily_trades
        if self.trades_today >= max_trades:
            logger.warning(f"Daily trade limit reached ({max_trades})")

    def _reset_daily_counters(self, timestamp: datetime) -> None:
        """Reset daily counters if new day."""
        trade_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if self.current_date != trade_date:
            self.trades_today = 0
            self.current_date = trade_date

    def process_signal(
        self,
        timestamp: datetime,
        asset: str,
        target_weight: float,
        price: float,
    ) -> Optional[TradeOrder]:
        """Process a trading signal and execute if appropriate.

        Converts a target weight (0.0 = flat, 1.0 = fully long) into
        a trade order, respecting position limits and halt conditions.

        Args:
            timestamp: Signal timestamp.
            asset: Asset symbol ("BTC" or "ETH").
            target_weight: Desired portfolio weight [0.0, 1.0].
            price: Current asset price.

        Returns:
            TradeOrder if a trade was executed, None if no action needed.

        Raises:
            HaltCondition if auto-halt is triggered.
        """
        if self.halted:
            logger.warning(f"Engine halted ({self.halt_reason}), ignoring signal")
            return None

        self._reset_daily_counters(timestamp)
        current_prices = {asset: price}
        # Include other held positions at their last known price
        for held_asset in self.positions:
            if held_asset not in current_prices and held_asset in self.entry_prices:
                current_prices[held_asset] = self.entry_prices[held_asset]

        self._check_halt_conditions(current_prices)

        total_value = self.get_portfolio_value(current_prices)

        # Clamp target weight to position limits
        max_weight = self.config.max_position_pct / 100.0
        target_weight = max(0.0, min(target_weight, max_weight))

        # Calculate target and current positions
        target_value = total_value * target_weight
        current_qty = self.positions.get(asset, 0.0)
        current_value = current_qty * price

        # Calculate trade needed
        trade_value = target_value - current_value

        # Skip tiny trades (< $100 or < 0.1% of portfolio)
        if abs(trade_value) < max(100.0, total_value * 0.001):
            return None

        # Check daily trade limit
        if self.trades_today >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached, skipping signal")
            return None

        # Execute trade
        if trade_value > 0:
            # BUY
            order = self._execute_buy(timestamp, asset, trade_value, price)
        else:
            # SELL
            order = self._execute_sell(timestamp, asset, abs(trade_value), price)

        if order:
            self.trades_today += 1
            self._update_equity(timestamp, current_prices)
            self._log_trade(order)

        return order

    def _execute_buy(self, timestamp: datetime, asset: str, value: float, price: float) -> Optional[TradeOrder]:
        """Execute a buy order.

        Args:
            timestamp: Order timestamp.
            asset: Asset to buy.
            value: Target USD value to buy.
            price: Current price.

        Returns:
            TradeOrder if executed, None if insufficient cash.
        """
        # Calculate quantity including costs
        total_cost_rate = self.fee_pct + self.slippage_pct
        effective_price = price * (1 + total_cost_rate)
        quantity = value / effective_price

        # Check cash availability
        total_cost = quantity * effective_price
        if total_cost > self.cash:
            # Buy what we can afford
            quantity = self.cash / effective_price
            total_cost = self.cash

        if quantity <= 0:
            return None

        # Update state
        old_qty = self.positions.get(asset, 0.0)
        old_cost = old_qty * self.entry_prices.get(asset, price)
        new_cost = quantity * price

        self.positions[asset] = old_qty + quantity
        # Weighted average entry price
        if old_qty + quantity > 0:
            self.entry_prices[asset] = (old_cost + new_cost) / (old_qty + quantity)
        self.cash -= total_cost

        order = TradeOrder(
            timestamp=timestamp,
            asset=asset,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            fee_pct=self.fee_pct,
            slippage_pct=self.slippage_pct,
        )

        logger.info(
            f"BUY {quantity:.6f} {asset} @ ${price:,.2f} "
            f"(cost: ${total_cost:,.2f}, fees: ${quantity * price * total_cost_rate:,.2f})"
        )

        self.trade_history.append(
            {
                "timestamp": str(timestamp),
                "asset": asset,
                "side": "buy",
                "quantity": quantity,
                "price": price,
                "value": quantity * price,
                "fees": quantity * price * total_cost_rate,
                "cash_after": self.cash,
            }
        )

        return order

    def _execute_sell(self, timestamp: datetime, asset: str, value: float, price: float) -> Optional[TradeOrder]:
        """Execute a sell order.

        Args:
            timestamp: Order timestamp.
            asset: Asset to sell.
            value: Target USD value to sell.
            price: Current price.

        Returns:
            TradeOrder if executed, None if no position to sell.
        """
        current_qty = self.positions.get(asset, 0.0)
        if current_qty <= 0:
            return None

        # Calculate quantity to sell
        quantity = min(value / price, current_qty)
        if quantity <= 0:
            return None

        # Calculate proceeds after costs
        total_cost_rate = self.fee_pct + self.slippage_pct
        proceeds = quantity * price * (1 - total_cost_rate)

        # Update state
        self.positions[asset] = current_qty - quantity
        self.cash += proceeds

        # Clean up zero positions
        if self.positions[asset] < 1e-10:
            del self.positions[asset]
            if asset in self.entry_prices:
                del self.entry_prices[asset]

        order = TradeOrder(
            timestamp=timestamp,
            asset=asset,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
            fee_pct=self.fee_pct,
            slippage_pct=self.slippage_pct,
        )

        logger.info(
            f"SELL {quantity:.6f} {asset} @ ${price:,.2f} "
            f"(proceeds: ${proceeds:,.2f}, fees: ${quantity * price * total_cost_rate:,.2f})"
        )

        self.trade_history.append(
            {
                "timestamp": str(timestamp),
                "asset": asset,
                "side": "sell",
                "quantity": quantity,
                "price": price,
                "value": quantity * price,
                "fees": quantity * price * total_cost_rate,
                "cash_after": self.cash,
            }
        )

        return order

    def _update_equity(self, timestamp: datetime, current_prices: dict[str, float]) -> None:
        """Update equity tracking after a trade."""
        total_value = self.get_portfolio_value(current_prices)
        self.peak_equity = max(self.peak_equity, total_value)

        self.equity_history.append(
            {
                "timestamp": str(timestamp),
                "total_value": total_value,
                "cash": self.cash,
                "drawdown_pct": self.get_drawdown(current_prices),
            }
        )

    def record_daily_snapshot(self, timestamp: datetime, current_prices: dict[str, float]) -> dict:
        """Record end-of-day portfolio snapshot.

        Args:
            timestamp: Snapshot timestamp.
            current_prices: Dict of asset -> current price.

        Returns:
            Dict with daily metrics.
        """
        total_value = self.get_portfolio_value(current_prices)
        self.peak_equity = max(self.peak_equity, total_value)

        prev_value = self.equity_history[-1]["total_value"] if self.equity_history else self.start_capital
        daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
        total_return = (total_value - self.start_capital) / self.start_capital

        snapshot = {
            "timestamp": str(timestamp),
            "total_value": total_value,
            "cash": self.cash,
            "positions": {
                asset: {"qty": qty, "price": current_prices.get(asset, 0.0)}
                for asset, qty in self.positions.items()
                if qty > 0
            },
            "daily_return_pct": daily_return * 100,
            "total_return_pct": total_return * 100,
            "drawdown_pct": self.get_drawdown(current_prices),
            "peak_equity": self.peak_equity,
            "n_trades_today": self.trades_today,
        }

        self.daily_pnl.append(snapshot)
        self._update_equity(timestamp, current_prices)

        return snapshot

    def _log_trade(self, order: TradeOrder) -> None:
        """Log trade to JSONL file."""
        if not self.log_path:
            return

        record = {
            "timestamp": str(order.timestamp),
            "asset": order.asset,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": order.price,
            "gross_value": order.gross_value,
            "total_cost": order.total_cost,
            "net_value": order.net_value,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_summary(self, current_prices: dict[str, float]) -> dict:
        """Get summary statistics for the trading session.

        Args:
            current_prices: Dict of asset -> current price.

        Returns:
            Dict with summary statistics.
        """
        total_value = self.get_portfolio_value(current_prices)
        total_return = (total_value - self.start_capital) / self.start_capital

        n_trades = len(self.trade_history)
        n_buys = sum(1 for t in self.trade_history if t["side"] == "buy")
        n_sells = sum(1 for t in self.trade_history if t["side"] == "sell")
        total_fees = sum(t["fees"] for t in self.trade_history)

        return {
            "start_capital": self.start_capital,
            "current_value": total_value,
            "total_return_pct": total_return * 100,
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": max((s.get("drawdown_pct", 0) for s in self.daily_pnl), default=0),
            "n_trades": n_trades,
            "n_buys": n_buys,
            "n_sells": n_sells,
            "total_fees": total_fees,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.cash = self.start_capital
        self.positions = {}
        self.entry_prices = {}
        self.equity_history = []
        self.trade_history = []
        self.daily_pnl = []
        self.peak_equity = self.start_capital
        self.halted = False
        self.halt_reason = None
        self.trades_today = 0
        self.current_date = None
        logger.info("Paper trading engine reset")
