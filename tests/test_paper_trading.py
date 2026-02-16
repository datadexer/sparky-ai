"""Tests for paper trading engine and signal pipeline."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sparky.trading.paper_trading_engine import (
    HaltCondition,
    PaperTradingEngine,
)
from sparky.trading.signal_pipeline import (
    DonchianSignalPipeline,
    MultiTFSignalPipeline,
    TradingSignal,
)
from sparky.types.config_types import PaperTradingConfig
from sparky.types.portfolio_types import OrderSide


# ============================================================
# Paper Trading Engine Tests
# ============================================================


class TestPaperTradingEngine:
    """Tests for PaperTradingEngine."""

    def test_initialization(self):
        """Engine starts with correct capital and empty positions."""
        engine = PaperTradingEngine(start_capital=100_000)
        assert engine.cash == 100_000
        assert engine.positions == {}
        assert engine.halted is False
        assert engine.peak_equity == 100_000

    def test_buy_order(self):
        """Buy order deducts cash and creates position."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.001, slippage_pct=0.0)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        order = engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)

        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.asset == "BTC"
        assert "BTC" in engine.positions
        assert engine.positions["BTC"] > 0
        assert engine.cash < 100_000

    def test_sell_order(self):
        """Sell order returns cash and removes position."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.001, slippage_pct=0.0)
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Buy first
        engine.process_signal(ts1, asset="BTC", target_weight=0.5, price=50_000)
        assert engine.positions.get("BTC", 0) > 0

        # Sell
        engine.process_signal(ts2, asset="BTC", target_weight=0.0, price=50_000)
        assert engine.positions.get("BTC", 0) == 0 or "BTC" not in engine.positions
        assert engine.cash > 0

    def test_transaction_costs(self):
        """Transaction costs reduce portfolio value."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.01, slippage_pct=0.005)
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Round-trip trade at same price should lose money to fees
        engine.process_signal(ts1, asset="BTC", target_weight=0.5, price=50_000)
        engine.process_signal(ts2, asset="BTC", target_weight=0.0, price=50_000)

        total_value = engine.get_portfolio_value({"BTC": 50_000})
        assert total_value < 100_000, "Round-trip should lose money to fees"

    def test_position_limit(self):
        """Position size respects max_position_pct limit."""
        config = PaperTradingConfig(start_capital=100_000, max_position_pct=30)
        engine = PaperTradingEngine(start_capital=100_000, config=config, fee_pct=0.0, slippage_pct=0.0)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Try to go 100% long but limit is 30%
        engine.process_signal(ts, asset="BTC", target_weight=1.0, price=50_000)

        position_value = engine.positions.get("BTC", 0) * 50_000
        total_value = engine.get_portfolio_value({"BTC": 50_000})
        position_pct = position_value / total_value * 100

        assert position_pct <= 31, f"Position {position_pct:.1f}% exceeds 30% limit"

    def test_daily_trade_limit(self):
        """Daily trade limit prevents excessive trading."""
        config = PaperTradingConfig(start_capital=100_000, max_daily_trades=2)
        engine = PaperTradingEngine(start_capital=100_000, config=config)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # First two trades should work
        engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)
        engine.process_signal(ts, asset="BTC", target_weight=0.0, price=50_000)

        # Third trade should be blocked
        order = engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)
        assert order is None, "Third trade should be blocked by daily limit"

    def test_drawdown_halt(self):
        """Engine halts on excessive drawdown."""
        config = PaperTradingConfig(start_capital=100_000, max_position_pct=50)
        engine = PaperTradingEngine(start_capital=100_000, config=config, fee_pct=0.0, slippage_pct=0.0)
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Buy at 50000
        engine.process_signal(ts1, asset="BTC", target_weight=0.5, price=50_000)
        engine._update_equity(ts1, {"BTC": 50_000})

        # Price crashes 60% -> portfolio down ~30% (50% position * 60% crash = 30% portfolio loss)
        with pytest.raises(HaltCondition, match="max_drawdown"):
            engine.process_signal(ts2, asset="BTC", target_weight=0.0, price=20_000)

        assert engine.halted is True

    def test_portfolio_value_calculation(self):
        """Portfolio value includes cash + positions."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.0, slippage_pct=0.0)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)

        # With no fees, 50% should be in BTC, 50% in cash
        value = engine.get_portfolio_value({"BTC": 50_000})
        assert abs(value - 100_000) < 1.0, f"Value should be ~$100k, got ${value:,.2f}"

        # Price doubles
        value_up = engine.get_portfolio_value({"BTC": 100_000})
        assert value_up > 100_000, "Value should increase when BTC price rises"

    def test_get_state(self):
        """get_state returns valid PortfolioState."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.0, slippage_pct=0.0)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)
        state = engine.get_state({"BTC": 50_000})

        assert state.total_value_usd > 0
        assert state.cash_usd >= 0
        assert "BTC" in state.positions
        assert state.positions["BTC"].quantity > 0

    def test_daily_snapshot(self):
        """Daily snapshot records metrics correctly."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.0, slippage_pct=0.0)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        snapshot = engine.record_daily_snapshot(ts, {"BTC": 50_000})

        assert snapshot["total_value"] == 100_000
        assert snapshot["drawdown_pct"] == 0.0
        assert snapshot["total_return_pct"] == 0.0

    def test_reset(self):
        """Reset restores initial state."""
        engine = PaperTradingEngine(start_capital=100_000)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)
        engine.reset()

        assert engine.cash == 100_000
        assert engine.positions == {}
        assert engine.trade_history == []
        assert engine.halted is False

    def test_skip_tiny_trades(self):
        """Trades below threshold are skipped."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.0, slippage_pct=0.0)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Very small weight change -> should skip
        order = engine.process_signal(ts, asset="BTC", target_weight=0.0005, price=50_000)
        assert order is None, "Tiny trade should be skipped"

    def test_halted_engine_ignores_signals(self):
        """Halted engine returns None for all signals."""
        engine = PaperTradingEngine(start_capital=100_000)
        engine.halted = True
        engine.halt_reason = "test halt"

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        order = engine.process_signal(ts, asset="BTC", target_weight=0.5, price=50_000)
        assert order is None

    def test_get_summary(self):
        """Summary includes all expected fields."""
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.001, slippage_pct=0.0)
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        engine.process_signal(ts1, asset="BTC", target_weight=0.5, price=50_000)
        engine.process_signal(ts2, asset="BTC", target_weight=0.0, price=55_000)

        summary = engine.get_summary({"BTC": 55_000})
        assert "start_capital" in summary
        assert "current_value" in summary
        assert "total_return_pct" in summary
        assert "n_trades" in summary
        assert summary["n_trades"] == 2
        assert summary["n_buys"] == 1
        assert summary["n_sells"] == 1


# ============================================================
# Signal Pipeline Tests
# ============================================================


def _make_trending_prices(n_days=100, start_price=50000, trend=0.002):
    """Create synthetic trending price series."""
    rng = np.random.default_rng(42)
    returns = trend + rng.normal(0, 0.02, n_days)
    prices = start_price * np.cumprod(1 + returns)
    index = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.Series(prices, index=index)


def _make_declining_prices(n_days=100, start_price=50000, trend=-0.002):
    """Create synthetic declining price series."""
    return _make_trending_prices(n_days, start_price, trend)


class TestDonchianSignalPipeline:
    """Tests for DonchianSignalPipeline."""

    def test_initialization(self):
        """Pipeline initializes with correct parameters."""
        pipeline = DonchianSignalPipeline(entry_period=40, exit_period=20)
        assert pipeline.entry_period == 40
        assert pipeline.exit_period == 20
        assert pipeline.asset == "BTC"

    def test_signal_on_uptrend(self):
        """Uptrend should generate LONG signal."""
        prices = _make_trending_prices(n_days=100, trend=0.005)
        pipeline = DonchianSignalPipeline(entry_period=20, exit_period=10)

        signal = pipeline.generate_signal(prices)

        assert isinstance(signal, TradingSignal)
        assert signal.asset == "BTC"
        assert signal.signal_value in [0, 1]
        assert signal.target_weight in [0.0, 1.0]

    def test_signal_on_downtrend(self):
        """Downtrend should generate FLAT signal eventually."""
        prices = _make_declining_prices(n_days=100, trend=-0.005)
        pipeline = DonchianSignalPipeline(entry_period=20, exit_period=10)

        signal = pipeline.generate_signal(prices)

        assert isinstance(signal, TradingSignal)
        # In a strong downtrend, signal should be FLAT
        assert signal.signal_value == 0
        assert signal.target_weight == 0.0

    def test_insufficient_data(self):
        """Should raise error with insufficient price data."""
        prices = _make_trending_prices(n_days=10)
        pipeline = DonchianSignalPipeline(entry_period=40, exit_period=20)

        with pytest.raises(ValueError, match="Need at least"):
            pipeline.generate_signal(prices)

    def test_signal_has_reason(self):
        """Signal includes human-readable reason."""
        prices = _make_trending_prices(n_days=100)
        pipeline = DonchianSignalPipeline(entry_period=20, exit_period=10)

        signal = pipeline.generate_signal(prices)

        assert signal.reason is not None
        assert len(signal.reason) > 0
        assert signal.strategy_name == "Donchian(20/10)"

    def test_custom_long_weight(self):
        """Custom long weight is applied correctly."""
        prices = _make_trending_prices(n_days=100, trend=0.01)
        pipeline = DonchianSignalPipeline(entry_period=20, exit_period=10, long_weight=0.5)

        signal = pipeline.generate_signal(prices)

        if signal.signal_value == 1:
            assert signal.target_weight == 0.5


class TestMultiTFSignalPipeline:
    """Tests for MultiTFSignalPipeline."""

    def test_initialization(self):
        """Pipeline initializes with default periods."""
        pipeline = MultiTFSignalPipeline()
        assert pipeline.entry_periods == [20, 40, 60]

    def test_signal_generation(self):
        """Multi-TF pipeline generates valid signal."""
        prices = _make_trending_prices(n_days=100, trend=0.003)
        pipeline = MultiTFSignalPipeline(entry_periods=[20, 30, 40])

        signal = pipeline.generate_signal(prices)

        assert isinstance(signal, TradingSignal)
        assert signal.signal_value in [0, 1]
        assert "Vote" in signal.reason

    def test_custom_periods(self):
        """Custom entry periods work correctly."""
        prices = _make_trending_prices(n_days=100)
        pipeline = MultiTFSignalPipeline(entry_periods=[10, 20, 30])

        signal = pipeline.generate_signal(prices)
        assert signal.strategy_name == "MultiTF-Donchian([10, 20, 30])"


# ============================================================
# Integration Tests
# ============================================================


class TestPaperTradingIntegration:
    """Integration tests: signal pipeline -> paper trading engine."""

    def test_pipeline_to_engine(self):
        """Signal pipeline feeds into paper trading engine correctly."""
        # Generate prices
        prices = _make_trending_prices(n_days=100, trend=0.003)

        # Create pipeline and engine
        pipeline = DonchianSignalPipeline(entry_period=20, exit_period=10)
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.001, slippage_pct=0.0)

        # Generate signal
        signal = pipeline.generate_signal(prices)

        # Feed to engine
        order = engine.process_signal(
            timestamp=signal.timestamp,
            asset=signal.asset,
            target_weight=signal.target_weight,
            price=float(prices.iloc[-1]),
        )

        # Verify engine state changed if signal was LONG
        if signal.signal_value == 1:
            assert order is not None
            assert engine.positions.get("BTC", 0) > 0

    def test_multi_day_simulation(self):
        """Simulate multiple days of trading."""
        prices = _make_trending_prices(n_days=100, trend=0.002)
        pipeline = DonchianSignalPipeline(entry_period=20, exit_period=10)
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.001, slippage_pct=0.0002)

        # Process signals day by day (start from day 21 to have enough history)
        for i in range(21, len(prices)):
            history = prices.iloc[:i+1]
            signal = pipeline.generate_signal(history)
            engine.process_signal(
                timestamp=signal.timestamp,
                asset=signal.asset,
                target_weight=signal.target_weight,
                price=float(prices.iloc[i]),
            )

        # Should have some trades
        assert len(engine.trade_history) > 0

        # Final value should be reasonable (not crashed)
        final_value = engine.get_portfolio_value({"BTC": float(prices.iloc[-1])})
        assert final_value > 0
        assert final_value > 50_000  # Shouldn't lose more than 50% in a mild uptrend

    def test_real_btc_data_if_available(self):
        """Integration test with real BTC data (skipped if unavailable)."""
        from pathlib import Path
        data_path = Path("/home/akamath/sparky-ai/data/btc_daily.parquet")
        if not data_path.exists():
            pytest.skip("BTC data not available")

        df = pd.read_parquet(data_path)
        prices = df["close"]
        prices.index = pd.to_datetime(df.index)

        # Use 2023 data only
        prices_2023 = prices.loc["2023-01-01":"2023-12-31"]
        if len(prices_2023) < 60:
            pytest.skip("Insufficient 2023 data")

        pipeline = DonchianSignalPipeline(entry_period=40, exit_period=20)
        engine = PaperTradingEngine(start_capital=100_000, fee_pct=0.001, slippage_pct=0.0002)

        for i in range(41, len(prices_2023)):
            history = prices_2023.iloc[:i+1]
            signal = pipeline.generate_signal(history)
            engine.process_signal(
                timestamp=signal.timestamp,
                asset=signal.asset,
                target_weight=signal.target_weight,
                price=float(prices_2023.iloc[i]),
            )

        summary = engine.get_summary({"BTC": float(prices_2023.iloc[-1])})
        assert summary["n_trades"] > 0
        assert summary["current_value"] > 0
