"""Trading module — paper and live trading infrastructure."""

from sparky.trading.paper_trading_engine import HaltCondition, PaperTradingEngine
from sparky.trading.signal_pipeline import (
    DonchianSignalPipeline,
    MultiTFSignalPipeline,
    TradingSignal,
)

__all__ = [
    "HaltCondition",
    "PaperTradingEngine",
    "DonchianSignalPipeline",
    "MultiTFSignalPipeline",
    "TradingSignal",
]
