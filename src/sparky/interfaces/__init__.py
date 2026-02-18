"""Interface protocols for Sparky AI components.

These protocols define the contracts between system components,
enabling loose coupling and testability. All protocols use
typing.Protocol for structural subtyping (duck typing with
type checker support).

Usage:
    from sparky.interfaces import StrategyProtocol, BacktesterProtocol
"""

from sparky.interfaces.backtester import BacktesterProtocol
from sparky.interfaces.data_feed import DataFeedProtocol
from sparky.interfaces.feature_pipeline import FeaturePipelineProtocol
from sparky.interfaces.position_sizer import PositionSizerProtocol
from sparky.interfaces.strategy import StrategyProtocol

__all__ = [
    "StrategyProtocol",
    "BacktesterProtocol",
    "DataFeedProtocol",
    "FeaturePipelineProtocol",
    "PositionSizerProtocol",
]
