"""Signal pipeline for generating trading signals from strategies.

Connects strategy implementations (Donchian, etc.) to the paper trading engine.
Fetches latest prices, generates signals, and converts to target weights.

Usage:
    pipeline = DonchianSignalPipeline(entry_period=40, exit_period=20)
    signal = pipeline.generate_signal(prices)
    # signal.target_weight is 0.0 (flat) or 1.0 (long)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from sparky.models.simple_baselines import donchian_channel_strategy

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """A trading signal with target weight and metadata."""

    timestamp: datetime
    asset: str
    target_weight: float  # 0.0 = flat, 1.0 = fully long
    signal_value: int  # Raw signal (0 or 1)
    reason: str
    strategy_name: str


class DonchianSignalPipeline:
    """Signal pipeline for single-TF Donchian channel strategy.

    Generates binary LONG/FLAT signals based on Donchian channel breakouts.
    The target weight is 0.0 (flat) or a configurable long weight.

    Args:
        entry_period: Donchian entry lookback (default 40).
        exit_period: Donchian exit lookback (default 20).
        long_weight: Portfolio weight when LONG (default 1.0 = 100%).
        asset: Asset symbol (default "BTC").
    """

    def __init__(
        self,
        entry_period: int = 40,
        exit_period: int = 20,
        long_weight: float = 1.0,
        asset: str = "BTC",
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.long_weight = long_weight
        self.asset = asset
        self.strategy_name = f"Donchian({entry_period}/{exit_period})"

    def generate_signal(self, prices: pd.Series) -> TradingSignal:
        """Generate a trading signal from price history.

        Args:
            prices: Historical close prices with DatetimeIndex.
                Must include at least entry_period+1 days of data.

        Returns:
            TradingSignal with target weight.
        """
        if len(prices) < self.entry_period + 1:
            raise ValueError(
                f"Need at least {self.entry_period + 1} prices, got {len(prices)}"
            )

        # Generate full signal series
        signals = donchian_channel_strategy(
            prices,
            entry_period=self.entry_period,
            exit_period=self.exit_period,
        )

        # Latest signal (today's close generates tomorrow's position)
        latest_signal = int(signals.iloc[-1])
        latest_date = signals.index[-1]

        # Determine target weight
        target_weight = self.long_weight if latest_signal == 1 else 0.0

        # Build reason string
        current_price = prices.iloc[-1]
        upper_channel = prices.rolling(window=self.entry_period).max().iloc[-2]
        lower_channel = prices.rolling(window=self.exit_period).min().iloc[-2]

        if latest_signal == 1:
            reason = (
                f"LONG: price ${current_price:,.0f} broke above "
                f"{self.entry_period}-day high ${upper_channel:,.0f}"
            )
        else:
            reason = (
                f"FLAT: price ${current_price:,.0f} below "
                f"{self.exit_period}-day low ${lower_channel:,.0f} "
                f"(or no breakout)"
            )

        signal = TradingSignal(
            timestamp=latest_date.to_pydatetime() if hasattr(latest_date, "to_pydatetime") else latest_date,
            asset=self.asset,
            target_weight=target_weight,
            signal_value=latest_signal,
            reason=reason,
            strategy_name=self.strategy_name,
        )

        logger.info(f"Signal: {signal.strategy_name} -> {signal.reason}")
        return signal


class MultiTFSignalPipeline:
    """Signal pipeline for multi-timeframe Donchian ensemble.

    Combines multiple Donchian strategies via majority vote.

    Args:
        entry_periods: List of entry periods (default [20, 40, 60]).
        long_weight: Portfolio weight when LONG.
        asset: Asset symbol.
    """

    def __init__(
        self,
        entry_periods: list[int] | None = None,
        long_weight: float = 1.0,
        asset: str = "BTC",
    ):
        self.entry_periods = entry_periods or [20, 40, 60]
        self.long_weight = long_weight
        self.asset = asset
        self.strategy_name = f"MultiTF-Donchian({self.entry_periods})"

    def generate_signal(self, prices: pd.Series) -> TradingSignal:
        """Generate a trading signal from majority vote of multiple timeframes.

        Args:
            prices: Historical close prices.

        Returns:
            TradingSignal with target weight.
        """
        max_period = max(self.entry_periods)
        if len(prices) < max_period + 1:
            raise ValueError(f"Need at least {max_period + 1} prices")

        # Generate signals for each timeframe
        votes = []
        for entry_period in self.entry_periods:
            exit_period = entry_period // 2
            signals = donchian_channel_strategy(
                prices,
                entry_period=entry_period,
                exit_period=exit_period,
            )
            votes.append(int(signals.iloc[-1]))

        # Majority vote
        long_votes = sum(votes)
        majority = len(self.entry_periods) / 2
        latest_signal = 1 if long_votes > majority else 0
        target_weight = self.long_weight if latest_signal == 1 else 0.0

        reason = f"Vote: {long_votes}/{len(self.entry_periods)} LONG -> {'LONG' if latest_signal else 'FLAT'}"

        return TradingSignal(
            timestamp=prices.index[-1].to_pydatetime() if hasattr(prices.index[-1], "to_pydatetime") else prices.index[-1],
            asset=self.asset,
            target_weight=target_weight,
            signal_value=latest_signal,
            reason=reason,
            strategy_name=self.strategy_name,
        )
