"""Pydantic models for market data structures.

These types are used at module boundaries to validate data flowing
between the data layer and feature engineering.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class OHLCVCandle(BaseModel):
    """Single OHLCV candle with validation.

    All prices must be positive. High >= Low.
    Timestamps are always UTC.
    """

    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    quote_volume: Optional[float] = Field(None, ge=0)

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        """High price must be >= low price."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError(f"high ({v}) must be >= low ({info.data['low']})")
        return v


class OnChainMetric(BaseModel):
    """Single on-chain metric observation.

    Represents one data point from any on-chain data source
    (BGeometrics, CoinMetrics, Blockchain.com).
    """

    timestamp: datetime
    asset: str = Field(..., pattern=r"^(BTC|ETH)$")
    metric_name: str
    value: float
    source: str  # e.g., "bgeometrics", "coinmetrics", "blockchain_com"

    model_config = {"frozen": True}
