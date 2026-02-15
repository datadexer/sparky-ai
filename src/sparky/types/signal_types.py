"""Pydantic models for signals and predictions.

These types represent model outputs flowing from the model layer
to the portfolio/trading layer.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Model prediction for a single asset and horizon.

    probability: predicted probability of price going up (0-1)
    direction: +1 (long), -1 (short), 0 (neutral)
    confidence: model's confidence in the prediction (0-1)
    """

    timestamp: datetime
    asset: str = Field(..., pattern=r"^(BTC|ETH)$")
    horizon_days: int = Field(..., gt=0)
    probability: float = Field(..., ge=0.0, le=1.0)
    direction: int = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    model_version: Optional[str] = None


class Signal(BaseModel):
    """Trading signal derived from one or more predictions.

    target_weight: desired portfolio weight for this asset (-1 to 1)
    Positive = long, negative = short, 0 = no position.
    """

    timestamp: datetime
    asset: str = Field(..., pattern=r"^(BTC|ETH)$")
    target_weight: float = Field(..., ge=-1.0, le=1.0)
    prediction: Prediction
    reason: str  # Human-readable explanation
