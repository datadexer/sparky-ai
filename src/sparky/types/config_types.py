"""Pydantic models for validated configuration schemas.

These types validate YAML configs at load time, catching
misconfigurations before they propagate into calculations.
"""

from typing import Optional

from pydantic import BaseModel, Field


class AutoHaltConfig(BaseModel):
    """Auto-halt triggers for paper/live trading."""

    max_drawdown_pct: float = Field(..., gt=0, le=100)
    max_daily_loss_pct: float = Field(..., gt=0, le=100)


class PaperTradingConfig(BaseModel):
    """Paper trading configuration from trading_rules.yaml."""

    enabled: bool = True
    start_capital: float = Field(default=100_000, gt=0)
    minimum_duration_days: int = Field(default=90, gt=0)
    max_position_pct: float = Field(default=50, gt=0, le=100)
    max_leverage: float = Field(default=1.0, ge=1.0)
    max_daily_trades: int = Field(default=10, gt=0)


class CostConfig(BaseModel):
    """Transaction cost model parameters."""

    exchange_fee_pct: float = Field(default=0.10, ge=0)
    slippage_btc_pct: float = Field(default=0.02, ge=0)
    slippage_eth_pct: float = Field(default=0.03, ge=0)
    spread_estimate_pct: float = Field(default=0.01, ge=0)


class ResearchStandardsConfig(BaseModel):
    """Research quality standards from trading_rules.yaml."""

    min_pvalue: float = Field(default=0.05, gt=0, lt=1)
    min_backtest_folds: int = Field(default=5, gt=0)
    multi_seed_max_std: float = Field(default=0.3, gt=0)
    max_features_per_model: int = Field(default=20, gt=0)
    max_hyperparameter_combos: int = Field(default=50, gt=0)
    require_out_of_sample: bool = True
    embargo_days: int = Field(default=7, ge=0)
    require_leakage_test: bool = True
    leakage_accuracy_threshold: float = Field(default=0.55, gt=0.5, lt=1)


class SystemConfig(BaseModel):
    """Top-level system configuration."""

    project_name: str = "sparky-ai"
    version: str = "0.1.0"
    assets: list[str] = Field(default=["BTC", "ETH"])
    history_start: str = "2017-01-01"


class TradingRulesConfig(BaseModel):
    """Top-level trading rules configuration."""

    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    research_standards: Optional[ResearchStandardsConfig] = None
    costs: Optional[CostConfig] = None
