# Phase 4: Paper Trading & Live

## Purpose
Deploy validated models to paper trading for out-of-sample confirmation,
then transition to live trading with real capital and proper risk controls.

## Tasks

| Task | Description |
|------|-------------|
| `paper_trading_engine` | Simulated execution engine tracking fills, slippage, and portfolio state |
| `signal_pipeline` | End-to-end pipeline: data fetch → features → model → signal → order |
| `monitoring_alerts` | Real-time monitoring: model drift, data staleness, drawdown alerts |
| `paper_trading_launch` | Deploy and run paper trading for minimum 4-week evaluation period |
| `live_trading_framework` | Real exchange integration via CCXT with position limits and kill switch |
| `live_trading_deployment` | Deploy to live with small initial capital and conservative risk limits |

## Completion Criteria
- Paper trading runs unattended for 4+ weeks without crashes
- Paper results are consistent with backtest expectations (within 2 sigma)
- Monitoring catches and alerts on simulated failure scenarios
- Live framework has working kill switch and maximum position limits
- Live deployment starts with <5% of intended final allocation

## Human Gate
**Type: Approve (twice)**
1. Human approves paper trading results before any live capital is deployed
2. Human approves live trading config (position limits, capital) before launch
