# DECISIONS.md — Human-Agent Communication Log

This file is the async communication channel between AK and the CEO agent.

## Format
- Agent writes questions tagged `[AGENT -> HUMAN]` with a date
- Human responds below tagged `[HUMAN -> AGENT]`
- Decisions are final once human responds

---

## Pending Decisions

### Phase 0 Complete — Informational Gate
**[AGENT -> HUMAN] 2026-02-15**

Phase 0 validation bedrock is complete. 63 tests passing across 5 test files:
- `test_returns.py`: simple returns, log returns, Sharpe, max drawdown, realized vol (14 tests)
- `test_technical.py`: RSI (Wilder's), EMA, MACD, momentum (13 tests)
- `test_cross_validation.py`: all indicators validated against pandas_ta (5 tests)
- `test_sign_conventions.py`: explicit sign convention tests preventing v1 bugs (11 tests)
- `test_activity_logger.py`: structured JSONL logging for all agents (12 tests)

Key finding during implementation: our initial RSI and EMA implementations diverged from
pandas_ta due to different initialization (SMA seed vs raw ewm). Fixed to match pandas_ta
exactly. This is exactly the kind of bug Phase 0 is designed to catch.

Ready for Phase 1 (data layer). Proceeding without waiting (informational gate).

## Resolved Decisions

### Decision 001: Data Source Architecture
**[HUMAN -> AGENT] Pre-resolved at bootstrap**
- On-chain: Fetch from BOTH BGeometrics and CoinMetrics Community, cross-validate, use best per metric
- Price/derivatives: CCXT with Binance primary, Bybit/OKX failover
- Validation reference: Blockchain.com for BTC raw metrics
- Market context: CoinGecko, 1 batch call/day
- Sentiment: DEFERRED until Phase 3 proves on-chain alpha
- Reddit collection does NOT start until models prove value

### Decision 002: Project Scope
**[HUMAN -> AGENT] Pre-resolved at bootstrap**
- BTC + ETH only. Expand only after proven edge.
- No code from previous projects.
- Experiment tracking via MLflow from day 1.

### Decision 003: Environment & Tooling
**[AUTO] 2026-02-15**
- Python 3.12 on aarch64 (DGX Spark) — plan specified 3.11, adapted to reality
- Using `uv` for package management — faster, lockfile for reproducibility, consistent across agents
- Minimum version pins (`>=`) instead of exact pins — exact pins from plan target x86/3.11
- No conda available — `uv venv` provides isolation
