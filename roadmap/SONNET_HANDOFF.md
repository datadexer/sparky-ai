# SONNET HANDOFF — Phases 0-2 Architecture & Context

Written by CEO agent (Opus) at Phase 2 completion. This document gives the
Phase 3+ agent full context to continue without re-discovery.

## 1. Architectural Decisions and WHY

### Data Pipeline: Parquet-First, Multi-Source
- **Why Parquet**: Column-oriented, fast reads, metadata support, smaller than CSV.
  Parquet stores metadata in the file itself (via PyArrow schema metadata), so
  we never lose track of source, date range, or row count.
- **Why multi-source**: BGeometrics has rate limits (8 req/hour free tier).
  CoinMetrics Community is fallback. Blockchain.com is validation reference.
  Source selector picks the best source per metric automatically.
- **Why incremental fetches**: BGeometrics quota is precious. Always use
  `store.get_last_timestamp()` → fetch delta. Never re-fetch existing data.

### Features: Registry Pattern with valid_from
- **Why registry**: Central catalog prevents duplicate features, enforces metadata
  (lookback, valid_from, data_source). The matrix builder uses this to correctly
  align features across assets with different data availability.
- **Why valid_from**: ETH features have protocol-dependent validity dates.
  EIP-1559 features are meaningless before Aug 2021. Staking features before
  Sep 2022. The registry returns NaN before valid_from — not an error.

### Backtester: Walk-Forward with Embargo
- **Why expanding window**: More training data per fold = better model estimation.
  Sliding window throws away valuable early data for no reason.
- **Why embargo**: 7-day gap between train and test prevents autocorrelation leakage.
  Crypto is 24/7, so the gap must be in calendar days, not business days.
- **Why next-day execution**: Features at close T → execute at open T+1. This is
  the honest way to measure returns. Without this, you're overstating performance
  by the overnight gap (small per trade, compounds over years).

### Leakage Detector: Mandatory Before Logging
- **Why shuffled-label**: If a model predicts shuffled noise above 55% accuracy,
  something in the features contains target information. This catches look-ahead
  bugs that are invisible in metrics.
- **Why mandatory**: The leakage detector runs BEFORE MLflow logging. If it fails,
  the result is NOT logged. This prevents polluting the experiment tracker with
  fictitious results.

## 2. Common Pitfalls

### Target Variable Timing (CRITICAL)
```
Day T close    → Features computed (uses data up to T close)
Day T+1 open   → Trade EXECUTES
Day T+1+N close → Target measured: close_{T+1+N} > open_{T+1}
```
Do NOT use `close_{T+N} > close_T` — that ignores the overnight gap.
The backtester handles this, but you must set up targets correctly.

### Leakage Risks
1. **Rolling features without min_periods**: `df.rolling(30).mean()` produces
   values from row 1. Always use `min_periods=30` or features will have
   look-ahead bias for the first 29 rows.
2. **Feature selection on full dataset**: Feature selection must happen INSIDE
   each walk-forward fold, not before. The current setup does this correctly.
3. **pct_change default fill**: Use `fill_method=None` in pandas >= 2.2 to
   avoid FutureWarning and forward-fill leakage.

### Timezone Alignment
All timestamps are UTC. Parquet preserves timezone info. If you see
timezone-naive timestamps, localize to UTC before any join/merge.

## 3. File Map

```
src/sparky/
├── data/                          # Data ingestion (Phase 1)
│   ├── price.py                   # CCXTPriceFetcher (Binance + failover)
│   ├── onchain_bgeometrics.py     # BGeometricsFetcher (9 BTC computed indicators)
│   ├── onchain_coinmetrics.py     # CoinMetricsFetcher (BTC+ETH raw on-chain)
│   ├── onchain_blockchain_com.py  # BlockchainComFetcher (validation reference)
│   ├── market_context.py          # CoinGeckoFetcher (market snapshot)
│   ├── source_selector.py         # Cross-validates, picks best source/metric
│   ├── quality.py                 # DataQualityChecker (completeness, range, staleness)
│   └── storage.py                 # DataStore (Parquet + metadata + manifest)
├── features/                      # Feature engineering (Phase 2)
│   ├── returns.py                 # simple_returns, log_returns, sharpe, drawdown, vol
│   ├── technical.py               # rsi, ema, macd, momentum
│   ├── onchain.py                 # hash_ribbon, nvt_zscore, mvrv_signal, sopr_signal, etc.
│   ├── registry.py                # FeatureRegistry + FeatureDefinition + matrix builder
│   └── selection.py               # FeatureSelector (correlation, importance, stability)
├── backtest/                      # Backtesting (Phase 2)
│   ├── engine.py                  # WalkForwardBacktester + BacktestResult
│   ├── costs.py                   # TransactionCostModel (fee + slippage + spread)
│   ├── statistics.py              # Bootstrap CI, Sharpe significance, strategy comparison
│   └── leakage_detector.py        # Shuffled-label, temporal boundary, timestamp audit
├── models/                        # Models (Phase 2 baselines, Phase 3 ML)
│   └── baselines.py               # BuyAndHold, SimpleMomentum, EqualWeight
├── tracking/                      # Experiment tracking (Phase 2)
│   └── experiment.py              # ExperimentTracker (wraps MLflow)
├── oversight/                     # Agent monitoring (Phase 0)
│   └── activity_logger.py         # AgentActivityLogger (JSONL)
└── types/                         # Pydantic models (Phase 0)
    ├── market_types.py            # OHLCVCandle, OnChainMetric
    ├── signal_types.py            # Prediction, Signal
    ├── portfolio_types.py         # Position, PortfolioState, TradeOrder
    └── config_types.py            # SystemConfig, TradingRulesConfig
```

## 4. Decision Trees

### If model Sharpe < baseline
1. Check leakage detector output — any failed checks?
2. Check feature importance — are features contributing meaningfully?
3. Check target variable timing — using T+1 open execution?
4. Check transaction costs — round trip eating all alpha?
5. Check data quality report — any gaps or anomalies?

### If model Sharpe >> baseline (suspiciously good)
1. Run shuffled-label test — if it passes with >55% accuracy, leakage!
2. Check if features contain lagged target (look-ahead)
3. Check if train/test periods overlap
4. Check multi-seed stability — if Sharpe varies wildly across seeds, it's noise

### If model accuracy ~50% but positive Sharpe
This is NORMAL for daily crypto. 52-55% accuracy is meaningful alpha.
The Sharpe comes from correct sizing on high-conviction signals, not accuracy.

## 5. Code Templates

### Add a New Feature
```python
from sparky.features.registry import FeatureRegistry, FeatureDefinition

registry = FeatureRegistry()
registry.register(FeatureDefinition(
    name="my_feature",
    category="onchain_btc",
    compute_fn=lambda data: my_computation(data["onchain"]["some_metric"]),
    input_columns=["some_metric"],
    lookback=30,
    data_source="bgeometrics",
    valid_from="2017-01-01",  # or None if always valid
    asset="btc",
))
```

### Run a New Experiment
```python
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.tracking.experiment import ExperimentTracker

backtester = WalkForwardBacktester(train_min_length=252, embargo_days=7)
cost_model = TransactionCostModel.for_btc()
detector = LeakageDetector()
tracker = ExperimentTracker()

# Run backtest (returns is the 4th positional arg)
result = backtester.run(model, X, y, returns, cost_model=cost_model)

# Check leakage BEFORE logging
# NOTE: run_all_checks() does NOT corrupt the model — shuffle trials use a deep copy.
report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)
if not report.passed:
    # DO NOT LOG — write error to RESEARCH_LOG.md
    raise ValueError(f"Leakage detected: {report.failed_checks}")

# Log to MLflow
run_id = tracker.log_experiment(
    name="my_experiment",
    config={"model": "xgboost", "features": features_used},
    metrics={"sharpe": result.per_fold_metrics[0]["sharpe"]},
)
```

## 6. Phase 3 Task Checklist

For each experiment/model training:
- [ ] Features computed with correct lookback and min_periods
- [ ] Target uses T+1 open execution (not same-day close)
- [ ] Walk-forward with >= 5 folds and 7-day embargo
- [ ] Transaction costs applied (0.26% BTC, 0.28% ETH round trip)
- [ ] Leakage detector passes ALL checks before logging
- [ ] Results logged to MLflow with data_manifest hash
- [ ] Multi-seed stability: run with 3+ random seeds
- [ ] Compare vs baseline Sharpe with bootstrap p-value

## 7. Known Quirks

- **BGeometrics**: Free tier 8 req/hour, 15/day. API at bitcoin-data.com (not bgeometrics.com).
  Token via `?token=` URL param. Response uses `"d"` for date, values as strings.
  Derivatives endpoints (funding, OI, basis) are Advanced-only — skip entirely.
- **CoinMetrics**: Community tier, no auth. Returns `"time"` column (not `"date"`).
  ETH `AdrBalCnt` may not exist at Community tier — handle gracefully.
- **pandas pct_change**: Use `fill_method=None` to avoid FutureWarning in pandas >= 2.2.
- **Parquet timezone**: DatetimeIndex frequency metadata is NOT preserved through Parquet.
  Use `check_freq=False` in assert_frame_equal.
- **numpy copy-on-write**: When extracting `.values` from a pandas operation chain,
  the array may be read-only. Always `.copy()` before in-place mutation.
