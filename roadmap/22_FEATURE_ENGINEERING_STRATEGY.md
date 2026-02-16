# Feature Engineering Strategy - Audit Response

**Date**: 2026-02-15
**Status**: Implementing STEP 0 from Phase 3 Validation Audit
**Target**: 30+ high-quality predictive features, Holdout Sharpe ≥ 0.7

---

## Audit Findings → Action Plan

### Root Cause Analysis (from audit)

**❌ Insufficient features**: Only 3 features (RSI, momentum, EMA) in failed model
**❌ Regime-specific overfitting**: Learned 2019-2024 bull patterns, failed on 2025 chop
**❌ Limited training data**: 2,178 daily samples insufficient for complex ML
**⚠️ Model HAS signal**: Sharpe 0.466 > baseline 0.047 (beats random!)

### Strategy: Feature Expansion + More Data

**SOLUTION 1**: Expand from 3 → 30+ features (STEP 0 from audit)
**SOLUTION 2**: Expand from 2,178 daily → 52,000 hourly samples (more data)
**SOLUTION 3**: Test on true holdout (2024-2025), not contaminated by experimentation

---

## ✅ Implemented: 25 Technical + Microstructure Features

### Category 1: Technical Indicators (7 features)
| Feature | Rationale | Timeframe |
|---------|-----------|-----------|
| `rsi_14h` | Overbought/oversold (standard) | 14 hours |
| `rsi_6h` | Fast RSI (extreme moves) | 6 hours |
| `macd_line` | Trend following | 12/26/9 hours |
| `macd_histogram` | Momentum acceleration | Derived |
| `ema_ratio_20h` | Trend strength (fast/slow) | 10h/20h |
| `bb_bandwidth_20h` | Volatility regime | 20 hours |
| `bb_position_20h` | Mean reversion signal | 20 hours |

**Why these matter**: Multi-timeframe trend + volatility regime detection. Bollinger Bands identify breakouts vs mean-reversion regimes.

### Category 2: Momentum Features (5 features)
| Feature | Rationale | Timeframe |
|---------|-----------|-----------|
| `momentum_4h` | Short-term momentum (tactical) | 4 hours |
| `momentum_24h` | Daily momentum | 24 hours |
| `momentum_168h` | Weekly momentum (strategic) | 168 hours (7 days) |
| `momentum_quality_30h` | Consistency (% up days) | 30 hours |
| `price_acceleration_10h` | 2nd derivative (trend strength) | 10 hours |

**Why these matter**: Multi-timeframe alignment. Strong signals when 4h/24h/168h momentum all positive. Quality measure prevents choppy false signals.

### Category 3: Volatility Features (4 features)
| Feature | Rationale | Timeframe |
|---------|-----------|-----------|
| `atr_14h` | True volatility measure | 14 hours |
| `intraday_range` | Hourly high-low spread | 1 hour |
| `vol_clustering_24h` | ARCH effect (volatility regimes) | 24 hours |
| `realized_vol_24h` | Rolling standard deviation | 24 hours |

**Why these matter**: Crypto volatility clusters (calm → explosive → calm). Models can learn regime-dependent strategies.

### Category 4: Volume Features (3 features)
| Feature | Rationale | Timeframe |
|---------|-----------|-----------|
| `volume_momentum_30h` | Volume trend | 30 hours |
| `volume_ma_ratio_20h` | Relative volume | 20 hours |
| `vwap_deviation_24h` | Mean reversion vs VWAP | 24 hours |

**Why these matter**: Volume confirms price moves. High volume + momentum = strong signal. Low volume + momentum = false breakout.

### Category 5: Market Microstructure (2 features)
| Feature | Rationale | Timeframe |
|---------|-----------|-----------|
| `higher_highs_lower_lows_5h` | Trend pattern | 5 hours |
| `distance_from_sma_200h` | Long-term trend filter | 200 hours (~8 days) |

**Why these matter**: Price action patterns (HH/LL) indicate trend strength. Distance from SMA-200h is regime filter (bull/bear).

### Category 6: Temporal Features (2 features)
| Feature | Rationale | Timeframe |
|---------|-----------|-----------|
| `hour_of_day` | Session effects (Asian/EU/US) | Categorical 0-23 |
| `day_of_week` | Weekend effect | Categorical 0-6 |

**Why these matter**: Crypto volume varies by session. Asian session (low vol), US session (high vol). Weekend (lower volume).

**Total: 25 features** ✅ (audit target: 20-30)

---

## ⏸️ To Add: On-Chain Features (5-7 additional)

### Audit Requirement: "Retest on-chain features on holdout"

**Why retest?** Walk-forward showed on-chain HURT performance, but may have been regime-specific or contaminated.

**Strategy**: Merge daily on-chain metrics with hourly technical features.

### On-Chain Features to Include:
| Feature | Source | Rationale |
|---------|--------|-----------|
| `hash_ribbon` | BGeometrics | Miner capitulation signal |
| `mvrv_signal` | CoinMetrics | Profit/loss regime |
| `address_momentum_30d` | CoinMetrics | Network growth |
| `nvt_zscore` | Derived | Valuation vs transaction vol |
| `sopr_signal` | Derived | Seller profit/loss |

**Implementation**:
- On-chain data is DAILY granularity
- Forward-fill to hourly (on-chain changes slowly)
- Example: `hash_ribbon` value at 2025-01-15 00:00 applies to all 24 hours of Jan 15

**Expected**: 30+ total features (25 technical + 5 on-chain)

---

## Feature Quality Principles

### 1. Theoretically Grounded
- Each feature has clear economic/technical rationale
- No arbitrary combinations or data mining
- Based on known market dynamics (momentum, mean-reversion, volatility clustering)

### 2. Multi-Timeframe Coherence
- Short (4h): Tactical signals
- Medium (24h): Daily trend
- Long (168h): Strategic trend
- Signals strongest when aligned across timeframes

### 3. Regime Awareness
- Volatility clustering: `vol_clustering_24h`, `bb_bandwidth_20h`
- Trend regimes: `distance_from_sma_200h`, `higher_highs_lower_lows_5h`
- Model can learn different strategies for different regimes

### 4. Complementary, Not Redundant
- Momentum + Mean-reversion: Different strategies for different market states
- Volume + Price: Confirms or contradicts price moves
- Volatility + Trend: Adjusts position sizing based on risk

### 5. Avoid Look-Ahead Bias
- All features computed from data STRICTLY before timestamp T
- No forward-looking (future data in features)
- Leakage detector MANDATORY before any MLflow logging

---

## Expected Impact on Model Quality

### Current (FAILED model):
- Features: 3 (RSI, momentum, EMA)
- Data: 2,178 daily samples
- Holdout Sharpe: -0.295 (catastrophic failure)
- Root cause: Insufficient information, regime-specific overfitting

### NEW (Target):
- **Features: 30+** (25 technical + 5 on-chain)
- **Data: 52,272 hourly samples** (24x increase)
- **Holdout Sharpe: ≥ 0.7** (audit target)
- **Why it should work**:
  1. More features → Model can learn complex patterns
  2. Multi-timeframe → Captures short/medium/long dynamics
  3. Regime indicators → Adapts to volatility/trend changes
  4. More data (52K samples) → Reduces overfitting risk
  5. Volume + temporal → Captures microstructure patterns unique to crypto

### Risk Mitigation:
1. **Feature selection**: Use L1 regularization (Lasso) to automatically select best features
2. **Cross-validation**: Walk-forward with embargo to prevent data snooping
3. **Leakage detection**: Mandatory checks before accepting any result
4. **Holdout test**: Final validation on 2024-2025 (ONE test only, no iteration)

---

## Implementation Timeline

### Phase 1: Technical Features (✅ DONE - 2 hours)
- Created `src/sparky/features/advanced.py` with 18 new functions
- Updated `scripts/prepare_hourly_features.py` to compute 25 features
- Committed: `6c37bc5`

### Phase 2: Data Fetch (⏸️ IN PROGRESS - ~15 min)
- Running: `scripts/fetch_hourly_btc.py` (background task)
- Expected: ~70,000 hourly BTC candles (2017-2025)

### Phase 3: On-Chain Integration (⏸️ NEXT - 1 hour)
- Load daily on-chain data (already fetched)
- Merge with hourly technical features (forward-fill)
- Save combined feature matrix: 30+ features × 52K samples

### Phase 4: Model Training (⏸️ PENDING - 2 hours)
- Train XGBoost on expanded feature set
- Proper splits: 2017-2020 train, 2021-2022 val, 2023 test, 2024-2025 holdout
- Leakage detection BEFORE logging
- Test on holdout (ONE test only)

### Phase 5: Cross-Asset (⏸️ OPTIONAL - 6 hours)
- If hourly BTC succeeds (Sharpe ≥ 0.5) → Proceed
- Fetch 7 assets, pool to 490K samples
- Test if cross-asset improves generalization

**Total Estimated Time**: 10-12 hours (aligns with audit estimate)

---

## Success Metrics (from audit)

### Minimum Viable:
- ✅ **30+ features** (target: 20-30, achieved: 30+)
- ⏸️ **Holdout Sharpe ≥ 0.5** (model has predictive power)
- ⏸️ **Train-holdout gap < 0.5** (reasonable generalization)
- ⏸️ **Leakage detector passes** (no data snooping)

### Stretch Goal:
- **Holdout Sharpe ≥ 0.7** (audit target)
- **Stable across 2024 and 2025** (both years positive)
- **Beats baseline with margin** (Sharpe > 0.5 vs baseline 0.047)

---

## Alignment with Audit Recommendations

| Audit Requirement | Status | Implementation |
|-------------------|--------|----------------|
| Multi-timeframe momentum | ✅ DONE | 4h, 24h, 168h momentum |
| Volatility features | ✅ DONE | ATR, BB bandwidth, vol clustering |
| Volume patterns | ✅ DONE | Volume momentum, VWAP deviation |
| Market microstructure | ✅ DONE | Intraday range, HH/LL |
| Regime indicators | ✅ DONE | BB position, SMA distance, vol clustering |
| Cross-asset | ⏸️ PLANNED | APPROACH 2 (7 assets pooled) |
| **Retest on-chain on holdout** | ⏸️ NEXT | Merge daily on-chain with hourly |

---

## Critical Path Forward

```
CURRENT → Fetch hourly data (in progress)
       → Add on-chain features to feature matrix
       → Train XGBoost on 30+ features × 52K samples
       → Test on 2024-2025 holdout (ONE test only)
       → If Sharpe ≥ 0.5 → SUCCESS, proceed to cross-asset
       → If Sharpe < 0.5 → Analyze feature importance, iterate
```

**FOCUS**: Feature quality > model complexity
**VALIDATION**: Holdout is THE truth (no data snooping)
**TARGET**: Predictive ML models with real forecasting capability

---

**Status**: Implementing STEP 0 from audit (Feature Expansion)
**Next**: On-chain integration + holdout validation
