# Alternative Data Integration Plan

**Date**: 2026-02-15
**Priority**: HIGH (Better features for ML models)
**Status**: Planning phase

---

## Objective

Integrate alternative data sources to improve ML model predictive power:
1. **Sentiment data** (social media, news)
2. **Macro indicators** (VIX, DXY, rates, equities)
3. **Derivatives** (funding rates, options flow, open interest)
4. **Cross-asset signals** (BTC-ETH correlation, alt-season indicators)

---

## Data Source Priorities

### TIER 1: Free & Immediately Available

#### 1. Macro Economic Indicators
**Source**: Federal Reserve Economic Data (FRED API - free)
- **VIX** (volatility index) — risk-off sentiment
- **DXY** (US Dollar Index) — BTC inverse correlation
- **10Y Treasury Yield** — risk-free rate
- **Gold (GLD)** — alternative safe haven
- **S&P 500** — risk-on sentiment
- **M2 Money Supply** — liquidity indicator

**Why**: BTC correlates with macro risk environment
**API**: `fredapi` Python library (free)
**Availability**: Daily data, goes back decades

---

#### 2. Cross-Asset Crypto Signals
**Source**: Our existing CCXT fetcher
- **ETH/BTC ratio** — alt-season indicator
- **BTC dominance** — market structure
- **ETH, SOL, BNB prices** — cross-correlation
- **Stablecoin supply** (USDT, USDC) — market liquidity

**Why**: BTC doesn't trade in isolation
**API**: Already have CCXT
**Availability**: Daily OHLCV for multiple assets

---

#### 3. Basic On-Chain Volume Metrics
**Source**: CoinMetrics Community (free)
- **Exchange inflows/outflows** — selling pressure
- **Large transaction count** — whale activity
- **Active addresses growth** — network adoption
- **Fee revenue** — network usage

**Why**: Better on-chain coverage than current
**API**: Already integrated
**Availability**: Daily, back to 2017

---

### TIER 2: Paid but High-Value

#### 4. Derivatives Data
**Source**: CoinGlass API (has free tier)
- **Funding rates** (BTC perp futures) — long/short bias
- **Open interest** — leverage in system
- **Liquidation data** — market stress
- **Long/short ratio** — sentiment

**Why**: Derivatives lead spot (informed traders)
**API**: CoinGlass free tier: 100 calls/day
**Availability**: Real-time + historical

---

#### 5. Social Sentiment (Basic)
**Source**: LunarCrush API (free tier exists)
- **Social volume** (Twitter/Reddit mentions)
- **Social sentiment** (positive/negative)
- **Influencer activity** — narrative shifts
- **Galaxy Score** (composite metric)

**Why**: Retail sentiment drives volatility
**API**: LunarCrush free tier: 100 points/day
**Availability**: Daily aggregates

---

### TIER 3: Advanced (Deprioritize for Now)

#### 6. Options Flow
**Source**: Deribit (requires subscription)
- Put/call ratio
- Implied volatility surface
- Max pain levels

**Why**: Advanced, may not add much signal
**Cost**: $$$ (skip for now)

---

#### 7. News Sentiment
**Source**: CryptoPanic API or NewsAPI
- Crypto news aggregation
- Sentiment scoring
- Event detection

**Why**: News drives short-term moves
**Cost**: Free tier limited
**Priority**: Lower (noisy signal)

---

## Implementation Priority

### Phase 1: Free Macro Data (2-3 hours)

**Immediate value, zero cost:**

1. **Fetch FRED data** (VIX, DXY, SPY, Gold, 10Y yield)
2. **Compute features**:
   - VIX 14-day change (fear spikes)
   - DXY 30-day momentum (dollar strength)
   - SPY/BTC correlation (risk-on/off regime)
   - Gold/BTC ratio (safe haven rotation)

3. **Integrate into feature matrix**:
   - Add 5-10 macro features
   - Test correlation with BTC returns
   - Validate feature importance in XGBoost

---

### Phase 2: Cross-Asset Crypto (1-2 hours)

**Already have the data fetcher:**

1. **Fetch ETH, BNB, SOL OHLCV**
2. **Compute features**:
   - ETH/BTC ratio momentum (alt-season)
   - BTC dominance (from market caps)
   - Correlation matrix (BTC-ETH-alts)
   - Relative strength (BTC vs alts)

3. **Add to feature matrix**:
   - 5-10 cross-asset features
   - Test predictive power

---

### Phase 3: Enhanced On-Chain (2-3 hours)

**Use CoinMetrics Community more deeply:**

1. **Fetch additional metrics**:
   - Exchange flows (net inflow/outflow)
   - Large transactions (>$1M, >$10M)
   - Entity-adjusted metrics
   - Realized cap, MVRV improvements

2. **Compute derivatives**:
   - Exchange flow momentum
   - Whale accumulation score
   - Network growth velocity

---

### Phase 4: Derivatives Data (3-4 hours)

**Requires CoinGlass API integration:**

1. **Set up CoinGlass fetcher**
2. **Fetch funding rates** (BTC perpetual swaps)
3. **Compute features**:
   - Funding rate divergence
   - OI change vs price
   - Liquidation cascade risk
   - Long/short ratio extremes

---

### Phase 5: Sentiment (Optional, 2-3 hours)

**LunarCrush or similar:**

1. **Set up LunarCrush API**
2. **Fetch social metrics**
3. **Test signal quality** (may be noisy)

---

## Expected Impact

### Current Feature Set (7 features):
- RSI, Momentum, EMA ratio (technical)
- Returns (momentum)
- Hash ribbon, address momentum, volume momentum (on-chain)

**Result**: Sharpe 0.999 on train/test, FAILED holdout (-0.39)

### Expanded Feature Set (30-50 features):
- Technical (7)
- Macro (10): VIX, DXY, SPY, Gold, yields, correlations
- Cross-asset (10): ETH/BTC, dominance, alt correlations
- Enhanced on-chain (10): Exchange flows, whales, network growth
- Derivatives (5-10): Funding, OI, liquidations

**Expected**: Better generalization through richer signal

---

## Feature Engineering Strategy

### 1. Regime Detection
- **Bull/bear regime** (based on 200-day MA)
- **Risk-on/off regime** (VIX, SPY correlation)
- **Liquidity regime** (M2 growth, stablecoin supply)

Features may work differently in different regimes.

### 2. Interaction Features
- **BTC-macro interactions**: BTC momentum × VIX spike
- **Cross-asset divergences**: BTC up, ETH/BTC down = warning
- **On-chain + price**: Price up, exchange inflows up = distribution

### 3. Lookback Windows
Test multiple timeframes:
- Short (7d, 14d): Tactical signals
- Medium (30d, 60d): Trend signals
- Long (90d, 180d): Regime signals

---

## Data Quality Validation

For each new data source:

1. **Completeness check**: No gaps, sufficient history
2. **Alignment check**: Matches BTC timestamp index
3. **Correlation analysis**: Check for leakage (shouldn't perfectly correlate with target)
4. **Stationarity test**: Check if series is stationary or needs differencing
5. **Feature importance**: XGBoost reports top features

---

## Validation Methodology (CRITICAL)

**NEVER repeat the data snooping mistake.**

### New Split Strategy:

```
2013-2018: Train (5 years) - 1,826 days
2019-2020: Validation (model selection) - 730 days
2021-2023: Test (hyperparameter tuning) - 1,095 days
2024-2025: Holdout (FINAL validation, ONE test only) - 730 days
```

**Rules**:
1. NEVER look at 2024-2025 holdout until final validation
2. Use 2021-2023 test set for all experimentation
3. Only test ONE final model on 2024-2025
4. If that fails, go back to drawing board (no peeking)

---

## Success Metrics

### Minimum Viable Improvement:
- **Holdout Sharpe >= 0.5** (positive, beats baseline 0.79 on risk-adjusted basis)
- **Train-test gap < 0.3** (indicates good generalization)
- **Leakage detector passes** (all checks)

### Stretch Goal:
- **Holdout Sharpe >= 1.0** (strong signal)
- **Stable across multiple assets** (BTC and ETH both work)
- **Drawdown < 50%** (risk-managed)

---

## Timeline

**Phase 1 (Macro)**: 2-3 hours
**Phase 2 (Cross-asset)**: 1-2 hours
**Phase 3 (Enhanced on-chain)**: 2-3 hours
**Phase 4 (Derivatives)**: 3-4 hours (if needed)

**Total**: 8-12 hours to integrate Tier 1 data sources

**Then**: Retrain models on expanded data (2013-2025) with expanded features

---

## Next Actions

1. ✅ Fetch 2013-2018 BTC historical data
2. ✅ Integrate FRED macro data (VIX, DXY, SPY, Gold)
3. ✅ Add cross-asset features (ETH, alts)
4. ✅ Enhanced on-chain from CoinMetrics
5. ⏸️ Retrain XGBoost with 30-50 features on 2013-2020 train set
6. ⏸️ Test on 2021-2023 test set
7. ⏸️ Final validation on 2024-2025 holdout (ONE test only)

---

**Status**: Ready to execute Phase 1 (Macro data integration)
