# Data Expansion Plan — 10,000+ Observations

**Date**: 2026-02-15
**Priority**: CRITICAL (Unblock deep learning)
**Status**: Ready to execute

---

## 🎯 CRITICAL MISSION

**Expand training data to 10,000+ observations to enable advanced ML models.**

**PROBLEM IDENTIFIED**: Data starvation limiting model capacity
- **Current**: 2,178 daily samples (2019-2025) — insufficient for deep learning
- **Target**: 10,000+ samples — enables LSTM, Transformers, deep architectures
- **Root cause of Phase 2-3 failure**: Models with high capacity need more data to generalize

---

## Three-Pronged Approach (DO ALL THREE)

### APPROACH 1: Higher Frequency Data (HIGHEST PRIORITY)

**Switch from daily → hourly candles**

**Math**:
- Current: 2,178 daily candles (2019-2025, 6 years)
- Hourly: 2,178 days × 24 hours/day = **52,272 hourly candles**
- **24x multiplier** — massive sample increase

**Strategy**:
- **Train on hourly features** (RSI-14 hourly, Momentum-30 hours, EMA-ratio-20 hours)
- **Predict daily direction** (hourly features at day T close → daily target T to T+1)
- **Why this works**: Hourly microstructure captures intraday patterns that aggregate to daily moves

**Expected Output**:
- Feature matrix: 52,272 rows × 7-50 features (hourly)
- Target labels: 2,178 rows (daily direction, aligned with feature index)
- Training samples: 52K hourly observations

**Implementation**:
1. Fetch hourly OHLCV 2017-2025 via CCXT (Binance, Bybit failover)
2. Compute features on hourly data (RSI-14h, Momentum-30h, EMA-ratio-20h)
3. Resample features to daily close (last hourly value of each day)
4. Generate daily targets (close T+1 > close T)
5. Train XGBoost/LSTM on 52K hourly samples, predict daily

**Timeline**: 4-6 hours (fetch + feature engineering + validation)

---

### APPROACH 2: Cross-Asset Training (HIGH PRIORITY)

**Train on multiple assets, test on BTC only**

**Assets** (7 major cryptos):
1. BTC (Bitcoin)
2. ETH (Ethereum)
3. SOL (Solana)
4. ADA (Cardano)
5. DOT (Polkadot)
6. MATIC (Polygon)
7. AVAX (Avalanche)

**Math**:
- 7 assets × 70,000 hourly candles each = **490,000 total samples**
- Pooled training set: 490K observations
- Test set: BTC 2024-2025 holdout only

**Strategy**:
- **Add `asset_id` as categorical feature** (one-hot encoded or label encoded)
- **Train generic crypto momentum predictor** on pooled data
- **Test ONLY on BTC holdout** (never touch BTC 2024-2025 during training)
- **Why this works**: All cryptos share similar technical dynamics (momentum, mean-reversion), cross-asset training learns universal patterns

**Expected Output**:
- Training: 490K hourly samples across 7 assets (2017-2023)
- Validation: Multi-asset 2024 (for hyperparameter tuning)
- Holdout: BTC 2025 only (final test)

**Implementation**:
1. Fetch hourly OHLCV for 7 assets (2017-2025)
2. Compute identical features for all assets
3. Add `asset_id` column (categorical: 0-6)
4. Pool all assets into single training DataFrame
5. Train on pooled data, test on BTC holdout

**Timeline**: 6-8 hours (multi-asset fetch + feature alignment + training)

---

### APPROACH 3: Extended History (MEDIUM PRIORITY)

**Extend data back to 2017 or 2015**

**Math**:
- **2017-2025** (8 years): 2,920 daily candles (+730 days) or **70,080 hourly** (+17,520)
- **2015-2025** (10 years): 3,650 daily candles (+1,460 days) or **87,600 hourly** (+35,040)

**Strategy**:
- Capture 2017-2018 bull/crash cycle for regime diversity
- More market cycles = better generalization
- Combine with APPROACH 1 (hourly) for 70K-87K hourly samples

**Expected Output**:
- BTC hourly: 70,080 hourly candles (2017-2025)
- BTC hourly: 87,600 hourly candles (2015-2025) if earlier data available

**Implementation**:
1. Check CCXT availability: Binance launched 2017 (can get 2017-2025)
2. For 2015-2016: Use CoinGecko or alternative source (daily only, then interpolate)
3. Fetch hourly where available, daily where not
4. Combine with existing 2019-2025 data

**Timeline**: 3-4 hours (historical fetch + data alignment)

---

## Recommended Execution (DO ALL THREE)

### Phase 1: Hourly BTC Data (HIGHEST PRIORITY - START HERE)

**Objective**: 52K hourly samples, predict daily

**Steps**:
1. Create `scripts/fetch_hourly_btc.py`
   - Fetch BTC/USDT hourly OHLCV 2017-2025 via CCXT
   - Save to `data/raw/btc/ohlcv_hourly.parquet`
   - Expected: ~70,000 hourly candles

2. Create `scripts/prepare_hourly_features.py`
   - Load hourly OHLCV
   - Compute hourly features (RSI-14h, Momentum-30h, EMA-ratio-20h)
   - Resample to daily close (last hourly value per day)
   - Generate daily targets (close T+1 > close T)
   - Save to `data/processed/feature_matrix_btc_hourly.parquet`

3. Create `scripts/train_on_hourly.py`
   - Load hourly feature matrix
   - Train XGBoost on hourly samples
   - Predict daily direction
   - Validate on 2024-2025 holdout (daily)

**Success Criteria**:
- ✓ 52K+ hourly samples for BTC
- ✓ Features aligned to daily targets
- ✓ Holdout Sharpe >= 0.5 (positive alpha)

**Timeline**: 4-6 hours

---

### Phase 2: Cross-Asset Pooling (HIGH PRIORITY)

**Objective**: 490K pooled samples, test on BTC

**Steps**:
1. Create `scripts/fetch_cross_asset_hourly.py`
   - Fetch 7 assets hourly OHLCV 2017-2025
   - Save to `data/raw/{asset}/ohlcv_hourly.parquet`

2. Create `scripts/prepare_cross_asset_features.py`
   - Load all 7 assets
   - Compute identical features for each
   - Add `asset_id` column (categorical)
   - Pool into single DataFrame
   - Save to `data/processed/feature_matrix_cross_asset_hourly.parquet`

3. Create `scripts/train_cross_asset.py`
   - Load pooled feature matrix (490K samples)
   - Train XGBoost on all assets (2017-2023)
   - Test ONLY on BTC 2024-2025 holdout
   - Compare to BTC-only model

**Success Criteria**:
- ✓ 490K+ pooled hourly samples
- ✓ Model generalizes to BTC holdout
- ✓ Cross-asset Sharpe >= BTC-only Sharpe

**Timeline**: 6-8 hours

---

### Phase 3: Extended History (MEDIUM PRIORITY)

**Objective**: 70K+ hourly samples back to 2017

**Steps**:
1. Modify `scripts/fetch_hourly_btc.py`
   - Change start_date from 2019 to 2017
   - Fetch 2017-2025 hourly OHLCV
   - Expected: ~70,000 hourly candles

2. Re-run feature preparation with extended data
3. Compare 2017-2025 model to 2019-2025 baseline

**Success Criteria**:
- ✓ 70K+ hourly samples (2017-2025)
- ✓ Captures 2017-2018 bull/crash cycle
- ✓ Model generalizes to 2024-2025 holdout

**Timeline**: 2-3 hours (incremental to Phase 1)

---

## Expected Impact

### Current Setup (FAILED)
- **Data**: 2,178 daily samples (2019-2025)
- **Features**: 7 (technical + on-chain)
- **Model**: XGBoost, 30d horizon
- **Result**: Sharpe 0.999 train → **-0.295 holdout** (catastrophic overfitting)

### NEW Setup (Target)
- **Data**: 52K-490K hourly samples (2017-2025, multi-asset)
- **Features**: 30-50 (technical + macro + cross-asset + on-chain + derivatives)
- **Model**: XGBoost/LSTM on hourly, predict daily
- **Expected**: Holdout Sharpe **>= 0.7** (real alpha, no overfitting)

**Why This Should Work**:
1. **More data** (52K vs 2K) → models can learn complex patterns without overfitting
2. **Hourly frequency** → captures intraday microstructure (volatility clustering, momentum persistence)
3. **Cross-asset training** → learns universal crypto dynamics (not BTC-specific noise)
4. **Extended history** → more market regimes (bull, bear, sideways)
5. **Better features** → macro + derivatives + cross-asset (30-50 vs 7)

---

## Validation Methodology (CRITICAL - NO DATA SNOOPING)

### New Split Strategy

```
2017-2020: Train (3 years, ~26K hourly samples)
2021-2022: Validation (2 years, ~17K hourly samples) — use for model selection
2023:      Test (1 year, ~8.7K hourly samples) — use for hyperparameter tuning
2024-2025: Holdout (2 years, ~17K hourly samples) — FINAL test, ONE run only
```

**RULES** (Non-Negotiable):
1. **NEVER touch 2024-2025 holdout until final validation**
2. Use 2021-2022 validation set for ALL experimentation (feature selection, model selection)
3. Use 2023 test set for hyperparameter tuning ONLY
4. Test ONE final model on 2024-2025
5. If that fails → back to drawing board, NO peeking

**Leakage Prevention**:
- Feature selection INSIDE each walk-forward fold (not on full dataset)
- Leakage detector MUST pass before logging to MLflow
- Holdout performance reported ONCE (no iterative testing)

---

## Integration with Alternative Data Plan

**Combine data expansion with feature expansion:**

1. **Fetch hourly data** (APPROACH 1-3 above) → 52K-490K samples
2. **Integrate macro data** (VIX, DXY, SPY, Gold via FRED) → +10 features
3. **Integrate cross-asset signals** (ETH/BTC ratio, dominance) → +10 features
4. **Integrate derivatives** (funding rates via CoinGlass) → +5-10 features
5. **Enhanced on-chain** (exchange flows, whale activity) → +10 features

**Total**:
- **Data**: 52K-490K hourly samples (vs 2K daily)
- **Features**: 30-50 features (vs 7)
- **Model**: Deep learning viable (LSTM, Transformers)

---

## Success Metrics

### Minimum Viable:
- **Holdout Sharpe >= 0.5** (positive alpha, no overfitting)
- **Train-holdout gap < 0.3** (good generalization)
- **Leakage detector passes** (all checks)

### Stretch Goal:
- **Holdout Sharpe >= 1.0** (strong alpha)
- **Cross-asset generalization** (works on ETH, SOL, not just BTC)
- **Drawdown < 50%** (risk-managed)

---

## Timeline Estimate

**Phase 1 (Hourly BTC)**: 4-6 hours
**Phase 2 (Cross-asset)**: 6-8 hours
**Phase 3 (Extended history)**: 2-3 hours (incremental)
**Alternative data integration**: 8-12 hours (from ALTERNATIVE_DATA_PLAN.md)

**Total**: 20-29 hours to full implementation
**Expected completion**: 2-3 days of focused work

---

## Next Actions (Execute in Order)

### Step 1: Hourly BTC Data (START HERE)
1. ✅ Create `scripts/fetch_hourly_btc.py`
2. ⏸️ Run: `python scripts/fetch_hourly_btc.py`
3. ⏸️ Verify: `data/raw/btc/ohlcv_hourly.parquet` has 70K+ rows
4. ⏸️ Create `scripts/prepare_hourly_features.py`
5. ⏸️ Run: `python scripts/prepare_hourly_features.py`
6. ⏸️ Verify: `data/processed/feature_matrix_btc_hourly.parquet` has 52K+ rows
7. ⏸️ Commit: `data: fetch BTC hourly OHLCV 2017-2025 (70K samples)`

### Step 2: Train on Hourly Data
1. ⏸️ Create `scripts/train_on_hourly.py`
2. ⏸️ Run: `python scripts/train_on_hourly.py`
3. ⏸️ Check: Holdout Sharpe >= 0.5?
4. ⏸️ If YES → Proceed to Phase 2 (cross-asset)
5. ⏸️ If NO → Debug (feature engineering, model complexity)

### Step 3: Cross-Asset Pooling
1. ⏸️ Create `scripts/fetch_cross_asset_hourly.py`
2. ⏸️ Run: Fetch 7 assets (BTC, ETH, SOL, ADA, DOT, MATIC, AVAX)
3. ⏸️ Create `scripts/prepare_cross_asset_features.py`
4. ⏸️ Create `scripts/train_cross_asset.py`
5. ⏸️ Check: Cross-asset Sharpe >= BTC-only?

### Step 4: Alternative Data Integration
1. ⏸️ Execute Phase 1-4 from ALTERNATIVE_DATA_PLAN.md
2. ⏸️ Integrate macro, cross-asset, derivatives, on-chain features
3. ⏸️ Retrain on 52K-490K samples with 30-50 features
4. ⏸️ Final validation on 2024-2025 holdout (ONE test)

---

## STOP ALL SIMPLE RULE TESTING

**FOCUS**:
- ✅ More data (hourly, cross-asset, extended history)
- ✅ Better features (macro, derivatives, on-chain)
- ✅ Proper validation (no data snooping, ONE holdout test)

**DO NOT**:
- ❌ Test momentum thresholds
- ❌ Test simple moving average crossovers
- ❌ Optimize on holdout set
- ❌ Iterate on 2024-2025 holdout

---

**Status**: Ready to execute Step 1 (Hourly BTC Data)

**Awaiting**: Approval to proceed with `scripts/fetch_hourly_btc.py` implementation
