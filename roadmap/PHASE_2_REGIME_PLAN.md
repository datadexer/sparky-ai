# Phase 2A: Regime-Aware Trading (CRITICAL)

**Status**: PLANNED (execute after Phase 1 if AUC ≥ 0.57)
**Priority**: 🔴 CRITICAL - Research shows this is root cause of Sharpe 0.646 underperformance
**Effort**: 3-4 hours
**Expected Impact**: Sharpe 0.646 → 0.85-1.0

---

## Problem Statement

**Current Issue:**
- Model trained on 2017-2023 (mixed regimes: bear, bull, sideways)
- Applied with FIXED threshold (0.50) and FULL position sizing (100%)
- 2024-2025 holdout was strong bull market (+98% BTC return)
- Strategy underperformed Buy & Hold: Sharpe 0.646 vs 0.950 (-56% return)

**Root Cause (Research-Validated):**
- Bitcoin exhibits "distinct volatility regimes more persistent than S&P 500"
- Markets are non-stationary — 2024-2025 regime differs from training period
- Static models fail when regime shifts (IMCA paper: dynamic adaptation achieves Sharpe 0.829)
- Regime-switching models show "potential benefits for investment management" (2025)

---

## Solution: Regime-Aware Position Sizing & Thresholds

### Regime Classification

**Volatility Regimes** (based on realized 30-day annualized volatility):
- **LOW**: <30% annualized (calm, trending markets)
- **MEDIUM**: 30-60% annualized (normal crypto volatility)
- **HIGH**: >60% annualized (chaotic, crisis periods)

**Historical Examples:**
- LOW: Mid-2023 to mid-2024 (BTC consolidation around $40K)
- MEDIUM: Q1 2021, Q4 2024 (normal bull market)
- HIGH: Mid-2021 (China ban), Early 2022 (Terra collapse), Late 2024 (uncertain period)

### Regime-Aware Rules

| Regime | Position Size | Probability Threshold | Rationale |
|--------|--------------|----------------------|-----------|
| **HIGH** | 50% | 0.55 | Reduce exposure during chaos, require higher confidence |
| **MEDIUM** | 75% | 0.52 | Moderate caution in normal volatility |
| **LOW** | 100% | 0.50 | Full exposure in calm trending markets |

**Key Insight:** During HIGH volatility regimes, prediction accuracy degrades. Better to:
- Reduce position size (risk management)
- Require higher probability threshold (avoid false signals)
- Stay mostly flat during extreme chaos

---

## Implementation Plan

### Step 1: Compute Regime Indicators (1 hour)

Add to `src/sparky/features/regime_indicators.py`:

```python
def compute_volatility_regime(prices: pd.Series, window: int = 30 * 24) -> pd.Series:
    """
    Compute volatility regime classification.

    Args:
        prices: Hourly close prices
        window: Rolling window (default 30 days × 24 hours)

    Returns:
        Series with values: 'low', 'medium', 'high'
    """
    # Realized volatility (annualized)
    returns = prices.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(24 * 365)  # Annualize hourly vol

    # Classify regimes
    regime = pd.Series(index=prices.index, dtype=str)
    regime[vol < 0.30] = 'low'
    regime[(vol >= 0.30) & (vol < 0.60)] = 'medium'
    regime[vol >= 0.60] = 'high'

    return regime
```

**Additional Regime Indicators** (optional enhancements):
- Trend strength (ADX)
- Market structure (higher highs/lower lows count)
- VIX proxy (if options data available)

### Step 2: Regime-Aware Signal Generator (1.5 hours)

Modify `src/sparky/models/signal_aggregator.py`:

```python
class RegimeAwareAggregator:
    """Aggregates hourly predictions with regime-aware thresholds."""

    REGIME_RULES = {
        'high':   {'position_size': 0.50, 'threshold': 0.55},
        'medium': {'position_size': 0.75, 'threshold': 0.52},
        'low':    {'position_size': 1.00, 'threshold': 0.50},
    }

    def aggregate_to_daily(
        self,
        hourly_probs: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Aggregate hourly predictions to daily signals with regime awareness.

        Returns:
            DataFrame with columns: date, signal, position_size, regime
        """
        # Daily mean probability
        daily_prob = hourly_probs.resample('D').mean()

        # Daily regime (most common regime that day)
        daily_regime = regimes.resample('D').agg(lambda x: x.mode()[0])

        # Regime-aware signals
        signals = []
        for date in daily_prob.index:
            prob = daily_prob[date]
            regime = daily_regime[date]
            rules = self.REGIME_RULES[regime]

            signal = 1 if prob > rules['threshold'] else 0
            position_size = rules['position_size'] if signal == 1 else 0

            signals.append({
                'date': date,
                'probability': prob,
                'regime': regime,
                'signal': signal,
                'position_size': position_size
            })

        return pd.DataFrame(signals).set_index('date')
```

### Step 3: Backtest with Regime Overlay (1 hour)

Create `scripts/backtest_regime_aware.py`:

```python
# Load Phase 1 cross-asset model
model = load_model('results/cross_asset_pooled/best_model.cbm')

# Generate hourly predictions on 2024-2025 holdout
holdout_probs = model.predict_proba(X_holdout_btc)[:, 1]

# Compute volatility regime for holdout period
regimes = compute_volatility_regime(holdout_prices)

# Aggregate with regime awareness
aggregator = RegimeAwareAggregator()
daily_signals = aggregator.aggregate_to_daily(holdout_probs, regimes)

# Backtest strategy
returns = compute_strategy_returns(
    signals=daily_signals['signal'],
    position_sizes=daily_signals['position_size'],
    prices=holdout_prices
)

# Compare to baseline
baseline_returns = (holdout_prices.pct_change() * 1.0).fillna(0)  # 100% long

print(f"Regime-Aware Sharpe: {sharpe_ratio(returns):.3f}")
print(f"Buy & Hold Sharpe:   {sharpe_ratio(baseline_returns):.3f}")
print(f"\nRegime Distribution:")
print(daily_signals['regime'].value_counts())
```

### Step 4: Validation & Analysis (0.5 hours)

**Key Metrics:**
- Sharpe ratio by regime (does strategy beat Buy & Hold in each regime?)
- Win rate by regime (does accuracy degrade in HIGH regime as expected?)
- Drawdown by regime (does position sizing reduce losses in HIGH?)
- Trade frequency by regime (should be lower in HIGH)

**Success Criteria:**
- Overall Sharpe ≥ 0.95 (match or beat Buy & Hold 0.950)
- HIGH regime: Lower drawdown than full exposure
- LOW regime: Capture majority of upside

---

## Expected Results

### Hypothesis

**2024-2025 Holdout Analysis:**
- Q1 2024: MEDIUM regime → 75% position size, threshold 0.52
- Q2-Q3 2024: LOW regime → 100% position size, threshold 0.50
- Q4 2024: HIGH regime → 50% position size, threshold 0.55
- Q1 2025: MEDIUM regime → 75% position size, threshold 0.52

**Impact on Performance:**
1. **Reduce losses in HIGH volatility** (50% position size)
2. **Capture upside in LOW volatility** (100% position size, low threshold)
3. **Lower trading frequency** (higher thresholds reduce whipsaws)
4. **Improve risk-adjusted returns** (adaptive risk management)

### Target Metrics

| Metric | Current (Static) | Target (Regime-Aware) | Improvement |
|--------|-----------------|----------------------|-------------|
| Sharpe | 0.646 | 0.90-1.05 | +40-62% |
| Max Drawdown | 31.42% | <28% | -10% |
| Win Rate | 32.2% | 38-42% | +18-30% |
| Trades/Year | 312 | 220-260 | -30% |

**Rationale for Improvement:**
- HIGH regime periods (20-30% of time) currently destroy alpha
- Reducing exposure during chaos prevents catastrophic losses
- Full exposure during calm periods captures trend following upside
- Higher thresholds reduce false signals in choppy markets

---

## Research Support

**Key Papers:**
1. **IMCA (2025)**: "Dynamically recalibrating model weights in real-time" → Sharpe 0.829
2. **Regime-switching forecasting (2024)**: "Significantly improves cryptocurrency prediction accuracy"
3. **Volatility modeling (2025)**: Bitcoin has "distinct volatility regimes more persistent than S&P 500"
4. **Regime-switching RL (2025)**: "Potential benefits for investment management"

**Industry Validation:**
- Professional crypto traders use volatility filters
- Institutional portfolios reduce exposure during VIX spikes
- Trend-following CTAs use volatility scaling (Risk Parity approach)

---

## Risks & Mitigations

**Risk 1:** Regime classification lag (30-day window)
- **Mitigation**: Use exponentially-weighted volatility (recent hours weighted more)
- **Alternative**: Add trend indicators (ADX) to detect regime shifts faster

**Risk 2:** Over-fitting to 2024-2025 regime patterns
- **Mitigation**: Validate on full history (2017-2023 walk-forward + 2024-2025 holdout)
- **Guard**: Don't tune regime thresholds on holdout (use literature-based defaults)

**Risk 3:** Regime rules too conservative (miss upside)
- **Mitigation**: Start with aggressive rules (90%/100%/100%), reduce if needed
- **Test**: Compare regime-aware to regime-agnostic on same holdout

---

## Integration with Phase 1 & 2B

**Phase 1 (Cross-Asset):** Improves base model AUC 0.536 → 0.57-0.58
**Phase 2A (Regime-Aware):** Applies regime overlay to Phase 1 model
**Phase 2B (Volume Features):** Further improves AUC 0.57 → 0.59-0.60

**Combined Effect:**
- Better predictions (Phase 1 + 2B): AUC 0.60
- Smarter execution (Phase 2A): Regime-aware thresholds
- **Target**: Sharpe 1.0-1.2 (profitable after 0.26% transaction costs)

---

## Deliverables

1. ✅ `src/sparky/features/regime_indicators.py` - Volatility regime computation
2. ✅ `src/sparky/models/signal_aggregator.py` - RegimeAwareAggregator class
3. ✅ `scripts/backtest_regime_aware.py` - Backtest with regime overlay
4. ✅ `tests/test_regime_indicators.py` - Test regime classification logic
5. ✅ `results/regime_aware/backtest_2024_2025.json` - Performance results
6. ✅ Research log entry with regime-aware Sharpe comparison

---

## Timeline

**Conditional on Phase 1 Success (AUC ≥ 0.57):**
- Hour 0-1: Implement regime indicator computation
- Hour 1-2.5: Build RegimeAwareAggregator
- Hour 2.5-3.5: Backtest on 2024-2025 holdout
- Hour 3.5-4: Validate, document, log results

**Total**: 4 hours

**Next Step:** If Sharpe ≥ 0.95 → Execute Phase 2B (volume features) → If combined Sharpe ≥ 1.0 → Build paper trading infrastructure
