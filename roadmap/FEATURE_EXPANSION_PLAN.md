# Feature Expansion Plan — Phase 3

**Current**: 23 hourly features → mean Sharpe 0.050 (far below baseline 1.062)

**Goal**: Expand to 50-100 features across multiple domains to beat Donchian baseline

---

## Current Feature Coverage (23 features)

**Technical (9)**: rsi_14h, rsi_6h, ema_ratio_20h, macd_line, macd_histogram, atr_14h, bb_bandwidth_20h, bb_position_20h, distance_from_sma_200h

**Momentum (5)**: momentum_4h, momentum_24h, momentum_168h, momentum_quality_30h, price_acceleration_10h

**Volume (3)**: volume_ma_ratio_20h, volume_momentum_30h, vwap_deviation_24h

**Volatility (2)**: realized_vol_24h, vol_clustering_24h

**Microstructure (2)**: intraday_range, higher_highs_lower_lows_5h

**Temporal (2)**: hour_of_day, day_of_week

---

## NEW Feature Families to Add

### 1. Order Flow & Microstructure (10 features)
**Rationale**: Hourly data reveals intraday patterns invisible in daily data

- `tick_direction_ratio_24h`: % of up-ticks (close > open) in last 24h
- `bid_ask_imbalance_proxy`: (high-close) / (high-low) ratio → sell pressure proxy
- `candle_body_ratio`: abs(close-open) / (high-low) → strength of directional moves
- `upper_wick_ratio`: (high-close) / (high-low) → rejection of highs
- `lower_wick_ratio`: (close-low) / (high-low) → rejection of lows
- `consecutive_green_candles`: count of consecutive close > open
- `consecutive_red_candles`: count of consecutive close < open
- `high_low_ratio_20h`: rolling 20h avg(high/low) → range expansion/contraction
- `overnight_gap`: (open_t - close_t-1) / close_t-1 → session gaps (crypto 24/7 but exchanges differ)
- `intraday_momentum_reversal`: sign(close-open) != sign(close_t-1-open_t-1)

### 2. Multi-Resolution Features (12 features)
**Rationale**: Capture patterns at multiple timescales (4h, 12h, 24h, 7d)

- `rsi_4h`, `rsi_12h`, `rsi_24h`, `rsi_168h`: RSI at 4 resolutions
- `momentum_12h`, `momentum_72h`: fill gaps between existing 4h/24h/168h
- `ema_cross_4h`: (EMA-9 - EMA-21) / EMA-21 at 4h
- `ema_cross_24h`: (EMA-9 - EMA-21) / EMA-21 at 24h
- `bb_squeeze_4h`: BB bandwidth < 20th percentile at 4h → breakout setup
- `bb_squeeze_24h`: BB bandwidth < 20th percentile at 24h
- `vol_regime_4h`: realized_vol_4h / realized_vol_24h → short-term vol spike
- `trend_alignment`: sign(momentum_4h) == sign(momentum_24h) == sign(momentum_168h)

### 3. Market Regime Indicators (8 features)
**Rationale**: Different features work in different regimes

- `drawdown_from_20h_high`: (high_20h - close) / high_20h
- `recovery_from_20h_low`: (close - low_20h) / low_20h
- `volatility_regime`: percentile_rank(realized_vol_24h, window=168h)
- `volume_regime`: percentile_rank(volume, window=168h)
- `trend_strength_adx_proxy`: max(momentum_24h, -momentum_24h) / realized_vol_24h
- `choppiness_index`: 1 - abs(momentum_168h) / sum(abs(returns_168h))
- `breakout_proximity_upper`: (close - sma_200h) / (max_200h - sma_200h)
- `breakout_proximity_lower`: (sma_200h - close) / (sma_200h - min_200h)

### 4. Cross-Timeframe Divergences (6 features)
**Rationale**: When short-term and long-term disagree, regime shift likely

- `momentum_divergence_4h_24h`: momentum_4h - momentum_24h
- `momentum_divergence_24h_168h`: momentum_24h - momentum_168h
- `vol_divergence_4h_24h`: realized_vol_4h - realized_vol_24h
- `volume_divergence_4h_24h`: volume_ma_ratio_4h - volume_ma_ratio_24h
- `rsi_divergence_4h_24h`: rsi_4h - rsi_24h → overbought/oversold mismatch
- `price_momentum_divergence`: sign(close-open) != sign(momentum_24h) → intraday reversal

### 5. Volume-Price Interaction (8 features)
**Rationale**: Volume confirms or contradicts price moves

- `volume_weighted_rsi`: RSI computed on VWAP instead of close
- `volume_surge_4h`: volume > 2 * volume_ma_20h
- `volume_exhaustion`: volume > 3 * volume_ma_20h AND momentum_4h < 0
- `price_volume_correlation_24h`: rolling correlation(returns_1h, volume_1h, window=24)
- `vwap_cross`: (close - vwap_24h) / close → VWAP breakout
- `vwap_momentum`: (vwap_24h - vwap_168h) / vwap_168h
- `accumulation_distribution`: cumsum(((close-low)-(high-close))/(high-low) * volume)
- `obv_momentum_24h`: (obv - obv_24h_ago) / obv_24h_ago → OBV rate of change

### 6. Statistical / Advanced (6 features)
**Rationale**: Capture non-linear patterns ML might exploit

- `return_skewness_24h`: skewness of 1h returns over 24h window
- `return_kurtosis_24h`: kurtosis → fat tails detection
- `hurst_exponent_168h`: trend persistence vs mean reversion
- `entropy_24h`: Shannon entropy of binned returns → predictability
- `fractal_dimension_24h`: 1/H → roughness of price series
- `autocorrelation_lag1`: correlation(return_t, return_t-1)

---

## Implementation Priority

### Phase 3A (IMMEDIATE — while sweep runs)
**Goal**: Add 20 high-impact features, re-run sweep

1. Order flow basics (5 features): tick_direction_ratio, candle_body_ratio, upper/lower_wick_ratio, consecutive_green/red
2. Multi-resolution RSI (3 features): rsi_4h, rsi_12h, rsi_168h
3. Regime indicators (4 features): drawdown_from_high, recovery_from_low, volatility_regime, volume_regime
4. Cross-timeframe divergences (4 features): momentum_divergence_4h_24h, momentum_divergence_24h_168h, vol_divergence, rsi_divergence
5. Volume-price (4 features): volume_surge, price_volume_correlation, vwap_cross, accumulation_distribution

**Files**:
- `scripts/add_microstructure_features.py`
- `scripts/add_regime_features.py`
- Update `scripts/prepare_hourly_features.py`

### Phase 3B (NEXT — if 3A improves results)
**Goal**: Add remaining 30 features for 70+ total

- All remaining multi-resolution features
- Statistical features (skewness, kurtosis, entropy, Hurst)
- Advanced volume features (OBV, volume-weighted indicators)

### Phase 3C (CONDITIONAL — if still below baseline)
**Goal**: Alternative data integration

- Cross-asset features (ETH correlation, alt-coin breadth)
- Derivatives data (funding rates, open interest)
- On-chain metrics (hourly resampled if available)

---

## Success Metrics

**TIER 3 — Continue**: Sharpe ≥0.4 after 20 new features → add 30 more
**TIER 4 — Pivot**: Sharpe <0.4 → try ensemble methods or different model families
**TIER 5 — Abandon**: Sharpe <0.2 after 70+ features → accept simple baselines superior

---

## Files to Create

1. `src/sparky/features/microstructure.py` — order flow features
2. `src/sparky/features/regime.py` — regime detection features
3. `src/sparky/features/statistical.py` — entropy, Hurst, skew/kurtosis
4. `scripts/add_microstructure_features.py` — compute + append to feature matrix
5. `scripts/add_regime_features.py` — compute + append
6. Update `tests/test_microstructure.py`, `tests/test_regime.py`
