# 58-Feature Hyperparameter Sweep - Partial Results

## Executive Summary

**Result**: ML with 58 features does NOT beat simple baseline  
**Best ML**: Sharpe 0.967 (Config 3)  
**Baseline**: Sharpe 1.062 (Multi-TF Donchian), 1.243 (Single-TF)  
**Gap**: -9% vs Multi-TF, -22% vs Single-TF  

## Dataset

- **Features**: 58 (23 original + 35 new)
- **Samples**: 4,795 daily (from 115K hourly candles, 2013-2026)
- **Validation**: Yearly walk-forward 2020-2023

## Feature Categories Added

1. **Microstructure (10)**: tick_direction_ratio, candle_body_ratio, upper/lower_wick_ratio, consecutive_green/red_candles, high_low_ratio, bid_ask_imbalance_proxy, intraday_momentum_reversal, overnight_gap
2. **Multi-resolution (3)**: rsi_4h, rsi_12h, rsi_168h
3. **Regime indicators (8)**: drawdown_from_20h_high, recovery_from_20h_low, volatility_regime, volume_regime, trend_strength_adx_proxy, choppiness_index, breakout_proximity_upper/lower
4. **Cross-timeframe divergences (6)**: momentum_divergence_4h_24h, momentum_divergence_24h_168h, vol_divergence_4h_24h, rsi_divergence_4h_24h, rsi_divergence_14h_168h, price_momentum_divergence
5. **Volume-price interaction (8)**: obv, obv_rate_of_change_24h, mfi_14h, volume_surge_4h, price_volume_correlation_24h, vwap_cross, volume_exhaustion, volume_weighted_rsi

## Results (16/54 configs tested)

### Top 10 Configs

| Rank | Sharpe | Model | Depth | LR | L2 |
|------|--------|-------|-------|----|----|
| 1 | 0.967 | CatBoost | 3 | 0.01 | 5.0 |
| 2 | 0.964 | CatBoost | 3 | 0.05 | 3.0 |
| 3 | 0.922 | CatBoost | 4 | 0.01 | 5.0 |
| 4 | 0.910 | CatBoost | 3 | 0.01 | 3.0 |
| 5 | 0.820 | CatBoost | 4 | 0.01 | 3.0 |
| 6 | 0.797 | CatBoost | 3 | 0.01 | 1.0 |
| 7 | 0.757 | CatBoost | 3 | 0.03 | 3.0 |
| 8 | 0.730 | CatBoost | 4 | 0.05 | 5.0 |
| 9 | 0.722 | CatBoost | 3 | 0.05 | 5.0 |
| 10 | 0.712 | CatBoost | 4 | 0.03 | 1.0 |

### Statistics

- **Mean Sharpe**: 0.752
- **Std**: 0.187
- **Range**: 0.360 to 0.967

## Year-by-Year Breakdown (Best Config)

Config 3 (Sharpe 0.967):

| Year | Sharpe | Accuracy | Pattern |
|------|--------|----------|---------|
| 2020 | +2.016 | 55.2% | Bull: strong |
| 2021 | +1.278 | 53.4% | Choppy bull: good |
| 2022 | -2.255 | 49.9% | **Bear: catastrophic** |
| 2023 | +0.876 | 52.1% | Recovery: moderate |

## Problem Identified

**ML overfits to bull market patterns, collapses in bear markets**

- Bull years (2020-2021): Sharpe 1.5-2.9 (excellent)
- Bear year (2022): Sharpe -2.0 to -2.7 (catastrophic)
- Recovery (2023): Sharpe 0.9-1.6 (moderate)

Simple Donchian baseline is more robust:
- 2022 Donchian: Sharpe -1.539 (bad but not catastrophic)
- 2022 ML best: Sharpe -2.255 (46% worse)

## Conclusion

**58-feature expansion does NOT beat simple baseline**

1. Additional features (microstructure, regime, divergences) captured bull market patterns but failed to generalize
2. ML models learned complex non-linear relationships that broke down in 2022 bear market
3. Simple Donchian breakout (20/10 or 40/20) remains superior strategy

**Recommendation**: Accept that simple baselines are better. Do NOT pursue further ML experiments with current approach. Alternative paths:
- Ensemble of simple strategies
- Regime-specific models (train separate models for bull/bear/choppy)
- Different paradigm (reinforcement learning, portfolio optimization)

## Technical Notes

- Sweep crashed at config 17/54 (processing year 2022 for config 17)
- Crash pattern: happens during 2022 bear year processing
- Likely cause: extreme negative Sharpes causing numerical instability
- 38 configs untested (27 CatBoost + 27 LightGBM remaining)

