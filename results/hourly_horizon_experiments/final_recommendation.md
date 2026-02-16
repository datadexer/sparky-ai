# Multi-Horizon XGBoost Comparison Report

## Experiment Summary

Trained XGBoost (max_depth=5, n_estimators=200, lr=0.05) on hourly BTC features
across 4 target horizons. Same hyperparameters for all horizons to isolate target effect.

## Results Table (Non-Overlapping Evaluation)

| Metric | 1h | 4h | 24h | exec24h |
|--------|----|----|-----|---------|
| Val Accuracy | 0.5416 | 0.5306 | 0.4932 | 0.4945 |
| Val ROC-AUC | 0.5549 | 0.5430 | 0.4979 | 0.4889 |
| Val F1 | 0.5831 | 0.5479 | 0.5455 | 0.5570 |
| Test Accuracy | 0.5357 | 0.5616 | 0.5288 | 0.5342 |
| Test ROC-AUC | 0.5521 | 0.5906 | 0.5566 | 0.5301 |
| Val-Test AUC Gap | 0.0028 | 0.0476 | 0.0586 | 0.0412 |
| Effective Train Samples | 35,000 | 8,766 | 1,461 | 1,461 |
| Leakage Check | PASS | PASS | PASS | PASS |
| Val Class Balance | 0.502 | 0.503 | 0.488 | 0.486 |

## Standard vs Non-Overlapping Evaluation

| Horizon | Val AUC (standard) | Val AUC (nonoverlap) | Inflation |
|---------|-------------------|---------------------|-----------|
| 1h | 0.5549 | 0.5549 | +0.0000 |
| 4h | 0.5378 | 0.5430 | -0.0052 |
| 24h | 0.5048 | 0.4979 | +0.0068 |
| exec24h | 0.4938 | 0.4889 | +0.0048 |

## Top-5 Features by Horizon

### 1h

1. `rsi_6h` (0.0684)
2. `higher_highs_lower_lows_5h` (0.0539)
3. `ema_ratio_20h` (0.0526)
4. `momentum_4h` (0.0490)
5. `momentum_168h` (0.0447)

### 4h

1. `rsi_6h` (0.0592)
2. `momentum_4h` (0.0570)
3. `higher_highs_lower_lows_5h` (0.0500)
4. `hour_of_day` (0.0482)
5. `price_acceleration_10h` (0.0467)

### 24h

1. `day_of_week` (0.0637)
2. `vwap_deviation_24h` (0.0603)
3. `momentum_168h` (0.0599)
4. `realized_vol_24h` (0.0594)
5. `atr_14h` (0.0547)

### exec24h

1. `day_of_week` (0.0634)
2. `vwap_deviation_24h` (0.0615)
3. `momentum_168h` (0.0596)
4. `realized_vol_24h` (0.0573)
5. `atr_14h` (0.0558)

## Decision Matrix Scores

Weights: {'nonoverlap_val_auc': 0.3, 'effective_samples': 0.25, 'val_test_consistency': 0.15, 'feature_coherence': 0.15, 'trading_utility': 0.15}

| Criterion | Weight | 1h | 4h | 24h | exec24h |
|-----------|--------| --- | --- | --- | --- |
| nonoverlap_val_auc | 30% | 0.549 | 0.430 | 0.000 | 0.000 |
| effective_samples | 25% | 1.000 | 0.877 | 0.146 | 0.146 |
| val_test_consistency | 15% | 0.982 | 0.683 | 0.609 | 0.725 |
| feature_coherence | 15% | 0.456 | 0.395 | 0.425 | 0.423 |
| trading_utility | 15% | 0.300 | 0.600 | 0.900 | 1.000 |
| **TOTAL** | | **0.675** | **0.600** | **0.327** | **0.359** |

## Recommendation

**Scenario: A**
**Winner: 1h** (weighted score: 0.675)

1h horizon wins on ROC-AUC with the most independent training samples (~35K).
Recommendation: Use 1h model. Aggregate 24 hourly predictions into daily confidence.
Daily signal = mean(P(up) for last 24 hours) > 0.5 → LONG

## Next Steps

1. Run holdout evaluation on 1h model (ONE test only)
2. If holdout confirms, build signal aggregation pipeline (1h → daily)
3. Proceed with cross-asset expansion for robustness