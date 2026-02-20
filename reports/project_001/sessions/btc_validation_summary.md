# BTC Validation Summary — p001_validation_btc Session 3
## Candidate Results
| ID | Sharpe@30 | Sharpe@50 | DSR | MaxDD | N_trades | Verdict | Val_Battery |
|----|-----------|-----------|-----|-------|----------|---------|-------------|
| btc_don4h_160_25_iv | 2.319 | 2.107 | 0.9999 | -0.124 | 1421 | PASS | CONDITIONAL |
| btc_don4h_60_20_iv | 2.042 | 1.808 | 0.9977 | -0.149 | 941 | PASS | CONDITIONAL |
| btc_don8h_82_20_iv | 2.220 | 2.088 | 0.9998 | -0.197 | 1074 | PASS | CONDITIONAL |

## Sub-Period Breakdown (Passing Candidates)

### btc_don4h_160_25_iv
| Period | Sharpe | MaxDD |
|--------|--------|-------|
| full | 2.319 | -0.124 |
| 2017+ | 1.603 | -0.124 |
| 2020+ | 1.200 | -0.124 |

### btc_don4h_60_20_iv
| Period | Sharpe | MaxDD |
|--------|--------|-------|
| full | 2.042 | -0.149 |
| 2017+ | 1.464 | -0.149 |
| 2020+ | 1.236 | -0.149 |

### btc_don8h_82_20_iv
| Period | Sharpe | MaxDD |
|--------|--------|-------|
| full | 2.220 | -0.197 |
| 2017+ | 1.687 | -0.197 |
| 2020+ | 1.404 | -0.197 |

## Hard/Soft Fails

**btc_don4h_160_25_iv**: hard=[], soft=['walk_forward_multi', 'tail_risk_analysis', 'correlation_stability']

**btc_don4h_60_20_iv**: hard=[], soft=['walk_forward_multi', 'tail_risk_analysis', 'correlation_stability']

**btc_don8h_82_20_iv**: hard=[], soft=['walk_forward_multi', 'tail_risk_analysis', 'correlation_stability']
