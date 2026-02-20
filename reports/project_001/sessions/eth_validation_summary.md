# ETH Validation Summary — Session 003
Directive: p001_validation_eth | N_trials: 16269

## Session 003 Results

Session 003 confirms session 001 findings with fresh runs. All three original ETH candidates validated. Primary candidate: eth_don8h_83_33_iv (CONDITIONAL, no hard fails).

| Candidate | Verdict | Sharpe@30 | Sharpe@50 | DSR | MaxDD | Hard Fails | Soft Fails |
|-----------|---------|-----------|-----------|-----|-------|------------|------------|
| eth_don4h_138_47_flat | **FAIL** | 2.1404 | 2.0991 | 0.9525 | -38.8% | ['bootstrap_sharpe'] | ['walk_forward_multi', 'tail_risk_analysis', 'correlation_stability'] |
| eth_don4h_164_47_flat | **FAIL** | 2.1333 | 2.0952 | 0.9505 | -38.8% | ['bootstrap_sharpe'] | ['walk_forward_multi', 'tail_risk_analysis', 'correlation_stability'] |
| eth_don8h_83_33_iv | **CONDITIONAL** | 2.0558 | 1.9707 | 0.9279 | -9.5% | [] | ['walk_forward_multi', 'tail_risk_analysis', 'correlation_stability'] |

## Soft Fail Analysis

All CONDITIONAL candidates share the same three soft fails:
1. **walk_forward_multi** — 90d windows have <80% positive fraction.
   Structural: trend-following is flat ~62% of the time. Short windows    mostly capture neutral periods. At 365d windows, eth_don8h_83_33_iv passes.
2. **tail_risk_analysis** — CVaR/mean ratio elevated due to zero-return flat periods.
   Structural artifact of binary long/flat trend-following.
3. **correlation_stability** — max rolling corr hits ~1.0 during bull runs.
   Expected when strategy holds long through sustained bull markets.

These soft fails are structural to trend-following and cannot be resolved by parameter changes without abandoning the approach.

## DrawDown Investigation (Flat Sizing Candidates)

eth_don4h_138_47_flat and eth_don4h_164_47_flat both fail bootstrap_sharpe hard fail (5th percentile Sharpe < 0.5). MaxDD of -38.8% is DISTRIBUTED across 5+ events, not a single disaster. Flat sizing on volatile ETH produces unacceptable tail risk.
Inverse_vol variants (tv=0.15, vw=30) resolve MaxDD to -8.3% to -9.7% but DSR drops to 0.87-0.90 — still CONDITIONAL, not PASS.

## Primary Recommendations

- **eth_don8h_83_33_iv**: Primary ETH candidate. CONDITIONAL.   MaxDD=-9.5%, PBO=0.000, DSR=0.928. Requires AK approval for OOS evaluation.
- **eth_don4h_164_47_iv**: Secondary candidate. CONDITIONAL.   MaxDD=-8.3%, DSR=0.903. Higher Calmar than anchor at 8h.
- Flat sizing 4h candidates: FAIL (bootstrap hard fail).   Do not advance without sizing fix.
