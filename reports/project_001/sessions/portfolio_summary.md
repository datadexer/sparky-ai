# Project 001 Portfolio Construction — Session 001 Summary

## Objective
Build and screen BTC-ETH portfolios from validated survivors. Identify optimal portfolio allocation.

## Survivors Used
- **BTC**: btc_don4h_160_25_iv (S30=2.319), btc_don4h_60_20_iv (S30=2.042), btc_don8h_82_20_iv (S30=2.220)
- **ETH**: eth_don8h_83_33_iv (S30=2.056), eth_don4h_164_47_iv (S30=2.001)

## Portfolios Tested: 8 (5 Tier 1, 3 Tier 2)
All 8 passed initial screen (Sharpe >= 1.0, MaxDD > -30%).

## Top Portfolios by Sharpe@30bps

| Portfolio | S@30 | S@50 | MaxDD | DSR | S2020+ |
|-----------|------|------|-------|-----|--------|
| btc82_50_eth83_50_8h | 2.151 | 2.021 | -8.9% | 0.960 | 1.860 |
| btc160_50_eth164iv_50 | 2.139 | 1.928 | -7.5% | 0.953 | 1.486 |
| btc160_50_eth83_50 | 2.009 | 1.869 | -4.7% | 0.904 | 1.823 |

## Weight Sensitivity Analysis (BTC82 8h + ETH83 8h)
ETH-heavier weighting is superior in both Sharpe and DSR:

| BTC% | ETH% | S@30 | MaxDD | DSR | S2020+ |
|------|------|------|-------|-----|--------|
| 30% | 70% | **2.217** | -7.0% | **0.972** | **1.943** |
| 40% | 60% | 2.200 | -7.3% | 0.969 | 1.914 |
| 50% | 50% | 2.151 | -8.9% | 0.960 | 1.860 |
| 60% | 40% | 2.076 | -10.9% | 0.942 | 1.787 |
| 70% | 30% | 1.985 | -13.2% | 0.909 | 1.699 |

## Optimal Portfolio: BTC82 30% + ETH83 70%
- **Sharpe@30bps**: 2.217 | **Sharpe@50bps**: 2.099
- **DSR@30**: 0.972 (highest of all portfolios) | **DSR@50**: 0.944
- **MaxDD**: -7.0% (vs best individual -12.5%)
- **2020+ Sharpe**: 1.943 (robust post-COVID including 2022 bear)
- **Validation**: CONDITIONAL (hard_fails=0, soft_fails=3 — same as component strategies)
- **BTC-ETH correlation**: 0.463 (moderate, diversification benefit confirmed)

## Investigation Findings

### Regime Decomposition
| Leg | Bull S | Bear S | Sideways S |
|-----|--------|--------|-----------|
| BTC82 8h | 2.826 | **-0.979** | 2.355 |
| ETH83 8h | 2.606 | **+0.465** | 2.071 |
| BTC160 4h | 3.076 | **-1.005** | 2.280 |

**ETH83 has POSITIVE bear regime Sharpe (+0.465) — key diversification benefit.**
BTC both strategies go negative in bear markets. ETH leg buffers portfolio drawdown.

### Edge Attribution
- BTC82: signal_edge=2.68 (121% of full Sharpe), sizing_edge=0.28 — strong signal
- ETH83: signal_edge=1.38 (67% of full Sharpe), sizing_edge=-0.02 — signal-driven
- Both strategies have genuine signal edge (>15% threshold met comfortably)

## Key Questions Answered
1. **Diversification reduces MaxDD?** YES — -7.0% vs -12.5% (btc160) or -19.7% (btc82)
2. **ETH saves portfolio in 2022 bear?** YES — ETH83 bear Sharpe=+0.465 while BTC goes negative
3. **Portfolio Sharpe > best individual?** NO — diversification preserves/reduces drawdown, not Sharpe
4. **Optimal BTC/ETH ratio?** BTC30/ETH70 for pure 8h portfolio

## Recommendation
**btc82_30pct_eth83_70pct** is the primary portfolio recommendation:
- Tier 1 criteria met: Sharpe=2.217 ≥ 1.0, MaxDD=-7.0% < 50% threshold
- DSR=0.972 is the highest of all portfolios tested (significant after 16,269 trials)
- Robust to 50bps stress: S@50=2.099 (still strong)
- 2020+ Sharpe=1.943 (no regime collapse)
- Both component strategies have independently validated CONDITIONAL status

## Next Steps (pending AK approval)
1. Walk-forward validation on optimal portfolio
2. OOS evaluation (requires explicit AK written approval)
