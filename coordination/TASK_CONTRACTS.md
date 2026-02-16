# Task Contracts - Binding Execution Agreements

## Purpose
CEO agent must sign contracts BEFORE starting work. Prevents premature pivoting.

## Active Contracts

### CONTRACT #001: ML + Regime Detection Research
**Status**: COMPLETED (superseded by CONTRACT #002)
**Signed**: 2026-02-16 15:48 UTC
**Assigned to**: CEO
**Estimated effort**: 7-9 hours
**Hard deadline**: 2026-02-17 EOD

**Binding Commitments**:
1. ✅ I will complete Phase 1 (Cross-Asset ML) - 2 hours minimum
2. ✅ I will complete Phase 2A (Regime Detection) - 3-4 hours minimum
3. ✅ I will complete Phase 2B (Volume Features) - 2 hours minimum
4. ✅ I will NOT pivot to other approaches until completing all 3 phases
5. ✅ I will NOT discuss deployment until Sharpe ≥1.0 achieved
6. ✅ I will report actual elapsed time in 15-minute increments
7. ✅ I will update RESEARCH_LOG.md after each phase with results

**Allowed Early Termination Conditions**:
- Phase 1 produces AUC <0.50 (catastrophic failure, below random)
- Unrecoverable technical error (system crash, data corruption)
- Human intervention (AK explicitly cancels contract)

**NOT Allowed Termination Reasons**:
- ❌ "This approach isn't working" after <5 configs tested
- ❌ "I found something better" (finish contract first, then propose)
- ❌ "Results are marginal" (complete all configs before judging aggregate)
- ❌ "I want to try [different approach]" (contract specifies scope)
- ❌ Presenting OPTION A/B/C/D menus (continue working, don't ask what to do)

**Escalation Protocol**:
- If CEO attempts to break contract → RBM sends WARNING
- If CEO persists → RBM escalates to HUMAN (AK)
- If CEO completes contract → NEW contract can be negotiated

**Deliverables**:
- [ ] Phase 1: Cross-asset ML results logged to RESEARCH_LOG.md
- [ ] Phase 2A: Regime detection results logged to RESEARCH_LOG.md
- [ ] Phase 2B: Volume features results logged to RESEARCH_LOG.md
- [ ] Combined validation: Sharpe metric with yearly walk-forward
- [ ] Time tracking: Actual hours in TIME_TRACKING.md

**Success Criteria**:
- Combined approach achieves Sharpe ≥1.0 → Contract fulfilled, SUCCESS
- Combined approach achieves Sharpe 0.85-1.0 → Contract fulfilled, MARGINAL
- Combined approach achieves Sharpe <0.85 → Contract fulfilled, escalate to RBM

---

### CONTRACT #002: Comprehensive ML Research Sprint
**Status**: COMPLETED (22 configs tested, no validated ML alpha, escalated to RBM)
**Signed**: 2026-02-16 17:36 UTC
**Assigned to**: CEO
**Estimated effort**: 12-16 hours
**Hard deadline**: 48 hours from contract start

**Context**: Previous research tested only Donchian breakout family (2 configs) and declared failure prematurely. System has been reformed to require deeper exploration. Available data: 115K BTC hourly rows, 80K ETH hourly rows, on-chain metrics, macro features. ALL training/validation must use data BEFORE 2024-06-01 (embargo boundary). Data after 2024-07-01 is strictly OOS.

**Binding Commitments**:
1. ✅ Phase A: Tree ensemble sweep (CatBoost/XGBoost/LightGBM) with ≥10 hyperparameter configs on IN-SAMPLE data only
2. ✅ Phase B: Feature ablation — test with/without on-chain, with/without macro, with/without cross-asset features (≥6 feature set variants)
3. ✅ Phase C: Regime-aware models — ≥3 regime detection methods (volatility threshold, HMM, trend-based) × ≥2 base models = ≥6 configs
4. ✅ Phase D: If any TIER 2+ result, request OOS approval from RBM, then run single OOS evaluation
5. ✅ I will NOT touch data after 2024-07-01 until OOS is explicitly approved
6. ✅ I will use TaskTimer for all work sessions
7. ✅ I will NOT present option menus — I will keep working until contract complete
8. ✅ I will run system_health_check.sh before spawning any sub-agent
9. ✅ I will NEVER have more than 2 sub-agents running simultaneously

**Allowed Early Termination**:
- TIER 1 result found and OOS-validated (SUCCESS)
- All phases complete with only TIER 4-5 results after 22+ configs (HONEST NEGATIVE)
- Human intervention (AK cancels)
- System health CRITICAL and cleanup doesn't resolve

**NOT Allowed Termination Reasons**:
- ❌ "This approach isn't working" after <5 configs tested
- ❌ "I found something better" (finish contract first, then propose)
- ❌ "Results are marginal" (complete all configs before judging aggregate)
- ❌ "I want to try [different approach]" (contract specifies scope)
- ❌ Presenting OPTION A/B/C/D menus (continue working, don't ask what to do)

**Success Criteria**:
- TIER 1 (Sharpe ≥1.0 in-sample, validated): Request OOS → deploy decision
- TIER 2 (Sharpe ≥0.7 in-sample, validated): Request OOS → paper trade if confirms
- TIER 3 (Sharpe ≥0.4, shows edge): Continue with Phase C regime overlay
- TIER 4-5 after all phases: Honest report, propose next research direction

---

### CONTRACT #003: Regime Ensemble Validation Sprint
**Status**: ACTIVE
**Signed**: 2026-02-16 18:20 UTC
**Assigned to**: CEO
**Estimated effort**: 4-8 hours
**Hard deadline**: 24 hours from contract start

**Context**: CONTRACT #002 produced a claimed Sharpe 2.656 from a "Regime-Weighted Ensemble" strategy. RBM review flagged this as 70%+ likely overfitting (7th attempt, 244% improvement, missing 2018, no validation protocol, contradicts earlier position sizing failure). Additionally, ALL approaches tested in `scripts/validate_regime_approaches.py` show 0 trades in 2018 — a systemic issue. This contract requires rigorous validation before any result can be trusted.

**Key files**:
- `scripts/validate_regime_approaches.py` — the script that produced the results
- `results/validation/regime_approaches_comparison.json` — raw results
- `src/sparky/models/regime_weighted_ensemble.py` — the model
- `src/sparky/models/regime_hmm.py` — HMM approach (also high Sharpe)
- `src/sparky/features/regime_indicators.py` — regime detection features
- `scripts/backtest_regime_aware.py` — the CatBoost + regime backtest

**Binding Commitments**:
1. ✅ **Step 1: 2018 Investigation** — Explain why ALL approaches have 0 trades in 2018. Is this a bug in the backtest framework? Data issue? If 2018 is legitimately excluded, the mean Sharpe is computed over 5 years not 6 — re-derive correct statistics.
2. ✅ **Step 2: Multi-Seed Stability** — Re-run the top 3 approaches (Regime-Weighted Ensemble, HMM 2-state, HMM 3-state) with seeds [42, 123, 456, 789, 1337]. Report mean and std of Sharpe across seeds. If std > 0.3 * mean, result is UNSTABLE.
3. ✅ **Step 3: Bootstrap 95% CI** — For each top approach, compute bootstrap 95% CI on the Sharpe ratio (1000 resamples of yearly results). Report lower bound. If CI_lower < 0.5, result is NOT statistically significant.
4. ✅ **Step 4: Leakage Audit** — Verify no look-ahead bias: (a) features used for regime detection don't use future returns, (b) position sizing doesn't use future volatility, (c) signal generation at time T uses only data from T-1 and earlier.
5. ✅ **Step 5: Anti-Flip-Flop Resolution** — Position sizing with the SAME regime concept failed (Sharpe 0.715). Regime-weighted ensemble with the SAME concept claims 2.656. Explain the specific mechanism difference OR if no valid explanation, downgrade the result.
6. ✅ **Step 6: Corrected Statistics** — After steps 1-5, compute the CORRECTED Sharpe ratio including: (a) proper inclusion/exclusion of 2018 with justification, (b) multi-seed median (not best seed), (c) bootstrap CI, (d) multiple testing correction (Bonferroni: divide alpha by number of approaches tested).
7. ✅ I will NOT present option menus — I will execute all 6 steps sequentially
8. ✅ I will NOT touch data after 2024-07-01 (OOS boundary)
9. ✅ I will save all results to `results/validation/contract_003/` directory

**Allowed Early Termination**:
- Step 1 reveals catastrophic framework bug that invalidates ALL prior results
- Human intervention (AK cancels)

**NOT Allowed Termination Reasons**:
- ❌ "Validation is taking too long" (complete all 6 steps)
- ❌ "Results look bad, no need to continue" (complete all 6 steps for the record)
- ❌ Presenting option menus
- ❌ Skipping steps because earlier steps showed problems

**Deliverables**:
- [ ] Step 1: `results/validation/contract_003/2018_investigation.json`
- [ ] Step 2: `results/validation/contract_003/multi_seed_stability.json`
- [ ] Step 3: `results/validation/contract_003/bootstrap_ci.json`
- [ ] Step 4: `results/validation/contract_003/leakage_audit.json`
- [ ] Step 5: `results/validation/contract_003/anti_flip_flop.json`
- [ ] Step 6: `results/validation/contract_003/corrected_statistics.json`
- [ ] Summary: `results/validation/contract_003/VALIDATION_VERDICT.md`

**Success Criteria**:
- Corrected Sharpe ≥ 1.0 with CI_lower ≥ 0.5 and multi-seed stable: VALIDATED — request OOS approval
- Corrected Sharpe 0.5-1.0 with CI_lower ≥ 0.3: MARGINAL — document honestly, propose improvements
- Corrected Sharpe < 0.5 or CI_lower < 0.0 or unstable: DEBUNKED — accept honest negative, move on

---

### CONTRACT #004: Feature-First ML Research Sprint (Post-Refactor)
**Status**: ACTIVE
**Signed**: 2026-02-16 UTC
**Assigned to**: CEO
**Estimated effort**: 8-12 hours
**Hard deadline**: 48 hours from contract start

**Context**: v3 structural refactor is complete. CEO now has: enforced data loader (holdout auto-truncated), experiment DB (dedup, query), GPU in all train scripts, timeout decorator, and two-stage sweep script. This contract directs the CEO to use the new infrastructure for a systematic ML research sprint.

**New Infrastructure Available**:
- `from sparky.data.loader import load` — auto-holdout enforcement
- `from sparky.tracking.experiment_db import get_db, is_duplicate, log_experiment` — dedup + logging
- `from sparky.oversight.timeout import with_timeout` — 15 min per config
- `scripts/sweep_two_stage.py` — two-stage sweep scaffold
- `scripts/analyze_features.py` — feature importance analysis
- `scripts/train_regime_aware.py` — regime-aware scaffold

**Binding Commitments**:
1. [ ] **Step 1: Feature Analysis** — Run `scripts/analyze_features.py`, identify top 20 features, save results
2. [ ] **Step 2: Two-Stage Sweep** — Run `scripts/sweep_two_stage.py` with top 20 features. Complete Stage 1 screening (all configs). Top 5 → Stage 2 walk-forward.
3. [ ] **Step 3: Regime-Aware** — Complete the TODO in `scripts/train_regime_aware.py`: implement at least 2 regime methods, wire walk-forward evaluation
4. [ ] **Step 4: Ensemble** — If any TIER 2+ results, test ensemble of top 3 configs
5. [ ] I will use experiment DB for ALL experiments (no duplicates)
6. [ ] I will use data loader for ALL data loading (no raw pd.read_parquet)
7. [ ] I will NOT present option menus — just work through steps 1-4 sequentially
8. [ ] I will check `get_summary(get_db())` at session start to avoid repeating work

**Allowed Early Termination**:
- TIER 1 result found and validated
- All steps complete with only TIER 4-5 results after 20+ configs (HONEST NEGATIVE)
- Human intervention

**Success Criteria**:
- TIER 1 (Sharpe ≥1.0 validated): Request OOS approval
- TIER 2 (Sharpe ≥0.7 validated): Request OOS → paper trade if confirms
- TIER 3 (Sharpe ≥0.4): Continue with next contract
- TIER 4-5 after all steps: Honest report, propose pivot

---

## Contract Template

```markdown
### CONTRACT #XXX: [Task Name]
**Status**: [DRAFT|ACTIVE|COMPLETED|BROKEN]
**Signed**: [UTC timestamp]
**Assigned to**: [Agent ID]
**Estimated effort**: [Hours]
**Hard deadline**: [Date]

**Binding Commitments**:
1.
2.
3.

**Allowed Early Termination Conditions**:
-

**NOT Allowed Termination Reasons**:
-

**Deliverables**:
- [ ]
- [ ]

**Success Criteria**:
-
```

---

## Enforcement Rules

1. **RBM monitors active contracts** - Check CEO progress every 2 hours
2. **Contract violations trigger escalation** - Human (AK) notified immediately
3. **No new contracts until current complete** - CEO cannot negotiate new work
4. **Honest time tracking mandatory** - No "Day N" labels, 15-min increments only
5. **Deliverables are blocking** - Cannot mark contract complete without all deliverables

---

## Historical Contracts

### CONTRACT #000: Donchian Strategy Testing (BROKEN)
**Status**: ❌ BROKEN - CEO abandoned STRATEGY_REPORT.md plan after <10 min
**Signed**: 2026-02-16 05:18 UTC (implicit in STRATEGY_REPORT.md)
**Outcome**: CEO tested 7 rule-based strategies (15 min), ignored ML + regime detection plan
**Violation**: Pivoted to different approach without completing contracted work
**Consequence**: Corrective directive issued, new explicit contract (#001) created
