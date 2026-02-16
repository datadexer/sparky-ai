# Task Contracts - Binding Execution Agreements

## Purpose
CEO agent must sign contracts BEFORE starting work. Prevents premature pivoting.

## Active Contracts

### CONTRACT #001: ML + Regime Detection Research
**Status**: ACTIVE
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
