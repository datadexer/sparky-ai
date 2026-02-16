#!/usr/bin/env python3
"""Send look-ahead bias bug report to CEO agent via coordination system."""

from pathlib import Path
from coordination import CoordinationAPI, MessagePriority

# Initialize coordination API
api = CoordinationAPI(Path("/home/akamath/sparky-ai"))

# Send CRITICAL message to CEO
api.send_message(
    from_agent="human-ak",
    to_agent="ceo",
    subject="CRITICAL BUG: Look-Ahead Bias - Sharpe 2.556 Claim is FALSE",
    body="""
CRITICAL BUG FOUND: The claimed Sharpe 2.556 for "momentum > 0.05" strategy is COMPLETELY FALSE.

TRUE PERFORMANCE: Sharpe -0.27 (loses money!)

ROOT CAUSE: Look-ahead bias in scripts/option3_strategic_pivot.py
- Signals at time T use close[T]
- Strategy captures returns[T] which is (close[T] - close[T-1]) / close[T-1]
- This uses close[T] to predict a return that ENDS at close[T]
- Classic look-ahead bias

THE FIX: One line change in run_experiment() function (line 44):
  CHANGE: strategy_returns = positions * returns_holdout
  TO:     strategy_returns = positions * returns_holdout.shift(-1).fillna(0)

PROOF: Run `python prove_bug.py` to verify:
- Buggy approach: Sharpe 2.5564 (matches claim)
- Correct approach: Sharpe -0.2725 (actual performance)
- Degradation: -2.83 Sharpe, -20.15% return

IMPACT:
- All Option 3 results are INVALID (both from this bug AND data snooping)
- "Breakthrough" strategy actually loses money
- Combined with p-hacking issue, Phase 3 validation has completely failed

DOCUMENTATION:
- Full bug report: roadmap/BUG_REPORT_LOOKAHEAD_BIAS.md
- Quick summary: LOOKAHEAD_BUG_SUMMARY.md
- Proof scripts: prove_bug.py, test_momentum_bug.py
- Results: results/experiments/bug_proof.json

RECOMMENDATION:
1. Fix the bug for code hygiene (instructions in bug report)
2. DO NOT re-run Option 3 (data snooping makes it invalid anyway)
3. Proceed to Phase 3 data expansion (get 10K+ samples)
4. Test strategies on truly unseen 2026 data when available

ACTION REQUIRED: Read full bug report and decide on path forward.
""",
    priority=MessagePriority.CRITICAL
)

print("✓ CRITICAL bug report sent to CEO agent")
print("\nMessage details:")
print("  From: human-ak")
print("  To: ceo")
print("  Subject: CRITICAL BUG: Look-Ahead Bias - Sharpe 2.556 Claim is FALSE")
print("  Priority: CRITICAL")
print("\nCEO will see this message on next startup checklist.")
