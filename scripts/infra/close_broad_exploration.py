"""Close out broad_exploration_20260218 orchestrator.

Superseded by parallel agent program (eth_strategies + btc_deep).
Unspent budget redistributed to agents A ($35), B ($35), C ($20).

Run: .venv/bin/python scripts/infra/close_broad_exploration.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("workflows/state/orchestrator_broad_exploration_20260218.json")

if not STATE_FILE.exists():
    print(f"State file not found: {STATE_FILE}")
    exit(1)

with open(STATE_FILE) as f:
    state = json.load(f)

print(f"Current status: {state['status']}")
print(f"Sessions: {state['session_count']}, Cost: ${state['total_cost_usd']:.2f}")
print(f"Best: Sharpe={state['best_result'].get('sharpe', 'N/A')}")

state["status"] = "done"
state["gate_message"] = (
    "Superseded by parallel agent program (agents A/B/C). "
    f"Closed {datetime.now(timezone.utc).isoformat()}. "
    f"Unspent budget redistributed to eth_strategies ($35), btc_deep ($35), portfolio_advanced ($20)."
)

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

print("Updated status to 'done'. Budget note added.")
