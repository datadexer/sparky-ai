# Sparky AI - Session Memory

## Workflow Rules
- **Always spin out validation agents** to audit experiment results after training runs complete. Do this proactively without being asked.
- **Never block on validation agents** — run experiments in parallel. Pattern: EXPERIMENT → LOG → START NEXT → check inbox between tasks.
- **Aggressive parallelism**: Use sub-agents for independent work streams (data fetching, feature engineering) while doing other work.
- **Never ask for permission to proceed** — just keep moving through the task queue.

## Project Patterns
- XGBoost crashes on `inf` values in features — always replace inf with NaN before training
- `volume_momentum_30h` feature produces inf values in hourly data (division by near-zero volume)
- `use_label_encoder` param in XGBoostModel triggers harmless warning — cosmetic only
- JSON serialization of numpy types: use `default=` with `raise TypeError` for unhandled types, never return `obj` unchanged (causes circular reference)
- Coordination CLI prefix: `PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py`
- Always run `startup ceo` at session start, `inbox-read ceo` after reading, `task-start`/`task-done` around work
