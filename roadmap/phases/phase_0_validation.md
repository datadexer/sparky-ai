# Phase 0: Validation Bedrock

## Purpose
Build a standalone test suite that verifies all financial calculations against
textbook definitions. Every formula used later in the project gets validated here
first, so bugs in returns, indicators, or statistics never silently corrupt results.

## Tasks

| Task | Description |
|------|-------------|
| `repo_setup` | Project structure, pyproject.toml, CI config, pre-commit hooks |
| `returns_calculations` | Simple, log, and excess return functions with edge-case handling |
| `technical_indicators` | SMA, EMA, RSI, MACD, Bollinger Bands — match reference implementations |
| `cross_validation` | Walk-forward and purged k-fold CV utilities for time-series data |
| `activity_logger` | Structured JSON logger for all pipeline activity and experiment tracking |
| `sign_convention_tests` | Explicit tests that long=positive, short=negative throughout codebase |

## Completion Criteria
- All calculation functions have unit tests comparing to known textbook values
- Sign convention tests pass for every returns/PnL function
- CI pipeline runs full test suite on every push
- Logger produces structured output consumable by later phases
- Zero external data dependencies — all tests use synthetic fixtures

## Human Gate
**Type: Review**
Human reviews test coverage report and spot-checks a sample of textbook comparisons
before proceeding to Phase 1.
