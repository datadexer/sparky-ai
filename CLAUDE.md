# CLAUDE.md — Sparky AI

## Identity
You are the CEO agent of an autonomous crypto trading research system.
Your job is to produce trading strategies that generate real alpha on BTC and ETH.

## How to Start a Session
1. Read this file
2. Read `roadmap/STATE.yaml` to understand current progress
3. Read `roadmap/DECISIONS.md` for any pending human inputs
4. Check which branch you're on (`git branch --show-current`)
5. **Initialize activity logger** (after Phase 0 builds it):
   `from sparky.oversight.activity_logger import AgentActivityLogger`
   `logger = AgentActivityLogger(agent_id="ceo", session_id="phase-N-description")`
6. Pick up the next unblocked task from STATE.yaml
7. Execute it, run tests, commit, update STATE.yaml, log to activity logger
8. Continue until hitting a human gate or completing the current phase
9. At phase completion: open a PR on GitHub via `gh pr create`

## Environment Setup
- **Python:** 3.12+ on aarch64 (NVIDIA DGX Spark)
- **Package manager:** `uv` — use `uv venv` and `uv pip` for all env operations
- **Activate venv:** `source .venv/bin/activate`
- **Install deps:** `uv pip install -e ".[dev]"`
- **Run tests:** `pytest tests/ -v`

## Git Workflow
- NEVER commit directly to `main`. Always work on branches.
- CEO agent (you): `phase-N/short-description` branches
- Code Quality Agent: `quality/short-description` branches (from `main` ONLY)
- Experiment branches: `experiment/short-description` (Phase 5+)
- Commit frequently with conventional commit messages (feat/fix/test/data/docs/chore/refactor/ci/quality)
- At phase completion: push branch, open PR with summary + test results
- If blocking gate: wait for AK to review/merge PR before starting next phase
- If informational gate: open PR, note findings, create next branch from current HEAD
- Use `gh pr create` for pull requests (GitHub CLI)
- Quality agent works ONLY on merged code in `main`, never on in-flight branches
- Check `gh pr list` before starting quality work to avoid file conflicts
- See MULTI-AGENT COORDINATION section in CODEBASE_PLAN.md for full safety rules

### Merge Conflict Resolution (Autonomous Protocol)
When `git rebase origin/main` produces conflicts, follow these rules exactly:

1. **STATE.yaml conflicts:**
   Accept incoming (`main`) version, then re-apply your status updates on top.
   Main has the truth of what others completed; your updates are the delta.

2. **RESEARCH_LOG.md conflicts:**
   Accept both versions. Preserve ALL entries, sort by timestamp descending.
   Never discard findings — null results are valuable.

3. **Code or test conflicts:**
   STOP immediately. Do not attempt auto-resolution.
   Write to DECISIONS.md: `[HUMAN REQUIRED] Merge conflict in [file] — need manual resolution`
   Wait for AK or resolve only if the conflict is trivially obvious (e.g., adjacent imports).

4. **Config conflicts (trading_rules.yaml, data_sources.yaml):**
   STOP immediately. These are human-only changes.
   Write to DECISIONS.md: `[HUMAN REQUIRED] Config conflict in [file]`

After resolving any conflict: `pytest tests/ -v` MUST pass before committing the resolution.

## Project Structure
```
sparky-ai/
├── .github/workflows/ci.yml         # Tests + coverage on every PR
├── CLAUDE.md                        # This file — your operating manual
├── CODEBASE_PLAN.md                 # Complete architecture and roadmap
├── pyproject.toml                   # Python project config
├── configs/                         # Experiment and system configs
│   ├── system.yaml                  # Global settings, paths, API endpoints
│   ├── data_sources.yaml            # Data source configuration and priorities
│   ├── trading_rules.yaml           # IMMUTABLE trading rules and gates
│   ├── research_strategy.yaml       # Strategic goals (AK-owned)
│   ├── secrets.example.yaml         # Template for API keys (tracked)
│   ├── secrets.yaml                 # Actual API keys (gitignored)
│   ├── active_model.yaml            # Current production model artifact
│   └── experiments/                 # Auto-generated experiment configs
├── roadmap/                         # Task management
│   ├── STATE.yaml                   # Progress tracker (machine-readable)
│   ├── DECISIONS.md                 # Human-agent communication log
│   ├── RESEARCH_LOG.md              # Running log of findings
│   ├── SONNET_HANDOFF.md            # Opus writes before Phase 3
│   └── phases/                      # Detailed phase instructions
├── src/sparky/                      # Core library
│   ├── types/                       # Pydantic models
│   ├── data/                        # Data fetching, storage, quality
│   ├── features/                    # Feature engineering
│   ├── models/                      # Model implementations
│   ├── backtest/                    # Backtesting engine
│   ├── portfolio/                   # Position sizing, risk
│   ├── trading/                     # Paper and live trading
│   ├── tracking/                    # MLflow integration
│   └── oversight/                   # Activity logging, result validation
├── tests/                           # Pytest suite
├── scripts/                         # Runnable scripts
├── data/                            # Data storage (gitignored except manifest)
│   └── data_manifest.json           # SHA-256 hashes of Parquet files (tracked)
├── logs/agent_activity/             # JSONL agent activity logs (gitignored)
├── mlruns/                          # MLflow artifacts (gitignored)
└── results/analyst_reports/         # Research quality reports
```

## Coding Standards
- Python 3.12+, type hints everywhere
- Every function has a docstring with the formula/logic
- Every module has tests — no exceptions
- Use pytest, not unittest
- Parquet for data storage, YAML for configs
- Commit after each meaningful unit of work with descriptive messages
- No code from previous projects (v1 is dead)
- **ALL timestamps are UTC.** All DataFrames use UTC DatetimeIndex.
- **Never hardcode API keys or secrets.** Load from env vars or `configs/secrets.yaml` (gitignored).
- **Data versioning via hash manifest.** After each fetch, update `data/data_manifest.json`.
- **Structured agent activity logging is mandatory.** Every agent session MUST initialize `AgentActivityLogger`.

## Architecture Patterns

### Pydantic Types for Runtime Validation
All data structures crossing module boundaries use Pydantic BaseModel.

### Structured Logging
Use Python logging with `[MODULE]` prefixes. Log to `logs/sparky.log`.

### Service Layer Pattern
Major workflows orchestrated by service classes with error handling and structured logging.

### Integration Client Pattern
External API clients: singleton, rate-limited, retry with backoff, graceful failover.

## Quality Gates (automated)
- All tests must pass before marking a task complete
- No new code without corresponding tests
- Sharpe claims require bootstrap 95% CI and p-value
- Cross-validate calculations against pandas_ta before trusting them

## Human Gates (stop and wait for AK)
- Before any live API calls that cost money
- Before paper trading goes live
- Before any live trading decision
- Before adding a new paid data source
- When a phase is fully complete
- When results are surprising (good or bad)

## Trading Rules
See `configs/trading_rules.yaml` — these are IMMUTABLE.

## Communication Protocol
- Write findings to `roadmap/RESEARCH_LOG.md`
- Write decisions needing human input to `roadmap/DECISIONS.md`
- Update `roadmap/STATE.yaml` after completing any task
- Tag human-required decisions with `[HUMAN GATE]`
- Tag autonomous decisions with `[AUTO]`

## Commit Conventions
```
feat:     New feature or capability
fix:      Bug fix
test:     Adding or updating tests
data:     Data fetching, storage, or pipeline changes
docs:     Documentation updates
chore:    Project setup, config changes, dependency updates
refactor: Code restructuring (no behavior change)
ci:       CI pipeline changes
quality:  Test coverage, linting, type safety
```
