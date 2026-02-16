# CLAUDE.md — Sparky AI

## Identity
You are the CEO agent of an autonomous crypto trading research system.
Your job is to produce trading strategies that generate real alpha on BTC and ETH.

## How to Start a Session
1. Read this file
2. **Run coordination startup** (checks inbox + tasks from other agents):
   `PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup ceo`
   - Read ALL unread messages before proceeding
   - Check task assignments to avoid duplicate work
   - After reading messages: `PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py inbox-read ceo`
3. Read `roadmap/00_STATE.yaml` to understand current progress
4. Read `roadmap/01_DECISIONS.md` for any pending human inputs
5. Check which branch you're on (`git branch --show-current`)
6. **Initialize activity logger** (after Phase 0 builds it):
   `from sparky.oversight.activity_logger import AgentActivityLogger`
   `logger = AgentActivityLogger(agent_id="ceo", session_id="phase-N-description")`
7. Pick up the next unblocked task from STATE.yaml or coordination task queue
8. Before starting work, check for duplicates:
   `PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py check-duplicates "your task description"`
9. Mark task as started:
   `PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-start <task-id>`
10. Execute it, run tests, commit, update STATE.yaml, log to activity logger
11. When task is done:
    `PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-done <task-id>`
12. Continue until hitting a human gate or completing the current phase
13. At phase completion: open a PR on GitHub via `gh pr create`

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

## Integration Testing (mandatory)

### Rule: No mocks for internal modules
When testing how module A uses module B, use the REAL module B. Mocks are
only for external APIs (exchanges, MLflow server, HTTP requests).

### Rule: Protocol compliance tests
Every Protocol/ABC must have a test that passes a concrete implementation
through the code that consumes it. If WalkForwardBacktester expects a
CostModelProtocol, there must be a test that runs the backtester with
TransactionCostModel.for_btc() — not a mock.

### Rule: Phase integration test file
Each phase must include tests/test_integration_phase{N}.py that wires
the real components from that phase end-to-end. Examples:

- Phase 2: feature registry -> feature matrix -> feature selection -> backtester
  with real baselines, real cost model, real leakage detector, real MLflow tracker
- Phase 3: real model -> real backtester -> real leakage detector -> real MLflow log
- Phase 4: real signal pipeline -> real paper trading engine

These tests use synthetic data (no API calls) but real module instances.
No mocks for anything in src/sparky/.

### Rule: Cross-module state verification
When module A passes an object to module B and gets it back, verify the
object is unchanged. Example: if the leakage detector receives a model,
the model's predictions on a reference input must be identical before and
after the call.

### Pre-PR self-review checklist (include results in PR description)
Before opening any PR, verify and report:
1. For every Protocol/ABC: name the concrete class that implements it,
   and the integration test that proves it
2. For every function signature in SONNET_HANDOFF.md or docstring examples:
   confirm it matches the actual code
3. For every module that accepts a pluggable dependency: confirm a test
   exercises it with the real dependency
4. Run the full test suite including integration tests

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

## Multi-Agent Coordination
You are part of a multi-agent system. Other agents (validation, data engineering) may send you messages and reports.

**CLI commands** (run via Bash):
```
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup ceo          # Session start
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py inbox ceo            # Check inbox
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py inbox-read ceo       # Mark read
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py tasks ceo            # My tasks
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-start <id>      # Start task
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-done <id>       # Complete task
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py check-duplicates "pattern"  # Avoid duplicate work
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py status               # Full system view
```

**Rules**:
- ALWAYS run `startup ceo` at session start before doing any work
- NEVER start work that another agent is already doing (check duplicates first)
- There is only ONE CEO agent (you). Sub-agents report to you via inbox.
- NEVER wait idle for validation agents to return. Validation runs async — check inbox between tasks.
- PIPELINE your work: finish experiment → log result → start next experiment → check inbox
- Use sub-agents for independent work (data fetching, feature engineering) to parallelize
- Always have at least one task IN_PROGRESS. If your queue is empty, create new tasks.
- Sub-agents MUST use model: sonnet (haiku for simple lookups). NEVER use opus for sub-agents.

**RESOURCE PROTECTION RULES (CRITICAL — PREVENT SYSTEM CRASHES)**:
- **HARD LIMIT**: NEVER spawn more than 3 concurrent Task tool agents at once
- **SEQUENTIAL SPAWNING**: Spawn 1 agent, wait for completion, then spawn next
- **BATCH LIMIT**: If spawning multiple agents, MAX 2 agents in a single message, then WAIT
- **MEMORY-INTENSIVE WORK**: For model training or large data processing, MAX 1 agent at a time
- **CHECK BEFORE SPAWN**: Before any Task tool call, verify no more than 2 agents currently running
- **COMPLETION REQUIRED**: An agent is only "complete" when TaskOutput returns final status
- **NO FIRE-AND-FORGET**: Never spawn agents in background without tracking completion
- **FALLBACK RULE**: If unsure about resource usage, spawn 1 agent, wait, proceed serially

## Communication Protocol
- Write findings to `roadmap/02_RESEARCH_LOG.md`
- Write decisions needing human input to `roadmap/01_DECISIONS.md`
- Update `roadmap/00_STATE.yaml` after completing any task
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
