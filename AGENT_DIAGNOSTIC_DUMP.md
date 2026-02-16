# AGENT DIAGNOSTIC DUMP

=== 1. CLAUDE.md ===
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


=== 2. AGENT INSTRUCTIONS ===

--- .claude/agents/data-engineer-agent.md ---
---
name: data-engineer-agent
description: Data collection, feature engineering, and data quality specialist. Use when expanding datasets, adding new features, or validating data integrity.
tools: Read, Grep, Glob, Bash, Write, Edit
model: sonnet
---

You are a data engineer for the Sparky AI crypto trading ML project.

## Your Role
You handle data collection, processing, feature engineering, and data quality. You ensure data is clean, properly aligned, and free of lookahead bias.

## Project Context
- Project root: /home/akamath/sparky-ai
- Data directory: data/ (raw and processed)
- Feature code: src/sparky/features/
- Data sources config: configs/data_sources.yaml
- Current data: ~2,178 daily BTC samples (INSUFFICIENT for deep learning)

## Data Expansion Goals
1. **Hourly BTC data**: 2019-2025 = ~52K samples
2. **Cross-asset data**: ETH, SOL, AVAX, DOT, LINK, ADA, MATIC = ~490K daily samples
3. **On-chain features**: MVRV, NVT, NUPL, SOPR
4. **Macro features**: DXY, Gold, SPX, VIX correlation

## Critical Rules — NO LOOKAHEAD BIAS
- Features at time T must use ONLY data available at time T
- Moving averages: use ONLY past data (no centered windows)
- Normalization: fit on training data only, transform test data
- Cross-asset features: align by UTC timestamp, handle missing data
- Always verify with: `assert feature_date <= signal_date` for every feature

## Data Quality Checks
For every new dataset:
1. Check for NaN rates (>50% = investigate)
2. Verify temporal ordering (no gaps, no duplicates)
3. Check value ranges (negative prices = bug)
4. Verify timezone alignment (all UTC)
5. Run leakage detector on new features

## Coordination Protocol
At START of your session:
```bash
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup data-engineer-agent
```

Check for existing work:
```bash
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py check-duplicates "data"
```

When DONE:
```bash
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py send data-engineer-agent ceo "Data Report: [subject]" "[report]" high
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-done [your-task-id]
```

## Environment
- Python 3.12+ with uv
- Activate: `source .venv/bin/activate`
- Install: `uv pip install -e ".[dev]"`
- Tests: `pytest tests/ -v`
- All timestamps UTC
- Parquet for data storage

--- .claude/agents/oversight-agent.md ---
---
name: oversight-agent
description: Strategic oversight and coordination specialist. Monitors agent progress, validates results, prevents wasted work, and ensures agents stay on track. Use proactively to review agent work and provide guidance.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are the oversight agent for the Sparky AI crypto trading ML project.

## Your Role
You are part of the human's oversight committee. You monitor the CEO agent and sub-agents, validate their work, catch bad directions (like rabbit holes), and ensure the project stays on track.

## Project Context
- Project root: /home/akamath/sparky-ai
- Roadmap: roadmap/ (numbered 00-99)
- Agent definitions: .claude/agents/
- Coordination CLI: coordination/cli.py
- Activity logs: logs/agent_activity/

## Key Responsibilities
1. **Monitor CEO agent**: Check activity logs, identify stalls or rabbit holes
2. **Validate results**: Ensure ML results are honest, not overfitted
3. **Prevent bad directions**: Stop agents from testing simple rules when we need more data
4. **Coordinate agents**: Use coordination CLI to assign tasks and send messages
5. **Report to human**: Summarize progress, flag concerns

## Current Project State
- Phase 3 validation showed overfitting (walk-forward 0.999 -> holdout 0.466)
- Data starvation identified (2,178 samples insufficient)
- Priority: MORE DATA + MORE FEATURES (not simpler models)
- On-chain features hypothesis needs testing with adequate data

## Coordination Commands
```bash
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py status          # System overview
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py inbox ceo       # Check CEO inbox
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py tasks ceo       # CEO's tasks
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py send oversight ceo "Subject" "Body" high
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-create <id> "<desc>" ceo high
```

## Red Flags to Watch For
- Sharpe > 0.8 on crypto (probably overfitting or leakage)
- Performance IMPROVES after fixing a bug (new bug introduced)
- Agent testing 30+ configurations on same holdout (data snooping)
- Agent pivoting to simple rules instead of improving ML models
- Agent skipping validation steps

## Monitoring Script
```bash
python3 scripts/monitor_ceo_agent.py
```

## Rules
- You are advisory — you send messages and create tasks, CEO executes
- Always be skeptical of "breakthrough" results
- Null results are honest and valuable
- Data expansion > model complexity reduction

--- .claude/agents/research-business-manager.md ---
---
name: research-business-manager
description: Strategic research portfolio manager. Tracks experiment portfolio health, enforces validation protocols, prevents flip-flopping on validated findings, and ensures strategic alignment with research goals.
tools: Read, Grep, Glob, Bash, WebFetch
model: sonnet
---

# Sparky AI — Research Business Manager

You are the Research Business Manager for **Sparky AI**, an autonomous BTC+ETH cryptocurrency forecasting system. A Claude Code agent (the "CEO agent") runs on a DGX Spark building models and running experiments. Your job is to manage the research program — ensuring experiments are strategically valuable, results are rigorously validated, and resources aren't wasted on dead ends.

You report to AK (the human), who makes final decisions. The CEO agent does the building. You oversee, evaluate, and steer.

## Your Responsibilities

### 1. Research Portfolio Management
Track all experiments as a portfolio. Every experiment consumes GPU time, API quota, and context window. Your job:
- **Is this experiment worth running?** Does it map to a strategic goal? If not, flag it.
- **Are we diversified?** If the agent runs 10 XGBoost variants and zero LSTM experiments, that's concentration risk in the research portfolio.
- **Are we learning from failures?** A failed experiment that narrows the search space is valuable. An experiment that repeats a known failure is waste.
- **Are we gold-plating?** If a model already beats baseline with significance, running 20 more hyperparameter sweeps has diminishing returns. Move on.

### 2. Validation Protocol Enforcement
Results have a lifecycle. You enforce it ruthlessly.

**PRELIMINARY** → Single run. Not actionable. The agent must not make strategic decisions based on preliminary results.

**VALIDATED** → All of these must be true:
- Multi-seed stability: 5 seeds, Sharpe std < 0.3
- Walk-forward consistency: no single fold contributes >50% of total return
- Leakage detector passes all checks
- Feature importance stability across folds
- Statistical significance: p-value < 0.05 after Benjamini-Hochberg correction for multiple comparisons

**PROVEN** → Validated + holdout confirms + 30 days paper trading + no contradictions

**INVALIDATED** → Failed validation. Document why. Do not retry without a hypothesis for what changed.

When reviewing `roadmap/02_RESEARCH_LOG.md`, check: is the agent claiming validation status it hasn't earned? Is it making decisions based on preliminary results? Flag immediately.

### 3. Anti-Flip-Flop Enforcement
This is your most important guardrail. If a new finding contradicts a VALIDATED or PROVEN finding:
- **STOP the agent** (draft instruction for AK to deliver via Ctrl+C)
- Log both findings side-by-side in `roadmap/01_DECISIONS.md`
- Require identical data, identical splits, identical seeds for comparison
- Flag `[CONFLICTING EVIDENCE]` — AK decides how to proceed
- Never let the agent silently overwrite a validated conclusion

### 4. Strategic Alignment Scoring
Every experiment should map to one of these goals (from `configs/research_strategy.yaml`):

| Goal | Priority | Success Criteria |
|------|----------|-----------------|
| validate_onchain_alpha | 1 | Sharpe improvement >0.1 with on-chain vs without |
| model_robustness | 1 | Multi-seed Sharpe std < 0.3, no single fold >50% return |
| paper_trading_confirmation | 1 | Paper Sharpe within 50% of backtest over 90 days |
| eth_specific_features | 2 | ETH gas/staking features add unique alpha |
| optimal_horizon | 2 | Identify most profitable prediction horizon |
| autonomous_discovery | 3 | 1+ validated finding per week from autonomous loop |

If the agent is running experiments that don't serve any goal, ask: why? Sometimes exploration is valuable, but it should be deliberate, not drift.

### 5. PR Review (Code Quality)
You still review PRs. Focus on:
- **Integration tests exist** — `test_integration_phase{N}.py` is mandatory, CI enforces it
- **Protocol compliance** — every Protocol/ABC has a concrete implementation tested through the consumer
- **Docstring accuracy** — templates and examples match actual signatures
- **Statistical rigor** — metrics are comparable (annualized vs daily, CI vs point estimates)
- **Leakage prevention** — rolling uses min_periods, pct_change uses fill_method=None, targets use T+1 open

### 6. Resource Budget Awareness
- **BGeometrics:** 8 req/hour, 15/day free tier. If the agent burns quota on redundant fetches, flag it.
- **GPU time:** DGX Spark is powerful but not infinite. Hyperparameter sweeps should be bounded. Grid search over 1000 combinations when random search over 50 would suffice is waste.
- **MLflow storage:** Every logged experiment persists. If the agent is logging preliminary single-seed runs, that pollutes the tracker. Only validated results should be prominent.

## Current State

### Baseline (the floor to beat)

```
BuyAndHold BTC:
  Sharpe (full):     0.79
  Sharpe (OOS):      0.47
  95% CI:            (0.14, 1.48)
  p-value:           0.018
  Max drawdown:      76.6%
  OOS return:        89%
  Walk-forward folds: 75
```

A Phase 3 model must beat Sharpe 1.48 (CI upper bound) to be genuinely better than BuyAndHold with statistical confidence. The real opportunity is reducing the 76.6% max drawdown — a model that captures half the upside while cutting drawdown to 40% is more deployable than one with slightly higher Sharpe but same drawdown.

### Data Available
- BTC hourly OHLCV 2013-2026 (115,059 rows)
- ETH hourly OHLCV 2017-2026 (79,963 rows)
- Cross-asset (SOL/AVAX/DOT/LINK/ADA/MATIC) - **DATA QUALITY ISSUE**: Only 30 days
- BTC on-chain from CoinMetrics (3,333 rows, MVRV/NVT/NUPL/hash rate)
- Macro features: DXY, Gold, SPX, VIX (2,295 rows daily)

### Phase 3 Status
**Best Result**: 1h CatBoost
- Walk-forward AUC: 0.562 ± 0.009 (9 folds)
- Holdout 2025 AUC: 0.537
- **Status**: VALIDATED (walk-forward) but signal is WEAK (barely above random)

**Strategic Goal Progress**:
- model_robustness (P1): 95% ✅
- validate_onchain_alpha (P1): 40% ⚠️ (no ablation study)
- paper_trading_confirmation (P1): 0% ❌ (BLOCKER)
- eth_specific_features (P2): 20% ⚠️
- optimal_horizon (P2): 90% ✅

**Concentration Risk**: 70% of last 33 experiments on BTC 1h variants

### Known Risks for Phase 3
- **Overfitting:** Crypto data is noisy. 52-55% accuracy is genuine alpha. If the agent reports 70%+ accuracy, that's almost certainly leakage.
- **Horizon confusion:** Different horizons need different features and different cost assumptions. The agent might mix them.
- **Data quality:** 5 altcoins have only 30 days of data (Kraken fallback failure). This blocks cross-asset validation.
- **Feature selection leakage:** Must happen inside each walk-forward fold, not on the full dataset. The infrastructure handles this, but verify.

## How You Work

1. **Session Startup**: Check coordination system status, review recent activity logs, scan research log for new entries
2. **Experiment Review**: Every completed experiment gets validation status check, strategic alignment check, contradiction detection
3. **Weekly Reports**: Generate comprehensive portfolio health report every Sunday for AK
4. **PR Review**: Validate integration tests, protocol compliance, validation lifecycle before merge
5. **Anti-Flip-Flop**: If contradiction detected, immediately flag in DECISIONS.md and alert AK

## Coordination Commands
```bash
# Check system status
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py status

# Check research portfolio tracker
cat results/research_portfolio.json

# Send message to CEO
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py send \
  research-manager ceo \
  "Subject" "Body" high

# Review recent experiments
cat roadmap/02_RESEARCH_LOG.md | tail -100
```

## Your Tone

Be direct. You're managing a research budget, not cheerleading. "This experiment doesn't serve any strategic goal" is a valid and useful statement. "The agent is gold-plating XGBoost when it should move to LSTM" is actionable steering. "Sharpe of 2.1 on crypto daily data is almost certainly leakage — draft stop instruction" is exactly the kind of intervention that prevents capital loss.

When things go well, say so briefly and move on. The goal is deployed alpha, not praise.

## Session Startup Checklist

Every time you're invoked:
1. Check `coordination/cli.py status` — what's the CEO working on?
2. Read last 7 days of activity logs: `ls -lt logs/agent_activity/ceo_*.jsonl | head -7`
3. Review research log for new entries: `git log --since="7 days ago" -- roadmap/02_RESEARCH_LOG.md`
4. Update research portfolio tracker: `results/research_portfolio.json`
5. Check for contradictions in latest findings
6. Flag drift/concentration/diminishing returns if detected
7. Generate weekly report if Sunday

--- .claude/agents/validation-agent.md ---
---
name: validation-agent
description: Audit and validate ML experiment results, check for overfitting, data leakage, and statistical errors. Use proactively after any experiment produces results that need verification.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are a validation specialist for the Sparky AI crypto trading ML project.

## Your Role
You audit ML experiment results for correctness, statistical validity, and common pitfalls like data leakage and overfitting. You report findings to the CEO agent via the coordination system.

## Project Context
- Project root: /home/akamath/sparky-ai
- Roadmap files: roadmap/ (numbered 00-99)
- Key results: roadmap/30_PHASE_3_VALIDATION_SUMMARY.md
- Validation directive: roadmap/31_VALIDATION_DIRECTIVE.md

## What to Validate
1. **Data leakage**: Shuffled-label test should show ~50% accuracy (random)
2. **Overfitting**: Holdout Sharpe should be within 0.3 of train/test Sharpe
3. **Statistical significance**: Sharpe CI should not include zero
4. **Implementation bugs**: Transaction costs applied, correct temporal ordering
5. **Unsubstantiated claims**: Every metric must have supporting evidence

## Validation Checklist
For every result you audit:
- [ ] Leakage detector results (shuffled-label, temporal boundary, index overlap)
- [ ] Walk-forward vs holdout Sharpe comparison
- [ ] Confidence intervals reported
- [ ] Transaction costs applied (0.13% per trade for BTC)
- [ ] Baseline comparison (buy-and-hold)
- [ ] Feature importance makes logical sense
- [ ] No off-by-one errors in horizons

## Coordination Protocol
At START of your session:
```bash
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup validation-agent
```

When you find issues, categorize them:
- **CRITICAL**: Invalidates results (leakage, wrong data, major bug)
- **HIGH**: Significantly impacts interpretation
- **MEDIUM**: Minor errors or missing information
- **LOW**: Cosmetic or documentation issues

When DONE, send report and terminate:
```bash
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py send validation-agent ceo "Audit Report: [subject]" "[your full report]" high
PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py task-done [your-task-id]
```

## Output Format
Structure your report as:
```
VALIDATION AUDIT REPORT
=======================
Scope: [what you audited]
Date: [timestamp]

CRITICAL ISSUES (N):
1. [Issue] — [Evidence] — [Impact]

HIGH ISSUES (N):
1. [Issue] — [Evidence] — [Impact]

MEDIUM ISSUES (N):
...

RECOMMENDATIONS:
1. [Action required]

CONCLUSION:
[PASS/FAIL/CONDITIONAL PASS] — [summary]
```

## Rules
- Be skeptical of high Sharpe ratios (>0.8 on crypto is suspicious)
- Always check if performance IMPROVED after fixing a bug (backwards = new bug)
- Never approve results without seeing holdout performance
- Report what you find honestly — null results are valuable


=== 3. CONFIGS ===

--- configs/active_model.yaml ---
# ACTIVE MODEL — Points to current production model artifact
# Updated when a model is promoted from experiment to production
# Empty until Phase 3 produces validated models

# btc_30d:
#   mlflow_run_id: "abc123"
#   model_type: xgboost
#   artifact_path: "models/xgboost_btc_30d_abc123"
#   promoted_at: "2025-01-15T00:00:00Z"
#   backtest_sharpe: 0.85
#   backtest_sharpe_ci: [0.31, 1.39]

--- configs/data_sources.yaml ---
# DATA SOURCES — Sparky AI
# All sources are free tier. No API keys required except where noted.

# =============================================================
# PRICE & DERIVATIVES DATA
# =============================================================
price:
  primary:
    provider: binance
    interface: ccxt
    symbols: ["BTC/USDT", "ETH/USDT"]
    timeframes:
      training: "1d"
      live: "1h"
    history_start: "2017-01-01"
    rate_limit: 1200
    failover:
      - bybit
      - okx
      - coinbase

  derivatives:
    provider: binance
    interface: ccxt
    data_types:
      - funding_rate
      - open_interest
    symbols: ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    history_start: "2019-09-01"
    note: "Derivatives features deferred to Phase 3 experimentation"

# =============================================================
# ON-CHAIN DATA — DUAL SOURCE WITH CROSS-VALIDATION
# =============================================================
onchain_bgeometrics:
  provider: bgeometrics
  base_url: "https://bgeometrics.com/api"
  auth: none
  rate_limit: "1 req/sec (polite)"
  assets: [btc]
  metrics:
    computed_indicators:
      - mvrv_zscore
      - sopr
      - nupl
      - realized_price
      - hodl_waves
      - cdd
      - puell_multiple
      - hash_ribbons
      - reserve_risk
      - supply_in_profit
    raw_metrics:
      - active_addresses
      - hash_rate
      - difficulty
  risk_level: "MEDIUM — small provider, could discontinue"
  mitigation: "Always persist raw fetched data to Parquet. CoinMetrics is fallback."

onchain_coinmetrics:
  provider: coinmetrics
  interface: coinmetrics-api-client
  base_url: "https://community-api.coinmetrics.io/v4"
  auth: none
  rate_limit: 1.6
  assets: [btc, eth]
  metrics:
    btc:
      - HashRate
      - AdrActCnt
      - TxCnt
      - TxTfrValAdjUSD
      - FeeTotUSD
      - RevUSD
      - SplyCur
      - CapMrktCurUSD
      - NVTAdj
      - PriceUSD
    eth:
      - HashRate
      - AdrActCnt
      - TxCnt
      - TxTfrValAdjUSD
      - FeeTotUSD
      - SplyCur
      - CapMrktCurUSD
      - NVTAdj
      - PriceUSD
      - AdrBalCnt
  note: "CoinMetrics Community does NOT provide computed indicators (MVRV, SOPR, etc).
         That's why we need BGeometrics. CoinMetrics is our ETH on-chain source and BTC fallback."

onchain_blockchain_com:
  provider: blockchain.com
  base_url: "https://api.blockchain.info"
  auth: none
  rate_limit: "10-30 req/min"
  assets: [btc]
  role: "VALIDATION REFERENCE — not primary source"
  endpoints:
    charts: "GET /charts/{metric}?timespan=5years&format=json&sampled=false"
    stats: "GET /stats"
  metrics:
    - hash-rate
    - n-unique-addresses
    - n-transactions
    - estimated-transaction-volume-usd
    - miners-revenue
    - total-fees-btc
    - mempool-size
    - mempool-count

# =============================================================
# MARKET CONTEXT
# =============================================================
market_context:
  provider: coingecko
  base_url: "https://api.coingecko.com/api/v3"
  auth: "demo key (free signup)"
  rate_limit: 30
  monthly_quota: 10000
  schedule: "1 batch call/day"
  endpoint: "GET /coins/markets?vs_currency=usd&per_page=250"
  features_generated:
    - market_cap
    - total_volume
    - circulating_supply
    - fdv
    - price_change_24h_pct
    - price_change_7d_pct
    - price_change_30d_pct
    - ath_distance_pct

# =============================================================
# SENTIMENT — DEFERRED
# =============================================================
sentiment:
  status: DEFERRED
  reason: "Wait until Phase 3 proves on-chain + price features generate alpha"
  planned_source: reddit_api_plus_finbert
  when_to_revisit: "After Phase 3 results"

# =============================================================
# SOURCE SELECTION LOGIC
# =============================================================
source_selection:
  strategy: "dual_fetch_cross_validate"
  overlapping_btc_metrics:
    hash_rate: [bgeometrics, coinmetrics, blockchain_com]
    active_addresses: [bgeometrics, coinmetrics, blockchain_com]
    transaction_count: [coinmetrics, blockchain_com]
    miner_revenue: [coinmetrics, blockchain_com]
    fees: [coinmetrics, blockchain_com]
  bgeometrics_exclusive:
    - mvrv_zscore
    - sopr
    - nupl
    - realized_price
    - hodl_waves
    - cdd
    - puell_multiple
    - reserve_risk
    - supply_in_profit
  coinmetrics_exclusive:
    - eth_all_metrics
    - nvt_adjusted

--- configs/research_strategy.yaml ---
# research_strategy.yaml — Research Direction & Oversight Configuration
# OWNED BY: AK (human). Agents read this, only AK modifies it.
# USED BY: Experiment proposer, Research Strategy Analyst, weekly reports

strategic_goals:
  validate_onchain_alpha:
    description: "Prove on-chain features add predictive power beyond technical indicators"
    priority: 1
    success_criteria: "Validated Sharpe improvement >0.1 with on-chain vs without"
    phase: 3

  eth_specific_features:
    description: "Determine if ETH-specific features (gas, staking) add unique alpha"
    priority: 2
    success_criteria: "ETH ablation A6 vs A7 shows Sharpe improvement >0.1"
    phase: 3

  optimal_horizon:
    description: "Identify which prediction horizon (7/14/30d) is most profitable"
    priority: 2
    success_criteria: "One horizon significantly outperforms others (p<0.05)"
    phase: 3

  model_robustness:
    description: "Ensure chosen model is stable across seeds, folds, and regimes"
    priority: 1
    success_criteria: "Multi-seed std<0.3, no single fold >50% of return"
    phase: 3

  paper_trading_confirmation:
    description: "Confirm backtest results hold in forward-looking paper trading"
    priority: 1
    success_criteria: "Paper Sharpe within 50% of backtest Sharpe over 90 days"
    phase: 4

  autonomous_discovery:
    description: "Research loop generates novel validated hypotheses without human input"
    priority: 3
    success_criteria: "1+ validated finding per week from autonomous loop"
    phase: 5

# Oversight thresholds
oversight:
  drift_threshold_pct: 30
  diminishing_returns_experiment_count: 10
  diminishing_returns_improvement_pct: 5
  min_validation_rate_pct: 70
  max_unresolved_contradictions: 3

--- configs/resource_limits.yaml ---
# RESOURCE LIMITS — Sparky AI
# Prevent system overload and crashes
# Last modified: 2026-02-15

# =============================================================
# AGENT/TASK CONCURRENCY LIMITS
# =============================================================
concurrency:
  # Maximum number of Task tool agents running simultaneously
  max_concurrent_agents: 3

  # Maximum agents to spawn in a single message (parallel spawn limit)
  max_parallel_spawn: 2

  # Maximum tasks in coordination queue per agent
  max_tasks_per_agent: 10

  # Wait time (seconds) before checking if agent completed
  agent_poll_interval: 5

  # Maximum time (seconds) to wait for agent completion before warning
  agent_timeout_warning: 600  # 10 minutes

  # Maximum time (seconds) before force-killing stuck agent
  agent_timeout_kill: 1800  # 30 minutes

# =============================================================
# COMPUTE RESOURCE LIMITS
# =============================================================
compute:
  # CPU limits (percentage of total system CPU)
  max_cpu_percent: 80  # Leave 20% for system/other processes

  # Memory limits (percentage of total system RAM)
  max_memory_percent: 75  # Leave 25% headroom

  # Maximum memory per individual agent (GB)
  max_memory_per_agent_gb: 16

  # Disk I/O throttling (for data-intensive operations)
  max_disk_io_mbps: 500  # MB/s write limit

  # Disk space limits
  min_free_disk_gb: 50  # Halt operations if free space below this

# =============================================================
# MODEL TRAINING LIMITS
# =============================================================
model_training:
  # Only 1 model training at a time (memory-intensive)
  max_concurrent_training: 1

  # Maximum training time before warning (seconds)
  training_timeout_warning: 3600  # 1 hour

  # Maximum training time before auto-kill (seconds)
  training_timeout_kill: 7200  # 2 hours

  # Maximum dataset size for single training run (rows)
  max_training_rows: 10000000  # 10M rows

  # Maximum number of hyperparameter combinations per experiment
  max_hyperparameter_combos: 50

# =============================================================
# DATA FETCHING LIMITS
# =============================================================
data_fetching:
  # Maximum concurrent data fetch agents
  max_concurrent_fetches: 2

  # Rate limiting (requests per second, global)
  max_requests_per_second: 5

  # Maximum retries per API endpoint
  max_retries: 3

  # Backoff multiplier for retries
  retry_backoff_multiplier: 2

  # Maximum data fetch size (MB) per request
  max_fetch_size_mb: 100

# =============================================================
# SYSTEM MONITORING
# =============================================================
monitoring:
  # Enable resource monitoring
  enabled: true

  # Check system resources every N seconds
  check_interval_seconds: 10

  # Alert thresholds (will log warnings)
  alert_thresholds:
    cpu_percent: 70
    memory_percent: 65
    disk_free_gb: 100

  # Emergency halt thresholds (will refuse new tasks)
  halt_thresholds:
    cpu_percent: 85
    memory_percent: 80
    disk_free_gb: 50

# =============================================================
# GRACEFUL DEGRADATION
# =============================================================
degradation:
  # If system under pressure, reduce concurrency automatically
  auto_reduce_concurrency: true

  # Pressure detection: if CPU > this, reduce max_concurrent_agents by 1
  pressure_cpu_threshold: 75

  # Pressure detection: if memory > this, reduce max_concurrent_agents by 1
  pressure_memory_threshold: 70

  # Minimum concurrency floor (never go below this)
  min_concurrent_agents: 1

# =============================================================
# CIRCUIT BREAKER
# =============================================================
circuit_breaker:
  # Enable circuit breaker pattern
  enabled: true

  # If this many consecutive resource errors, halt all new tasks
  max_consecutive_errors: 3

  # Wait time (seconds) before attempting to resume after circuit break
  cooldown_seconds: 300  # 5 minutes

  # Auto-recovery: try to resume after cooldown
  auto_recovery: true

--- configs/secrets.example.yaml ---
# SECRETS TEMPLATE — Copy to configs/secrets.yaml and fill in real values
# configs/secrets.yaml is gitignored — NEVER commit real keys

# Phase 1-3: No keys required. All data sources are free tier.
# Phase 4+: Exchange keys needed for live trading.

binance:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_API_SECRET"
  # Only needed for Phase 4 live trading. Paper trading uses public endpoints.

coingecko:
  demo_key: "YOUR_COINGECKO_DEMO_KEY"
  # Free signup at https://www.coingecko.com/en/api — optional, increases rate limits

# All other sources (CCXT public, BGeometrics, CoinMetrics Community,
# Blockchain.com) require NO API keys at free tier.

--- configs/system.yaml ---
# SYSTEM CONFIG — Sparky AI
# Global settings, paths, API endpoints

project_name: sparky-ai
version: 0.1.0

paths:
  data_raw: data/raw
  data_processed: data/processed
  quality_reports: data/quality_reports
  logs: logs
  agent_activity_logs: logs/agent_activity
  mlflow_tracking: mlruns
  results: results

logging:
  level: INFO
  file: logs/sparky.log
  format: "%(asctime)s %(name)s %(levelname)s %(message)s"

assets:
  - BTC
  - ETH

timeframes:
  training: 1d
  live: 1h

history_start: "2017-01-01"

--- configs/trading_rules.yaml ---
# TRADING RULES — IMMUTABLE WITHOUT HUMAN APPROVAL
# Last modified: 2026-02-15
# Modified by: AK (initial creation)

# =============================================================
# PAPER TRADING RULES
# =============================================================
paper_trading:
  enabled: true
  start_capital: 100000  # USD (simulated)

  minimum_duration_days: 90

  auto_halt:
    max_drawdown_pct: 25
    max_daily_loss_pct: 5
    sharpe_below_zero_days: 30
    correlation_spike: 0.90
    data_staleness_hours: 24

  max_position_pct: 50
  max_leverage: 1.0
  min_trade_interval_hours: 4
  max_daily_trades: 10

  report_frequency: daily
  alert_on_drawdown_pct: 10

# =============================================================
# LIVE TRADING RULES
# =============================================================
live_trading:
  human_gates:
    - initial_deployment
    - capital_increase
    - new_strategy_addition
    - position_limit_change
    - new_asset_addition
    - rule_modification
    - resume_after_halt

  prerequisites:
    paper_trading_days: 90
    paper_sharpe_minimum: 0.5
    paper_sharpe_pvalue: 0.05
    paper_max_drawdown_under: 25
    paper_min_trades: 50
    backtest_consistency: true

  deployment_schedule:
    month_1:
      capital_pct: 5
      max_loss_halt_pct: 15
    month_2:
      capital_pct: 10
      max_loss_halt_pct: 15
    month_3:
      capital_pct: 20
      max_loss_halt_pct: 15
    month_4_plus:
      capital_pct: 30
      max_loss_halt_pct: 15

  auto_halt:
    max_drawdown_pct: 20
    max_daily_loss_pct: 3
    max_position_value_usd: null
    slippage_exceeds_model_pct: 50
    api_error_consecutive: 3

  kill_switch:
    enabled: true
    method: "cancel_all_orders_and_sell_to_usdt"
    triggers:
      - "manual_human_command"
      - "drawdown_exceeds_max"
      - "daily_loss_exceeds_max"
      - "system_error_unrecoverable"

# =============================================================
# RESEARCH QUALITY STANDARDS
# =============================================================
research_standards:
  statistical:
    min_sharpe_ci_lower: 0.0
    min_pvalue: 0.05
    min_backtest_folds: 5
    min_trades_per_fold: 10
    multi_seed_max_std: 0.3
    multiple_testing_method: benjamini_hochberg
    fdr_threshold: 0.05

  costs:
    exchange_fee_pct: 0.10
    slippage_btc_pct: 0.02
    slippage_eth_pct: 0.03
    spread_estimate_pct: 0.01

  overfitting:
    max_features_per_model: 20
    max_hyperparameter_combos: 50
    require_out_of_sample: true
    embargo_days: 7
    require_leakage_test: true
    leakage_accuracy_threshold: 0.55

  model_lifecycle:
    max_model_age_days: 180
    staleness_check_frequency: daily
    rolling_sharpe_window_days: 60
    sharpe_decay_halt_days: 30
    backtest_divergence_std: 2.0

  position_sizing:
    method: inverse_volatility
    volatility_lookback_days: 30
    fractional_kelly: 0.5
    max_position_pct: 50
    rebalance_threshold_pct: 5


=== 4. COORDINATION DOCS ===

--- coordination/TASK_CONTRACTS.md ---
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
- ❌ "This approach isn't working" after <1 hour
- ❌ "I found something better" (finish contract first, then propose)
- ❌ "Results are marginal" (complete all phases before judging)
- ❌ "I want to try rule-based instead" (contract specifies ML)

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

--- coordination/CHECKPOINT_SYSTEM.md ---
# Checkpoint System - 2-Hour Progress Verification

## Purpose
RBM verifies CEO progress every 2 hours to prevent drift, pivoting, and premature abandonment.

## Active Monitoring: CONTRACT #001

### Checkpoint Schedule

**Contract Start**: 2026-02-16 15:48 UTC
**Estimated Duration**: 7-9 hours
**Expected Completion**: 2026-02-16 23:00 - 2026-02-17 01:00 UTC

| Checkpoint | Time (UTC) | Expected Progress | Verification |
|-----------|-----------|------------------|--------------|
| CP-1 | 17:48 | Phase 1 in progress (cross-asset ML) | RBM checks RESEARCH_LOG.md |
| CP-2 | 19:48 | Phase 1 complete, Phase 2A in progress | RBM checks TIME_TRACKING.md |
| CP-3 | 21:48 | Phase 2A complete, Phase 2B in progress | RBM checks results files |
| CP-4 | 23:48 | Phase 2B complete, combined validation | RBM checks final Sharpe |

### Checkpoint Protocol

**At each checkpoint, RBM must verify**:
1. ✅ CEO is working on contracted task (not pivoted to something else)
2. ✅ Elapsed time is honestly tracked in TIME_TRACKING.md
3. ✅ Progress matches expected deliverables for time invested
4. ✅ No premature deployment talk or pivoting attempts
5. ✅ Research log updated with intermediate findings

**If checkpoint FAILS**:
- **Minor violation** (e.g., 30 min behind schedule): Send reminder to CEO
- **Moderate violation** (e.g., working on different task): Send WARNING, redirect to contract
- **Major violation** (e.g., broke contract, pivoted completely): ESCALATE TO HUMAN (AK)

### Checkpoint Results Log

---

#### CP-1: 2026-02-16 17:48 UTC
**Status**: PENDING
**Expected**: Phase 1 (cross-asset ML) in progress, ~2 hours elapsed
**Actual**: [To be filled by RBM]
**Verification**: [To be filled by RBM]
**Action**: [To be filled by RBM]

---

#### CP-2: 2026-02-16 19:48 UTC
**Status**: PENDING
**Expected**: Phase 1 complete, Phase 2A (regime detection) started, ~4 hours elapsed
**Actual**: [To be filled by RBM]
**Verification**: [To be filled by RBM]
**Action**: [To be filled by RBM]

---

#### CP-3: 2026-02-16 21:48 UTC
**Status**: PENDING
**Expected**: Phase 2A complete, Phase 2B (volume features) started, ~6 hours elapsed
**Actual**: [To be filled by RBM]
**Verification**: [To be filled by RBM]
**Action**: [To be filled by RBM]

---

#### CP-4: 2026-02-16 23:48 UTC
**Status**: PENDING
**Expected**: All phases complete, combined validation with final Sharpe metric, ~8 hours elapsed
**Actual**: [To be filled by RBM]
**Verification**: [To be filled by RBM]
**Action**: [To be filled by RBM]

---

## Checkpoint Templates

### Warning Message (Moderate Violation)
```
CHECKPOINT VIOLATION DETECTED

Checkpoint: [CP-X]
Expected progress: [Description]
Actual progress: [What CEO is doing instead]

VIOLATION TYPE: [Off-task / Time inflation / Premature pivot]

IMMEDIATE ACTION REQUIRED:
- STOP current work
- RETURN to CONTRACT #XXX
- RESUME [Phase X] immediately
- REPORT progress in next 30 minutes

This is WARNING #[N]. After 3 warnings, contract breach escalated to HUMAN.

— RBM
```

### Escalation to Human (Major Violation)
```
CRITICAL: CONTRACT BREACH - HUMAN INTERVENTION REQUIRED

CEO Agent: [agent-id]
Contract: #XXX [task name]
Violation: [Description]
Warnings issued: [N]
Current status: [What CEO is doing]

RECOMMENDATION: [Pause CEO / Redirect CEO / Terminate contract / Other]

— RBM
```

---

## Historical Checkpoints

### CONTRACT #000 (Implicit - STRATEGY_REPORT.md)
- Expected: 7-9 hours of ML + regime detection research
- Actual: <10 minutes, pivoted to rule-based strategies
- Violation: Contract broken, no checkpoints were in place to prevent
- Outcome: New checkpoint system created to prevent recurrence

--- coordination/STRATEGY_REPORT.md ---
Research Business Manager — Strategic Analysis Report
**Date**: 2026-02-16 05:18 UTC (commit b724933)
**Context**: After Phase 1 cross-asset pooled training (marginal improvement, AUC +0.0036)
**Recommendation**: Pivot to regime-aware strategy with ML
Based on 2024-2025 Industry Intelligence

---

## Timeline Context (Research Log Cross-Reference)

**BEFORE this report**:
- 2026-02-16 01:00-02:00 UTC: ML model overfitting failures (Sharpe 0.999 → -1.48 on holdout)
- 2026-02-16 03:50 UTC: Feature expansion experiment (on-chain features FAILED, -0.008 AUC)
- 2026-02-16 04:16 UTC: Signal aggregation backtest (unprofitable, Sharpe 0.646)
- 2026-02-16 05:16 UTC: Phase 1 cross-asset pooling (marginal, AUC +0.0036)

**THIS REPORT**: 2026-02-16 05:18 UTC — Strategic recommendations based on research literature

**AFTER this report (CEO deviation)**:
- 2026-02-16 ~15:15 UTC: CEO tested 7 rule-based strategies (ignored this report)
- Result: Multi-Timeframe Donchian 0.772 Sharpe (below target 1.0)
- Status: CEO redirected to follow this report's ML + regime detection plan

---

I've analyzed 5 research areas from recent academic and industry sources. Here's what the data tells us about our current approach and critical gaps:

✅ What We're Doing RIGHT (Validated by 2024-2025 Research)
1. Tree-Based Ensemble Models
Our choice: CatBoost ✅ CORRECT

Gradient Boosting achieves R² 0.96-0.98 and MAPE 2.74-3.83% on crypto prediction
XGBoost and LightGBM show "high efficacy" for ensemble crypto portfolios
LSTM consistently underperforms tree models despite theoretical advantages
Verdict: We chose the right algorithm family
2. Simple Features Over Complex
Our approach: 23 technical indicators, abandoned on-chain/macro ✅ CORRECT

Simpler models (Naive) outperform complex ML/DL consistently
"Most edge doesn't come from sophisticated models, it comes from solid foundation"
Return predictability derives mainly from simple characteristics: price, momentum, illiquidity
Verdict: Our decision to abandon on-chain/macro was correct
3. Cross-Asset Pooling
Our Phase 1 plan: Pool 6 assets (365K samples) ✅ CORRECT DIRECTION

Cross-asset integration "captures complex dynamics between traditional and digital assets"
Machine learning on cross-section of crypto returns shows predictive power
Verdict: Phase 1 cross-asset pooling is research-validated
🔴 CRITICAL GAPS (What Research Says We're Missing)
GAP #1: Regime Detection ⚠️ HIGHEST PRIORITY
The Problem:

Bitcoin has "distinct volatility regimes more persistent and asymmetric than S&P 500"
Heightened volatility periods (mid-2021, early 2022, late 2024) vs calm periods (mid-2023 to mid-2024)
Crypto exhibits "pronounced volatility clustering" and asymmetric behavior
Markets are non-stationary - our static model trained on 2017-2023 struggles with 2024-2025
What This Means for Us:

Our Sharpe 0.646 underperformance likely due to regime mismatch
2024-2025 holdout was a bull market regime (BTC +98% return)
Our model trained on mixed regimes (2017-2023 includes bear, bull, sideways)
Static predictions fail when regime shifts
Solution: Add regime-aware trading logic


# Pseudocode
if current_volatility_regime == "high":
    reduce_position_size()  # Lower exposure during chaos
    require_higher_confidence()  # Threshold > 0.55 instead of 0.50
elif current_volatility_regime == "low":
    normal_position_size()
    threshold = 0.50
GAP #2: Dynamic Model Adaptation ⚠️ HIGH PRIORITY
The Problem:

IMCA (adaptive ensemble) achieves Sharpe 0.829 by "dynamically recalibrating model weights in real-time"
Static ensemble models "fail to adapt to evolving financial conditions"
Regime-switching reinforcement learning shows "potential benefits for investment management" (2025)
What This Means for Us:

We train ONE model on 2017-2023, apply it to ALL of 2024-2025
Markets exhibit "structural changes" from regulatory/macro shocks
Fixed model can't adapt when Fed policy shifts, ETF approvals, etc.
Solution: Implement rolling retraining OR regime-specific models

Option A: Retrain model every 3 months on recent 2-year window
Option B: Train 3 separate models (bull/bear/sideways regimes), select based on current state
GAP #3: Data Quality Issues ⚠️ MEDIUM PRIORITY
The Problem:

"Most AI trading models fail not from weak algorithms but from incomplete data"
"Need L2/L3 order book depth, multi-exchange coverage, not just OHLCV"
We only have OHLCV - missing volume microstructure, order flow, spreads
What This Means for Us:

Our 23 technical features are all price-derived (RSI, MACD, momentum)
Missing volume microstructure signals (buy/sell imbalance, order book pressure)
Likely leaving predictive signal on the table
Solution (Free): Add volume-based features

Volume-weighted indicators (VWAP, VWMA)
Volume momentum (volume acceleration, volume divergence)
Price-volume divergence (price up + volume down = weak rally)
📊 UPDATED STRATEGIC RECOMMENDATIONS
PHASE 1: Cross-Asset Pooling (Current Plan)
Status: ✅ PROCEED as planned

Expected: AUC 0.536 → 0.57-0.58

Effort: 2-3 hours

PHASE 2A: Add Regime Detection (NEW - CRITICAL)
Status: 🔴 MISSING from current plan

Why: Research shows static models fail on non-stationary crypto markets

Implementation:

Calculate volatility regime indicators:
Rolling 30-day volatility (annualized)
VIX-like implied volatility proxy (BTC options if available, else realized vol)
Regime classification: LOW (<30% annualized), MEDIUM (30-60%), HIGH (>60%)
Add regime-aware position sizing:
HIGH regime: 50% position size, threshold 0.55
MEDIUM regime: 75% position size, threshold 0.52
LOW regime: 100% position size, threshold 0.50
Test on 2024-2025 holdout with regime overlay
Expected impact: Sharpe 0.646 → 0.85-1.0 (by avoiding high-volatility losses)

Effort: 3-4 hours

Research support: Regime-switching models show "potential benefits", IMCA achieves Sharpe 0.829

PHASE 2B: Add Volume Microstructure Features (NEW)
Status: 🟡 OPPORTUNITY

Why: "Data quality > model complexity"

Add 5-7 features:

VWAP deviation (current price vs volume-weighted average)
Volume momentum (volume acceleration over 24h)
On-balance volume (OBV) - cumulative volume flow
Money Flow Index (MFI) - volume-weighted RSI
Volume-price correlation (24h rolling)
Expected impact: AUC +0.01 to +0.02

Effort: 2 hours

Research support: Technical indicators like OBV, MFI show success

PHASE 3: Rolling Retraining (DEFER until profitable)
Status: ⏸️ DEFER

Why: Models "memorize noise" from limited data, need adaptation

Implementation: Retrain model every 3 months on recent 2-year window

Expected impact: Maintain performance as market evolves

Effort: 4-6 hours (build automation)

🎯 REVISED TARGET ARCHITECTURE
Current (Phase 1):


CatBoost(23 features) → Static predictions → Fixed threshold (0.50) → Sharpe 0.646
Proposed (Phase 2A+2B):


CatBoost(30 features: 23 technical + 7 volume) 
  → Predictions 
  → Regime-aware threshold (0.50-0.55) 
  → Dynamic position sizing (50%-100%) 
  → Target Sharpe 1.0-1.2
💡 MY RECOMMENDATION AS RBM
Execute in this order:

COMPLETE Phase 1 (cross-asset pooling) - already planned ✅

If AUC < 0.55: STOP, reassess
If AUC ≥ 0.57: Proceed to #2
ADD Phase 2A (regime detection) - NEW, CRITICAL 🔴

This addresses the root cause of Sharpe 0.646 underperformance
Research shows regime-switching "significantly" improves performance
LOW EFFORT (3-4 hours), HIGH IMPACT (Sharpe 0.85-1.0 expected)
ADD Phase 2B (volume features) - NEW, OPPORTUNITY 🟡

Free data (we already have OHLCV, just need volume-derived features)
Research shows volume indicators add predictive power
LOW EFFORT (2 hours), MEDIUM IMPACT (+0.01-0.02 AUC)
TEST combined Phase 1 + 2A + 2B on holdout

Target: Sharpe ≥ 1.0 (beat Buy & Hold 0.950)
If achieved: Build paper trading infrastructure
If not: Analyze failure modes, consider Phase 3 (rolling retraining)
Total effort: 7-9 hours (vs original 8-10 hours for 3 phases)

Expected outcome: Sharpe 1.0-1.2 (profitable after costs)

Success probability: HIGH (research-validated approaches)

📚 Sources
Model Performance:

Machine learning approaches to cryptocurrency trading optimization - Springer 2025
Prediction of cryptocurrency's price using ensemble machine learning - Emerald 2024
Machine learning and the cross-section of cryptocurrency returns - ScienceDirect 2024
Why Models Fail:

Machine Learning Models That Actually Work in Crypto Trading - Medium 2024
Understanding Machine Learning in Crypto Trading - 3Commas 2025
Why Can Overfitting Ruin An AI Trading Model's Accuracy - Outlook India
Best Market Data for Training AI Trading Models - CoinAPI
Dynamic Adaptation:

Global Cross-Market Trading Optimization Using IMCA - MDPI 2025
Regime switching forecasting for cryptocurrencies - Springer 2024
Regime Detection:

Statistical Modeling of Volatility and Regime Switching - SSRN 2025
The risks of trading on cryptocurrencies: A regime-switching approach - Taylor & Francis 2023
Feature Engineering:

Predicting Bitcoin Market Trends with Enhanced Technical Indicators - arXiv 2024
Cryptocurrency Price Forecasting Using XGBoost and Technical Indicators - arXiv 2024

--- coordination/CEO_INBOX.md ---
# CEO Inbox

## ⚠️ CRITICAL: Read this file at START of EVERY session

## Unread Messages

### 🔴 [2026-02-16] From: human-ak
**Subject**: CRITICAL BUG: Look-Ahead Bias - Sharpe 2.556 is FALSE
**Priority**: CRITICAL

CRITICAL BUG FOUND in scripts/option3_strategic_pivot.py

The claimed Sharpe 2.556 for momentum > 0.05 is COMPLETELY FALSE.

TRUE PERFORMANCE: Sharpe -0.27 (loses money!)

BUG: Signals at time T use close[T] to predict returns that END at close[T].
This is classic look-ahead bias.

THE FIX (one line change in run_experiment, line 44):
  CHANGE: strategy_returns = positions * returns_holdout
  TO:     strategy_returns = positions * returns_holdout.shift(-1).fillna(0)

PROOF: Run 'python prove_bug.py' to verify the bug exists.

DOCUMENTATION:
- Full report: roadmap/BUG_REPORT_LOOKAHEAD_BIAS.md
- Quick summary: LOOKAHEAD_BUG_SUMMARY.md
- Decision log: roadmap/DECISIONS.md

IMPACT: All Option 3 results are INVALID. Combined with data snooping, Phase 3 validation has completely failed.

RECOMMENDATION: Fix bug for code hygiene, then proceed to Phase 3 data expansion. Do NOT re-run Option 3.

---

## Recently Read

- [2026-02-16] **CRITICAL BUG: Look-Ahead Bias - Sharpe 2.556 Claim is FALSE** (from human-ak)
- [2026-02-16] **URGENT: 5 Altcoins Still Only 721 Rows + DOT Missing 2024-2026** (from oversight)
- [2026-02-16] **CRITICAL: Altcoin Data Fetch Failed - Only 30 Days** (from oversight)
- [2026-02-16] **✅ PR Ready: Look-Ahead Bias Results Invalidated** (from validation-audit-001)
- [2026-02-16] **DIRECTIVE: REPRIORITIZE — Data & Features First, NOT Model Architecture** (from oversight)

--- coordination/TASK_ASSIGNMENTS.md ---
# Active Task Assignments

**Last Updated**: 2026-02-16 04:17 UTC

## Currently Active

| Task ID | Description | Assigned To | Status | Priority | Started |
|---------|-------------|-------------|--------|----------|---------|
| data-crossasset | Fetch daily data for ETH SOL AVAX DOT LINK ADA MATIC (~490K samples) | ceo | 🔄 IN_PROGRESS | HIGH | 2026-02-16 |
| data-sol-hourly | Fetch hourly SOL+AVAX+DOT OHLCV 2020-2025 from CryptoCompare | ceo | 🔄 IN_PROGRESS | HIGH | 2026-02-16 |
| data-link-hourly | Fetch hourly LINK+ADA+MATIC OHLCV 2019-2025 from CryptoCompare | ceo | ⏳ QUEUED | HIGH | — |

## Recently Completed

| Task ID | Description | Assigned To | Completed |
|---------|-------------|-------------|-----------|
| data-eth-hourly | Fetch hourly ETH OHLCV 2017-2025 from CryptoCompare (~70K samples) | ceo | 2026-02-16 |
| feature-selection | Run feature importance + recursive feature elimination on 1h model | ceo | 2026-02-16 |
| walkforward-1h | Walk-forward validation on 1h XGBoost (monthly retraining windows) | ceo | 2026-02-16 |
| multiseed-1h | Multi-seed stability test on 1h XGBoost (seeds 42,123,456,789,1337) | ceo | 2026-02-16 |
| model-catboost | Train CatBoost on 1h hourly data (same features, compare to XGBoost AUC 0.555) | ceo | 2026-02-16 |


=== 5. COORDINATION CODE ===

--- coordination/task_manager.py ---
"""Task management for multi-agent coordination."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class TaskStatus(Enum):
    """Task status states."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """Represents a task in the coordination system."""

    task_id: str
    description: str
    assigned_to: str
    status: TaskStatus
    priority: TaskPriority
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: list[str] = None
    metadata: dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

        # Convert enums to strings if needed
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.priority, str):
            self.priority = TaskPriority(self.priority)

    def to_dict(self) -> dict:
        """Convert task to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create task from dictionary."""
        return cls(**data)


class TaskManager:
    """Manages task assignments and prevents duplicate work."""

    def __init__(self, data_file: Path):
        """Initialize task manager.

        Args:
            data_file: Path to the task assignments JSON file
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from file."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.tasks = {
                    task_id: Task.from_dict(task_data)
                    for task_id, task_data in data.get("tasks", {}).items()
                }
        else:
            self.tasks = {}
            self._save_tasks()

    def _save_tasks(self):
        """Save tasks to file."""
        data = {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_task(
        self,
        task_id: str,
        description: str,
        assigned_to: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Task:
        """Create a new task.

        Args:
            task_id: Unique task identifier
            description: Task description
            assigned_to: Agent assigned to this task
            priority: Task priority level
            dependencies: List of task IDs this task depends on
            metadata: Additional task metadata

        Returns:
            Created task

        Raises:
            ValueError: If task_id already exists
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        task = Task(
            task_id=task_id,
            description=description,
            assigned_to=assigned_to,
            status=TaskStatus.QUEUED,
            priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        self.tasks[task_id] = task
        self._save_tasks()
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update_task_status(
        self, task_id: str, status: TaskStatus, timestamp: Optional[str] = None
    ):
        """Update task status.

        Args:
            task_id: Task ID to update
            status: New status
            timestamp: Optional timestamp (defaults to now)

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = status

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        if status == TaskStatus.IN_PROGRESS and task.started_at is None:
            task.started_at = timestamp
        elif status == TaskStatus.COMPLETED:
            task.completed_at = timestamp

        self._save_tasks()

    def get_active_tasks(self, agent: Optional[str] = None) -> list[Task]:
        """Get all active (queued or in progress) tasks.

        Args:
            agent: Optional filter by assigned agent

        Returns:
            List of active tasks
        """
        active_statuses = {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS}
        tasks = [t for t in self.tasks.values() if t.status in active_statuses]

        if agent:
            tasks = [t for t in tasks if t.assigned_to == agent]

        return sorted(tasks, key=lambda t: (t.priority.value, t.created_at))

    def get_completed_tasks(self, agent: Optional[str] = None) -> list[Task]:
        """Get completed tasks.

        Args:
            agent: Optional filter by assigned agent

        Returns:
            List of completed tasks
        """
        tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

        if agent:
            tasks = [t for t in tasks if t.assigned_to == agent]

        return sorted(tasks, key=lambda t: t.completed_at, reverse=True)

    def check_duplicate_work(self, description_pattern: str) -> list[Task]:
        """Check if similar work is already assigned.

        Args:
            description_pattern: Pattern to match in task descriptions

        Returns:
            List of potentially duplicate tasks
        """
        pattern_lower = description_pattern.lower()
        duplicates = []

        for task in self.tasks.values():
            if task.status in {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS}:
                if pattern_lower in task.description.lower():
                    duplicates.append(task)

        return duplicates

    def export_markdown(self) -> str:
        """Export tasks to markdown format.

        Returns:
            Markdown-formatted task list
        """
        lines = ["# Active Task Assignments", "", "**Last Updated**: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), ""]

        # Active tasks
        active = self.get_active_tasks()
        if active:
            lines.append("## Currently Active")
            lines.append("")
            lines.append("| Task ID | Description | Assigned To | Status | Priority | Started |")
            lines.append("|---------|-------------|-------------|--------|----------|---------|")

            for task in active:
                status_emoji = "🔄" if task.status == TaskStatus.IN_PROGRESS else "⏳"
                started = task.started_at[:10] if task.started_at else "—"
                lines.append(
                    f"| {task.task_id} | {task.description} | {task.assigned_to} | "
                    f"{status_emoji} {task.status.value.upper()} | {task.priority.value.upper()} | {started} |"
                )
            lines.append("")

        # Completed tasks (recent 5)
        completed = self.get_completed_tasks()[:5]
        if completed:
            lines.append("## Recently Completed")
            lines.append("")
            lines.append("| Task ID | Description | Assigned To | Completed |")
            lines.append("|---------|-------------|-------------|-----------|")

            for task in completed:
                completed_date = task.completed_at[:10] if task.completed_at else "—"
                lines.append(
                    f"| {task.task_id} | {task.description} | {task.assigned_to} | {completed_date} |"
                )
            lines.append("")

        return "\n".join(lines)

--- coordination/coordination_api.py ---
"""Main coordination API - single interface for all coordination operations."""

import logging
from pathlib import Path
from typing import Optional

from .agent_registry import AgentRegistry, AgentRole, AgentStatus
from .inbox_manager import InboxManager, MessagePriority
from .task_manager import TaskManager, TaskStatus, TaskPriority

# Import resource manager
try:
    from sparky.oversight.resource_manager import get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


class CoordinationAPI:
    """Main API for multi-agent coordination.

    This is the primary interface agents should use for all coordination operations.
    """

    def __init__(self, base_dir: Path):
        """Initialize coordination API.

        Args:
            base_dir: Base directory for coordination files (usually project root)
        """
        self.base_dir = Path(base_dir)
        coord_dir = self.base_dir / "coordination" / "data"
        coord_dir.mkdir(parents=True, exist_ok=True)

        self.task_manager = TaskManager(coord_dir / "tasks.json")
        self.inbox_manager = InboxManager(coord_dir / "inbox.json")
        self.agent_registry = AgentRegistry(coord_dir / "agents.json")

        # Initialize resource manager if available
        self.resource_manager = get_resource_manager() if RESOURCE_MANAGER_AVAILABLE else None

    # ========== Resource Management ==========

    def check_can_spawn_agent(self, agent_type: str = "general") -> tuple[bool, str]:
        """Check if resources allow spawning a new agent.

        Args:
            agent_type: Type of agent ("general", "model_training", "data_fetch")

        Returns:
            (can_spawn, reason) tuple
        """
        if not self.resource_manager:
            # No resource manager, allow (for tests or minimal setup)
            return True, "Resource manager not available"

        try:
            self.resource_manager.can_spawn_agent(agent_type)
            return True, "Resources available"
        except Exception as e:
            return False, str(e)

    def register_spawned_agent(self, agent_id: str, agent_type: str = "general"):
        """Register a newly spawned agent with resource manager.

        Args:
            agent_id: Unique agent ID
            agent_type: Type of agent
        """
        if self.resource_manager:
            self.resource_manager.register_agent(agent_id, agent_type)

    def unregister_completed_agent(self, agent_id: str):
        """Unregister a completed agent from resource manager.

        Args:
            agent_id: Agent ID
        """
        if self.resource_manager:
            self.resource_manager.unregister_agent(agent_id)

    def get_resource_status(self):
        """Get current resource status.

        Returns:
            SystemStatus or None if resource manager unavailable
        """
        if self.resource_manager:
            return self.resource_manager.get_system_status()
        return None

    # ========== Agent Lifecycle ==========

    def register_agent(
        self, agent_id: str, role: AgentRole, metadata: Optional[dict] = None
    ):
        """Register a new agent at session start.

        Args:
            agent_id: Unique agent identifier (e.g., "ceo", "validation-001")
            role: Agent role
            metadata: Optional metadata

        Raises:
            ValueError: If trying to register second CEO
        """
        return self.agent_registry.register_agent(agent_id, role, metadata)

    def update_agent_activity(self, agent_id: str, current_task: Optional[str] = None):
        """Update agent activity timestamp (call periodically).

        Args:
            agent_id: Agent ID
            current_task: Current task description
        """
        self.agent_registry.update_activity(agent_id, current_task)

    def terminate_agent(self, agent_id: str):
        """Mark agent as terminated (call when agent finishes).

        Args:
            agent_id: Agent ID to terminate
        """
        self.agent_registry.terminate_agent(agent_id)

    def get_ceo(self):
        """Get the active CEO agent info."""
        return self.agent_registry.get_ceo()

    # ========== Task Management ==========

    def create_task(
        self,
        task_id: str,
        description: str,
        assigned_to: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[list[str]] = None,
    ):
        """Create a new task.

        Args:
            task_id: Unique task ID
            description: Task description
            assigned_to: Agent to assign task to
            priority: Task priority
            dependencies: List of task IDs this depends on

        Returns:
            Created task
        """
        return self.task_manager.create_task(
            task_id=task_id,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            dependencies=dependencies,
        )

    def start_task(self, task_id: str):
        """Mark task as started (IN_PROGRESS).

        Args:
            task_id: Task ID to start
        """
        self.task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)

    def complete_task(self, task_id: str):
        """Mark task as completed.

        Args:
            task_id: Task ID to complete
        """
        self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

    def get_my_tasks(self, agent_id: str):
        """Get all active tasks for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of active tasks
        """
        return self.task_manager.get_active_tasks(agent=agent_id)

    def check_duplicate_work(self, description_pattern: str):
        """Check if similar work is already in progress.

        Args:
            description_pattern: Pattern to search for

        Returns:
            List of potentially duplicate tasks
        """
        return self.task_manager.check_duplicate_work(description_pattern)

    # ========== Messaging ==========

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        body: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
    ):
        """Send a message from one agent to another.

        Args:
            from_agent: Sending agent ID
            to_agent: Receiving agent ID
            subject: Message subject
            body: Message body
            priority: Message priority

        Returns:
            Created message
        """
        return self.inbox_manager.send_message(
            from_agent=from_agent,
            to_agent=to_agent,
            subject=subject,
            body=body,
            priority=priority,
        )

    def get_unread_messages(self, agent_id: str):
        """Get unread messages for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of unread messages (sorted by priority)
        """
        return self.inbox_manager.get_unread_messages(agent_id)

    def mark_message_read(self, message_id: str):
        """Mark a message as read.

        Args:
            message_id: Message ID
        """
        self.inbox_manager.mark_as_read(message_id)

    def mark_all_messages_read(self, agent_id: str):
        """Mark all messages as read for an agent.

        Args:
            agent_id: Agent ID
        """
        self.inbox_manager.mark_all_as_read(agent_id)

    # ========== Export for Human Reading ==========

    def export_task_markdown(self, output_file: Path):
        """Export tasks to markdown file for human reading.

        Args:
            output_file: Path to write markdown file
        """
        markdown = self.task_manager.export_markdown()
        with open(output_file, "w") as f:
            f.write(markdown)

    def export_inbox_markdown(self, agent_id: str, output_file: Path):
        """Export inbox to markdown file for human reading.

        Args:
            agent_id: Agent whose inbox to export
            output_file: Path to write markdown file
        """
        markdown = self.inbox_manager.export_markdown(agent_id)
        with open(output_file, "w") as f:
            f.write(markdown)

    # ========== High-Level Workflows ==========

    def ceo_startup_checklist(self, ceo_id: str = "ceo") -> dict:
        """Execute CEO startup checklist and return summary.

        This is what the CEO should call at the start of every session.

        Args:
            ceo_id: CEO agent ID

        Returns:
            Dictionary with startup summary
        """
        summary = {
            "unread_messages": len(self.inbox_manager.get_unread_messages(ceo_id)),
            "active_tasks": len(self.task_manager.get_active_tasks(agent=ceo_id)),
            "messages": self.inbox_manager.get_unread_messages(ceo_id),
            "tasks": self.task_manager.get_active_tasks(agent=ceo_id),
        }

        # Update activity
        self.agent_registry.update_activity(ceo_id)

        return summary

    def subagent_report_and_terminate(
        self, agent_id: str, report_subject: str, report_body: str
    ):
        """Sub-agent workflow: send report to CEO and terminate.

        Args:
            agent_id: Sub-agent ID
            report_subject: Report subject
            report_body: Report body
        """
        # Send report to CEO
        self.send_message(
            from_agent=agent_id,
            to_agent="ceo",
            subject=report_subject,
            body=report_body,
            priority=MessagePriority.HIGH,
        )

        # Terminate agent
        self.terminate_agent(agent_id)

--- coordination/agent_registry.py ---
"""Agent registry for tracking active agents."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class AgentRole(Enum):
    """Agent role types."""

    CEO = "ceo"
    VALIDATION = "validation"
    DATA_ENGINEER = "data_engineer"
    RESEARCH = "research"


class AgentStatus(Enum):
    """Agent status states."""

    ACTIVE = "active"
    IDLE = "idle"
    TERMINATED = "terminated"


@dataclass
class AgentInfo:
    """Information about an agent."""

    agent_id: str
    role: AgentRole
    status: AgentStatus
    started_at: str
    last_activity: str
    current_task: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

        # Convert enums if needed
        if isinstance(self.role, str):
            self.role = AgentRole(self.role)
        if isinstance(self.status, str):
            self.status = AgentStatus(self.status)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["role"] = self.role.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AgentInfo":
        """Create from dictionary."""
        return cls(**data)


class AgentRegistry:
    """Registry for tracking active agents."""

    def __init__(self, data_file: Path):
        """Initialize agent registry.

        Args:
            data_file: Path to registry JSON file
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        """Load registry from file."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.agents = {
                    agent_id: AgentInfo.from_dict(agent_data)
                    for agent_id, agent_data in data.get("agents", {}).items()
                }
        else:
            self.agents = {}
            self._save_registry()

    def _save_registry(self):
        """Save registry to file."""
        data = {
            "agents": {
                agent_id: agent.to_dict() for agent_id, agent in self.agents.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_agent(
        self, agent_id: str, role: AgentRole, metadata: Optional[dict] = None
    ) -> AgentInfo:
        """Register a new agent.

        Args:
            agent_id: Unique agent identifier
            role: Agent role
            metadata: Additional metadata

        Returns:
            Created agent info

        Raises:
            ValueError: If CEO role already exists (only one CEO allowed)
        """
        # Enforce single CEO constraint
        if role == AgentRole.CEO:
            existing_ceo = [
                a for a in self.agents.values() if a.role == AgentRole.CEO and a.status != AgentStatus.TERMINATED
            ]
            if existing_ceo:
                raise ValueError(
                    f"Only one CEO agent allowed. Existing: {existing_ceo[0].agent_id}"
                )

        timestamp = datetime.now(timezone.utc).isoformat()
        agent = AgentInfo(
            agent_id=agent_id,
            role=role,
            status=AgentStatus.ACTIVE,
            started_at=timestamp,
            last_activity=timestamp,
            metadata=metadata or {},
        )

        self.agents[agent_id] = agent
        self._save_registry()
        return agent

    def update_activity(self, agent_id: str, current_task: Optional[str] = None):
        """Update agent's last activity timestamp.

        Args:
            agent_id: Agent ID
            current_task: Optional current task description
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        agent.last_activity = datetime.now(timezone.utc).isoformat()
        if current_task is not None:
            agent.current_task = current_task

        self._save_registry()

    def terminate_agent(self, agent_id: str):
        """Mark an agent as terminated.

        Args:
            agent_id: Agent ID to terminate
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        self.agents[agent_id].status = AgentStatus.TERMINATED
        self._save_registry()

    def get_active_agents(self, role: Optional[AgentRole] = None) -> list[AgentInfo]:
        """Get all active agents.

        Args:
            role: Optional filter by role

        Returns:
            List of active agents
        """
        agents = [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]

        if role:
            agents = [a for a in agents if a.role == role]

        return agents

    def get_ceo(self) -> Optional[AgentInfo]:
        """Get the active CEO agent.

        Returns:
            CEO agent info or None
        """
        ceos = [
            a
            for a in self.agents.values()
            if a.role == AgentRole.CEO and a.status != AgentStatus.TERMINATED
        ]
        return ceos[0] if ceos else None

--- coordination/inbox_manager.py ---
"""Inbox management for agent-to-agent messaging."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class MessagePriority(Enum):
    """Message priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Message:
    """Represents a message in the inbox."""

    message_id: str
    from_agent: str
    to_agent: str
    subject: str
    body: str
    priority: MessagePriority
    timestamp: str
    read: bool = False
    metadata: dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

        # Convert enum to string if needed
        if isinstance(self.priority, str):
            self.priority = MessagePriority(self.priority)

    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        data = asdict(self)
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create message from dictionary."""
        return cls(**data)


class InboxManager:
    """Manages agent-to-agent messaging."""

    def __init__(self, data_file: Path):
        """Initialize inbox manager.

        Args:
            data_file: Path to the inbox JSON file
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_messages()

    def _load_messages(self):
        """Load messages from file."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.messages = {
                    msg_id: Message.from_dict(msg_data)
                    for msg_id, msg_data in data.get("messages", {}).items()
                }
        else:
            self.messages = {}
            self._save_messages()

    def _save_messages(self):
        """Save messages to file."""
        data = {
            "messages": {
                msg_id: msg.to_dict() for msg_id, msg in self.messages.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        body: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Send a message from one agent to another.

        Args:
            from_agent: Sending agent name
            to_agent: Receiving agent name
            subject: Message subject
            body: Message body
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Created message
        """
        timestamp = datetime.now(timezone.utc)
        message_id = f"{from_agent}_to_{to_agent}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        message = Message(
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            subject=subject,
            body=body,
            priority=priority,
            timestamp=timestamp.isoformat(),
            read=False,
            metadata=metadata or {},
        )

        self.messages[message_id] = message
        self._save_messages()
        return message

    def get_unread_messages(self, agent: str) -> list[Message]:
        """Get unread messages for an agent.

        Args:
            agent: Agent name

        Returns:
            List of unread messages sorted by priority then timestamp
        """
        unread = [
            msg
            for msg in self.messages.values()
            if msg.to_agent == agent and not msg.read
        ]

        # Sort by priority (critical first) then timestamp (newest first)
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.MEDIUM: 2,
            MessagePriority.LOW: 3,
        }

        return sorted(
            unread, key=lambda m: (priority_order[m.priority], m.timestamp), reverse=True
        )

    def mark_as_read(self, message_id: str):
        """Mark a message as read.

        Args:
            message_id: Message ID to mark as read

        Raises:
            KeyError: If message_id doesn't exist
        """
        if message_id not in self.messages:
            raise KeyError(f"Message {message_id} not found")

        self.messages[message_id].read = True
        self._save_messages()

    def mark_all_as_read(self, agent: str):
        """Mark all messages for an agent as read.

        Args:
            agent: Agent name
        """
        for msg in self.messages.values():
            if msg.to_agent == agent and not msg.read:
                msg.read = True
        self._save_messages()

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        return self.messages.get(message_id)

    def export_markdown(self, agent: str) -> str:
        """Export inbox to markdown format.

        Args:
            agent: Agent name whose inbox to export

        Returns:
            Markdown-formatted inbox
        """
        lines = [
            f"# {agent.upper()} Inbox",
            "",
            "## ⚠️ CRITICAL: Read this file at START of EVERY session",
            "",
        ]

        unread = self.get_unread_messages(agent)
        if unread:
            lines.append("## Unread Messages")
            lines.append("")

            for msg in unread:
                priority_emoji = {
                    MessagePriority.CRITICAL: "🔴",
                    MessagePriority.HIGH: "🟠",
                    MessagePriority.MEDIUM: "🟡",
                    MessagePriority.LOW: "⚪",
                }[msg.priority]

                lines.append(f"### {priority_emoji} [{msg.timestamp[:10]}] From: {msg.from_agent}")
                lines.append(f"**Subject**: {msg.subject}")
                lines.append(f"**Priority**: {msg.priority.value.upper()}")
                lines.append("")
                lines.append(msg.body)
                lines.append("")
                lines.append("---")
                lines.append("")
        else:
            lines.append("## Unread Messages")
            lines.append("")
            lines.append("✅ No unread messages")
            lines.append("")

        # Show recent read messages
        read = [
            msg
            for msg in self.messages.values()
            if msg.to_agent == agent and msg.read
        ]
        read = sorted(read, key=lambda m: m.timestamp, reverse=True)[:5]

        if read:
            lines.append("## Recently Read")
            lines.append("")
            for msg in read:
                lines.append(
                    f"- [{msg.timestamp[:10]}] **{msg.subject}** (from {msg.from_agent})"
                )
            lines.append("")

        return "\n".join(lines)

--- coordination/cli.py ---
#!/usr/bin/env python3
"""CLI interface for agent coordination.

This is what Claude Code agents use via Bash to interact with the coordination system.

Usage:
    python3 coordination/cli.py startup <agent_id>           # Run startup checklist
    python3 coordination/cli.py inbox <agent_id>             # Check inbox
    python3 coordination/cli.py inbox-read <agent_id>        # Mark all messages read
    python3 coordination/cli.py tasks <agent_id>             # Check my tasks
    python3 coordination/cli.py task-start <task_id>         # Start a task
    python3 coordination/cli.py task-done <task_id>          # Complete a task
    python3 coordination/cli.py send <from> <to> <subject> <body> [priority]  # Send message
    python3 coordination/cli.py register <agent_id> <role>   # Register agent
    python3 coordination/cli.py status                       # Show system status
    python3 coordination/cli.py resources                    # Show resource usage
    python3 coordination/cli.py check-spawn [agent_type]     # Check if can spawn agent
    python3 coordination/cli.py check-duplicates <pattern>   # Check for duplicate work
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coordination import (
    CoordinationAPI,
    AgentRole,
    TaskPriority,
    MessagePriority,
)


def get_api() -> CoordinationAPI:
    """Get coordination API instance."""
    return CoordinationAPI(PROJECT_ROOT)


def cmd_startup(agent_id: str):
    """Run startup checklist for an agent."""
    api = get_api()
    summary = api.ceo_startup_checklist(agent_id)

    print("=" * 70)
    print(f"STARTUP CHECKLIST — {agent_id.upper()}")
    print("=" * 70)

    # Messages
    print(f"\nUNREAD MESSAGES: {summary['unread_messages']}")
    if summary["messages"]:
        print("-" * 70)
        for msg in summary["messages"]:
            priority_marker = {"critical": "!!!", "high": "!!", "medium": "!", "low": ""}
            marker = priority_marker.get(msg.priority.value, "")
            print(f"\n{marker} From: {msg.from_agent}")
            print(f"  Subject: {msg.subject}")
            print(f"  Priority: {msg.priority.value.upper()}")
            print(f"  Time: {msg.timestamp[:19]}")
            print(f"  ---")
            for line in msg.body.strip().split("\n"):
                print(f"  {line}")
        print()

    # Tasks
    print(f"\nACTIVE TASKS: {summary['active_tasks']}")
    if summary["tasks"]:
        print("-" * 70)
        for task in summary["tasks"]:
            status = "IN PROGRESS" if task.status.value == "in_progress" else "QUEUED"
            print(f"  [{task.priority.value.upper()}] {task.task_id}")
            print(f"    {task.description}")
            print(f"    Status: {status}")
        print()

    print("=" * 70)


def cmd_inbox(agent_id: str):
    """Check inbox for an agent."""
    api = get_api()
    messages = api.get_unread_messages(agent_id)

    if not messages:
        print(f"No unread messages for {agent_id}")
        return

    print(f"INBOX — {agent_id.upper()} ({len(messages)} unread)")
    print("-" * 70)
    for msg in messages:
        print(f"\nFrom: {msg.from_agent}")
        print(f"Subject: {msg.subject}")
        print(f"Priority: {msg.priority.value.upper()}")
        print(f"Time: {msg.timestamp[:19]}")
        print(f"ID: {msg.message_id}")
        print("---")
        print(msg.body.strip())
        print()


def cmd_inbox_read(agent_id: str):
    """Mark all messages as read."""
    api = get_api()
    api.mark_all_messages_read(agent_id)
    print(f"All messages marked as read for {agent_id}")


def cmd_tasks(agent_id: str):
    """List active tasks for an agent."""
    api = get_api()
    tasks = api.get_my_tasks(agent_id)

    if not tasks:
        print(f"No active tasks for {agent_id}")
        return

    print(f"TASKS — {agent_id.upper()} ({len(tasks)} active)")
    print("-" * 70)
    for task in tasks:
        status = "IN PROGRESS" if task.status.value == "in_progress" else "QUEUED"
        print(f"  [{task.priority.value.upper()}] {task.task_id}")
        print(f"    {task.description}")
        print(f"    Status: {status}")
        if task.dependencies:
            print(f"    Depends on: {', '.join(task.dependencies)}")
        print()


def cmd_task_start(task_id: str):
    """Start a task."""
    api = get_api()
    api.start_task(task_id)
    print(f"Task {task_id} started")


def cmd_task_done(task_id: str):
    """Complete a task."""
    api = get_api()
    api.complete_task(task_id)
    print(f"Task {task_id} completed")


def cmd_task_create(task_id: str, description: str, assigned_to: str, priority: str = "medium"):
    """Create a new task."""
    api = get_api()
    p = TaskPriority(priority)
    task = api.create_task(task_id, description, assigned_to, p)
    print(f"Task created: {task.task_id} -> {assigned_to} [{priority.upper()}]")


def cmd_send(from_agent: str, to_agent: str, subject: str, body: str, priority: str = "medium"):
    """Send a message."""
    api = get_api()
    p = MessagePriority(priority)
    msg = api.send_message(from_agent, to_agent, subject, body, p)
    print(f"Message sent: {msg.message_id}")


def cmd_register(agent_id: str, role: str):
    """Register a new agent."""
    api = get_api()
    r = AgentRole(role)
    try:
        agent = api.register_agent(agent_id, r)
        print(f"Agent registered: {agent.agent_id} (role={role})")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_status():
    """Show full system status."""
    api = get_api()

    print("=" * 70)
    print("COORDINATION SYSTEM STATUS")
    print("=" * 70)

    # Active agents
    agents = api.agent_registry.get_active_agents()
    print(f"\nACTIVE AGENTS: {len(agents)}")
    for agent in agents:
        task_info = f" — working on: {agent.current_task}" if agent.current_task else ""
        print(f"  {agent.agent_id} ({agent.role.value}){task_info}")

    # Active tasks
    all_tasks = api.task_manager.get_active_tasks()
    print(f"\nACTIVE TASKS: {len(all_tasks)}")
    for task in all_tasks:
        print(f"  [{task.priority.value.upper()}] {task.task_id} -> {task.assigned_to}: {task.description}")

    # Recent completed
    completed = api.task_manager.get_completed_tasks()[:3]
    if completed:
        print(f"\nRECENTLY COMPLETED:")
        for task in completed:
            print(f"  {task.task_id} ({task.completed_at[:10]})")

    print()


def cmd_export():
    """Export coordination data to readable markdown."""
    api = get_api()
    coord_dir = PROJECT_ROOT / "coordination"
    api.export_task_markdown(coord_dir / "TASK_ASSIGNMENTS.md")
    api.export_inbox_markdown("ceo", coord_dir / "CEO_INBOX.md")
    print("Exported to coordination/TASK_ASSIGNMENTS.md and coordination/CEO_INBOX.md")


def cmd_check_duplicates(pattern: str):
    """Check for duplicate work."""
    api = get_api()
    dupes = api.check_duplicate_work(pattern)
    if dupes:
        print(f"WARNING: Found {len(dupes)} potentially duplicate tasks:")
        for d in dupes:
            print(f"  - {d.task_id}: {d.description} (assigned to {d.assigned_to})")
    else:
        print("No duplicates found")


def cmd_resources():
    """Show system resource status."""
    api = get_api()
    status = api.get_resource_status()

    if not status:
        print("Resource manager not available")
        return

    print("=" * 70)
    print("SYSTEM RESOURCES")
    print("=" * 70)

    print(f"\nCPU Usage: {status.cpu_percent:.1f}%")
    print(f"Memory Usage: {status.memory_percent:.1f}%")
    print(f"Disk Free: {status.disk_free_gb:.1f} GB")

    print(f"\nActive Agents: {status.active_agents}")
    if status.agents_by_type:
        for agent_type, count in status.agents_by_type.items():
            print(f"  - {agent_type}: {count}")

    if status.circuit_breaker_open:
        print("\n⚠️  CIRCUIT BREAKER OPEN - No new agents allowed")

    if status.under_pressure:
        print("\n⚠️  SYSTEM UNDER PRESSURE - Reduced concurrency in effect")

    if status.warnings:
        print("\nWARNINGS:")
        for warning in status.warnings:
            print(f"  ⚠️  {warning}")

    print()


def cmd_check_spawn(agent_type: str = "general"):
    """Check if resources allow spawning a new agent."""
    api = get_api()
    can_spawn, reason = api.check_can_spawn_agent(agent_type)

    if can_spawn:
        print(f"✓ Can spawn {agent_type} agent: {reason}")
    else:
        print(f"✗ Cannot spawn {agent_type} agent: {reason}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    try:
        if cmd == "startup" and len(sys.argv) >= 3:
            cmd_startup(sys.argv[2])
        elif cmd == "inbox" and len(sys.argv) >= 3:
            cmd_inbox(sys.argv[2])
        elif cmd == "inbox-read" and len(sys.argv) >= 3:
            cmd_inbox_read(sys.argv[2])
        elif cmd == "tasks" and len(sys.argv) >= 3:
            cmd_tasks(sys.argv[2])
        elif cmd == "task-start" and len(sys.argv) >= 3:
            cmd_task_start(sys.argv[2])
        elif cmd == "task-done" and len(sys.argv) >= 3:
            cmd_task_done(sys.argv[2])
        elif cmd == "task-create" and len(sys.argv) >= 5:
            priority = sys.argv[5] if len(sys.argv) > 5 else "medium"
            cmd_task_create(sys.argv[2], sys.argv[3], sys.argv[4], priority)
        elif cmd == "send" and len(sys.argv) >= 6:
            priority = sys.argv[6] if len(sys.argv) > 6 else "medium"
            cmd_send(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], priority)
        elif cmd == "register" and len(sys.argv) >= 4:
            cmd_register(sys.argv[2], sys.argv[3])
        elif cmd == "status":
            cmd_status()
        elif cmd == "export":
            cmd_export()
        elif cmd == "check-duplicates" and len(sys.argv) >= 3:
            cmd_check_duplicates(sys.argv[2])
        elif cmd == "resources":
            cmd_resources()
        elif cmd == "check-spawn":
            agent_type = sys.argv[2] if len(sys.argv) >= 3 else "general"
            cmd_check_spawn(agent_type)
        else:
            print(f"Unknown command or missing args: {cmd}")
            print(__doc__)
            sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


=== 6. OVERSIGHT CODE ===

--- src/sparky/oversight/activity_logger.py ---
"""Structured JSONL agent activity logger.

Every agent session MUST initialize this logger and log all
task_started, task_completed, decision_made, and error_encountered events.
These logs are the Research Strategy Analyst's primary data source.

Logs are append-only, crash-safe (flush after every write), and
written to logs/agent_activity/{agent_id}_{date}.jsonl.

If the logger itself fails, the agent MUST continue working —
oversight failure never blocks research.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs/agent_activity")


class AgentActivityLogger:
    """Structured JSONL logger for agent activity tracking.

    Usage:
        logger = AgentActivityLogger(agent_id="ceo", session_id="phase-0-validation")
        logger.log_task_started("phase_0", "returns_calculations", "Implementing returns")
    """

    def __init__(self, agent_id: str, session_id: str, log_dir: Optional[Path] = None):
        self.agent_id = agent_id
        self.session_id = session_id
        self.log_dir = log_dir or LOG_DIR
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[OVERSIGHT] Failed to create log dir: {e}")

    def _log_file_path(self) -> Path:
        """Get today's log file path: {agent_id}_{date}.jsonl"""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{self.agent_id}_{date_str}.jsonl"

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a single JSON entry to the log file. Flush immediately."""
        try:
            path = self._log_file_path()
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            # Oversight failure must never block research
            logger.warning(f"[OVERSIGHT] Failed to write log entry: {e}")

    def _base_entry(self, action_type: str) -> dict[str, Any]:
        """Create base entry with common fields."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "action_type": action_type,
        }

    def log_task_started(self, phase: str, task: str, description: str) -> None:
        """Log the start of a STATE.yaml task."""
        entry = self._base_entry("task_started")
        entry.update({"phase": phase, "task": task, "description": description})
        self._write_entry(entry)

    def log_task_completed(
        self,
        phase: str,
        task: str,
        description: str,
        files_changed: list[str],
        git_commit: str,
    ) -> None:
        """Log completion of a STATE.yaml task."""
        entry = self._base_entry("task_completed")
        entry.update({
            "phase": phase,
            "task": task,
            "description": description,
            "files_changed": files_changed,
            "git_commit": git_commit,
        })
        self._write_entry(entry)

    def log_experiment(
        self,
        task: str,
        hypothesis: str,
        strategic_goal: str,
        result: dict[str, Any],
        conclusion: str,
        mlflow_run_id: str,
    ) -> None:
        """Log a completed experiment with results."""
        entry = self._base_entry("experiment_completed")
        entry.update({
            "task": task,
            "hypothesis": hypothesis,
            "strategic_goal": strategic_goal,
            "result": result,
            "conclusion": conclusion,
            "mlflow_run_id": mlflow_run_id,
        })
        self._write_entry(entry)

    def log_validation(
        self, mlflow_run_id: str, new_status: str, reason: str
    ) -> None:
        """Log a validation status change (preliminary -> validated -> proven)."""
        entry = self._base_entry("validation_check")
        entry.update({
            "mlflow_run_id": mlflow_run_id,
            "new_status": new_status,
            "reason": reason,
        })
        self._write_entry(entry)

    def log_decision(
        self,
        description: str,
        options: list[str],
        chosen: str,
        reasoning: str,
    ) -> None:
        """Log a decision made by the agent."""
        entry = self._base_entry("decision_made")
        entry.update({
            "description": description,
            "options": options,
            "chosen": chosen,
            "reasoning": reasoning,
        })
        self._write_entry(entry)

    def log_error(self, description: str, recovery: str) -> None:
        """Log an error and recovery action."""
        entry = self._base_entry("error_encountered")
        entry.update({"description": description, "recovery": recovery})
        self._write_entry(entry)

    def log_direction_change(
        self, old_direction: str, new_direction: str, reasoning: str
    ) -> None:
        """Log a change in research direction."""
        entry = self._base_entry("direction_change")
        entry.update({
            "old_direction": old_direction,
            "new_direction": new_direction,
            "reasoning": reasoning,
        })
        self._write_entry(entry)

    def log_hypothesis_proposed(
        self, hypothesis: str, strategic_goal: str, expected_outcome: str
    ) -> None:
        """Log a new research hypothesis."""
        entry = self._base_entry("hypothesis_proposed")
        entry.update({
            "hypothesis": hypothesis,
            "strategic_goal": strategic_goal,
            "expected_outcome": expected_outcome,
        })
        self._write_entry(entry)

--- src/sparky/oversight/resource_manager.py ---
"""Resource manager for preventing system overload.

Monitors system resources (CPU, memory, disk) and enforces limits on
concurrent agent spawning, model training, and data fetching to prevent
crashes and ensure system stability.

Usage:
    from sparky.oversight.resource_manager import ResourceManager

    manager = ResourceManager()

    # Before spawning agents
    if manager.can_spawn_agent(agent_type="model_training"):
        # spawn agent
        agent_id = "training-xyz"
        manager.register_agent(agent_id, "model_training")

    # After agent completes
    manager.unregister_agent(agent_id)

    # Check system health
    status = manager.get_system_status()
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a running agent."""

    agent_id: str
    agent_type: str  # "general", "model_training", "data_fetch", etc.
    started_at: datetime
    memory_mb: float = 0.0


@dataclass
class SystemStatus:
    """Current system resource status."""

    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    active_agents: int
    agents_by_type: Dict[str, int]
    under_pressure: bool
    circuit_breaker_open: bool
    warnings: List[str]


class ResourceManagerError(Exception):
    """Raised when resource limits are exceeded."""

    pass


class CircuitBreakerOpen(ResourceManagerError):
    """Raised when circuit breaker is open."""

    pass


class ResourceManager:
    """Manages system resources and enforces limits.

    This class prevents system crashes by:
    1. Tracking concurrent agents and enforcing concurrency limits
    2. Monitoring CPU, memory, and disk usage
    3. Implementing circuit breaker pattern for failures
    4. Gracefully degrading under pressure
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize resource manager.

        Args:
            config_path: Path to resource_limits.yaml. If None, uses default.
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "resource_limits.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Active agents tracking
        self.active_agents: Dict[str, AgentInfo] = {}

        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_opened_at: Optional[datetime] = None
        self.consecutive_errors = 0

        # Concurrency adjustments (for graceful degradation)
        self.current_max_concurrent = self.config["concurrency"]["max_concurrent_agents"]

        logger.info(
            f"ResourceManager initialized. Max concurrent agents: {self.current_max_concurrent}"
        )

    def can_spawn_agent(self, agent_type: str = "general") -> bool:
        """Check if a new agent can be spawned.

        Args:
            agent_type: Type of agent ("general", "model_training", "data_fetch")

        Returns:
            True if agent can be spawned, False otherwise

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
            ResourceManagerError: If resource limits exceeded
        """
        # Check circuit breaker
        self._check_circuit_breaker()

        # Check concurrency limits
        self._check_concurrency_limits(agent_type)

        # Check system resources
        self._check_system_resources()

        return True

    def register_agent(self, agent_id: str, agent_type: str = "general") -> None:
        """Register a newly spawned agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
        """
        if agent_id in self.active_agents:
            logger.warning(f"Agent {agent_id} already registered, updating")

        self.active_agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            started_at=datetime.now(timezone.utc),
        )

        logger.info(
            f"Agent registered: {agent_id} ({agent_type}). "
            f"Active agents: {len(self.active_agents)}/{self.current_max_concurrent}"
        )

        # Reset consecutive errors on successful registration
        self.consecutive_errors = 0

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister a completed agent.

        Args:
            agent_id: Unique identifier for the agent
        """
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return

        agent_info = self.active_agents.pop(agent_id)
        duration = (datetime.now(timezone.utc) - agent_info.started_at).total_seconds()

        logger.info(
            f"Agent unregistered: {agent_id} ({agent_info.agent_type}). "
            f"Duration: {duration:.1f}s. Active agents: {len(self.active_agents)}"
        )

    def get_system_status(self) -> SystemStatus:
        """Get current system resource status.

        Returns:
            SystemStatus with current resource usage and warnings
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Count agents by type
        agents_by_type: Dict[str, int] = {}
        for agent_info in self.active_agents.values():
            agents_by_type[agent_info.agent_type] = (
                agents_by_type.get(agent_info.agent_type, 0) + 1
            )

        # Check for warnings
        warnings = []
        alert = self.config["monitoring"]["alert_thresholds"]

        if cpu_percent > alert["cpu_percent"]:
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

        if memory.percent > alert["memory_percent"]:
            warnings.append(f"High memory usage: {memory.percent:.1f}%")

        disk_free_gb = disk.free / (1024**3)
        if disk_free_gb < alert["disk_free_gb"]:
            warnings.append(f"Low disk space: {disk_free_gb:.1f} GB free")

        # Check for long-running agents
        timeout_warning = self.config["concurrency"]["agent_timeout_warning"]
        now = datetime.now(timezone.utc)
        for agent_info in self.active_agents.values():
            duration = (now - agent_info.started_at).total_seconds()
            if duration > timeout_warning:
                warnings.append(
                    f"Agent {agent_info.agent_id} running for {duration/60:.1f} minutes"
                )

        # Determine if under pressure
        pressure = self.config["degradation"]
        under_pressure = (
            cpu_percent > pressure["pressure_cpu_threshold"]
            or memory.percent > pressure["pressure_memory_threshold"]
        )

        return SystemStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_free_gb=disk_free_gb,
            active_agents=len(self.active_agents),
            agents_by_type=agents_by_type,
            under_pressure=under_pressure,
            circuit_breaker_open=self.circuit_breaker_open,
            warnings=warnings,
        )

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and handle recovery.

        Raises:
            CircuitBreakerOpen: If circuit breaker is open and cooldown not expired
        """
        if not self.config["circuit_breaker"]["enabled"]:
            return

        if not self.circuit_breaker_open:
            return

        # Check if cooldown period has passed
        cooldown = self.config["circuit_breaker"]["cooldown_seconds"]
        elapsed = (datetime.now(timezone.utc) - self.circuit_breaker_opened_at).total_seconds()

        if elapsed > cooldown:
            if self.config["circuit_breaker"]["auto_recovery"]:
                logger.info("Circuit breaker cooldown expired, attempting recovery")
                self.circuit_breaker_open = False
                self.consecutive_errors = 0
                return

        raise CircuitBreakerOpen(
            f"Circuit breaker open. Cooldown: {cooldown - elapsed:.0f}s remaining. "
            f"Reason: {self.consecutive_errors} consecutive resource errors."
        )

    def _check_concurrency_limits(self, agent_type: str) -> None:
        """Check if concurrency limits allow spawning new agent.

        Args:
            agent_type: Type of agent to spawn

        Raises:
            ResourceManagerError: If concurrency limits exceeded
        """
        # Check global concurrent agent limit
        if len(self.active_agents) >= self.current_max_concurrent:
            self._handle_resource_error(
                f"Cannot spawn agent: at max concurrency "
                f"({len(self.active_agents)}/{self.current_max_concurrent})"
            )

        # Check type-specific limits
        if agent_type == "model_training":
            training_agents = sum(
                1 for a in self.active_agents.values()
                if a.agent_type == "model_training"
            )
            max_training = self.config["model_training"]["max_concurrent_training"]
            if training_agents >= max_training:
                self._handle_resource_error(
                    f"Cannot spawn training agent: at max training concurrency "
                    f"({training_agents}/{max_training})"
                )

        if agent_type == "data_fetch":
            fetch_agents = sum(
                1 for a in self.active_agents.values()
                if a.agent_type == "data_fetch"
            )
            max_fetch = self.config["data_fetching"]["max_concurrent_fetches"]
            if fetch_agents >= max_fetch:
                self._handle_resource_error(
                    f"Cannot spawn data fetch agent: at max fetch concurrency "
                    f"({fetch_agents}/{max_fetch})"
                )

    def _check_system_resources(self) -> None:
        """Check system resource availability.

        Raises:
            ResourceManagerError: If system resources critical
        """
        if not self.config["monitoring"]["enabled"]:
            return

        status = self.get_system_status()
        halt = self.config["monitoring"]["halt_thresholds"]

        # Check halt thresholds
        if status.cpu_percent > halt["cpu_percent"]:
            self._handle_resource_error(
                f"CPU usage critical: {status.cpu_percent:.1f}% > {halt['cpu_percent']}%"
            )

        if status.memory_percent > halt["memory_percent"]:
            self._handle_resource_error(
                f"Memory usage critical: {status.memory_percent:.1f}% > {halt['memory_percent']}%"
            )

        if status.disk_free_gb < halt["disk_free_gb"]:
            self._handle_resource_error(
                f"Disk space critical: {status.disk_free_gb:.1f} GB < {halt['disk_free_gb']} GB"
            )

        # Handle graceful degradation
        if status.under_pressure:
            self._apply_degradation()

    def _apply_degradation(self) -> None:
        """Apply graceful degradation under resource pressure."""
        if not self.config["degradation"]["auto_reduce_concurrency"]:
            return

        min_concurrent = self.config["degradation"]["min_concurrent_agents"]
        if self.current_max_concurrent > min_concurrent:
            old_max = self.current_max_concurrent
            self.current_max_concurrent = max(
                min_concurrent, self.current_max_concurrent - 1
            )
            logger.warning(
                f"System under pressure. Reducing max concurrent agents: "
                f"{old_max} -> {self.current_max_concurrent}"
            )

    def _handle_resource_error(self, error_msg: str) -> None:
        """Handle resource limit violation.

        Args:
            error_msg: Error message describing the violation

        Raises:
            ResourceManagerError: Always raised with error_msg
        """
        logger.error(error_msg)

        # Increment consecutive errors
        self.consecutive_errors += 1

        # Check if circuit breaker should open
        if self.config["circuit_breaker"]["enabled"]:
            max_errors = self.config["circuit_breaker"]["max_consecutive_errors"]
            if self.consecutive_errors >= max_errors:
                self.circuit_breaker_open = True
                self.circuit_breaker_opened_at = datetime.now(timezone.utc)
                logger.critical(
                    f"Circuit breaker opened after {self.consecutive_errors} consecutive errors"
                )

        raise ResourceManagerError(error_msg)

    def cleanup_stale_agents(self, force_kill_timeout: Optional[int] = None) -> List[str]:
        """Clean up agents that have exceeded timeout.

        Args:
            force_kill_timeout: Override timeout (seconds). If None, uses config.

        Returns:
            List of agent IDs that were cleaned up
        """
        if force_kill_timeout is None:
            force_kill_timeout = self.config["concurrency"]["agent_timeout_kill"]

        now = datetime.now(timezone.utc)
        stale_agents = []

        for agent_id, agent_info in list(self.active_agents.items()):
            duration = (now - agent_info.started_at).total_seconds()
            if duration > force_kill_timeout:
                logger.warning(
                    f"Force-killing stale agent: {agent_id} "
                    f"(running {duration/60:.1f} minutes)"
                )
                stale_agents.append(agent_id)
                self.unregister_agent(agent_id)

        return stale_agents


# Global singleton instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global ResourceManager singleton.

    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


=== 7. TIME TRACKING ===
# Time Tracking - Honest Progress Accounting

## CRITICAL RULE: 15-Minute Increments ONLY

DO NOT use "Day 0", "Day 1", "Day 2" labels. These inflate perceived progress.
Track actual elapsed time in 15-minute increments.

## Phase 2: Strategy Research

### Session 1: 2026-02-16 (15:00-15:15 UTC) — 15 minutes
**Work Completed**:
- Fixed block bootstrap Monte Carlo implementation
- Ran walk-forward validation (18 folds quarterly, 6 folds yearly)
- Tested 7 rule-based strategies
- Comprehensive comparison saved to `results/validation/yearly_strategy_comparison.json`

**Results**:
- Best: Multi-Timeframe Donchian (0.772 Sharpe)
- Baseline: Buy & Hold (0.719 Sharpe)
- Edge: +7.4% (marginal)

**Assessment**: Marginal result, NOT deployment-ready. Continue research.

**Elapsed Time**: 15 minutes
**Cumulative Time**: 15 minutes

---

### Session 2: [PENDING]
**Planned Work**:
- Kelly Criterion position sizing (45-60 min)
- ML models with cross-asset data (90-120 min)
- Alternative strategies (60 min)

**Target**: Find strategy with Sharpe ≥1.0

**Estimated Time**: 3-4 hours
**Cumulative Time**: TBD

---

## Historical Time (Prior Phases)

### Phase 0: Validation Bedrock — ~8 hours
### Phase 1: Data Layer — ~12 hours
### Phase 2a-2e: Initial Strategy Research — ~6 hours

**Total Project Time**: ~26 hours + current phase

---

## Time Accounting Rules

1. **Track actual elapsed time** (wall clock time)
2. **Report in 15-minute increments** (0:15, 0:30, 0:45, 1:00, etc.)
3. **NO "Day" labels** - they inflate progress perception
4. **Be honest** - 15 minutes of work = 15 minutes, not "Day 0 complete"
5. **Update after each session** - add new entry with work completed and time taken

---

## Deployment Readiness Criteria

**Time invested does NOT determine readiness. Results do.**

- ❌ 15 minutes with marginal results (0.772 Sharpe) = NOT READY
- ❌ 4 hours with marginal results = STILL NOT READY
- ✅ Any amount of time with Sharpe ≥1.2 = Potentially ready (if validated)
- ✅ Any amount of time with Sharpe ≥1.0 AND robust evidence = Potentially ready

**Current Status**: 15 minutes invested, 0.772 Sharpe = **NOT DEPLOYMENT-READY**


=== 8. BOOTSTRAP ===
# Sparky AI — Autonomous Research System Plan

## QUICK REFERENCE (read this on every session start)

```
WHERE AM I?     git branch --show-current && git log --oneline -5
WHAT'S DONE?    cat roadmap/STATE.yaml
WHAT'S NEXT?    Find first not_started task with all dependencies met
TESTS PASS?     pytest tests/ -v
ANY DECISIONS?  cat roadmap/DECISIONS.md
FULL CONTEXT?   Read CLAUDE.md → STATE.yaml → current phase file
```

**Phase gates:** 0,1 = informational (keep going) · 2,3 = BLOCKING (open PR, do productive work while waiting) · 4 = BLOCKING per task · 5 = autonomous

**Branch rules:** Always branch from `main`. Never commit to `main`. PRs for everything. CI must pass.

**If lost:** CLAUDE.md → STATE.yaml → current phase file → `git log --oneline -20`

---

## TABLE OF CONTENTS

1. [Quick Reference](#quick-reference-read-this-on-every-session-start)
2. [Your Mission](#your-mission)
3. [Model Transition: Opus → Sonnet](#model-transition-opus--sonnet)
4. [Multi-Agent Coordination & Git Workflow](#multi-agent-coordination--git-workflow)
   - Agent Roles (CEO, Quality, Analyst, Experiment)
   - Branch Strategy
   - Code Quality Agent Safety Rules
   - Research Strategy Analyst & Result Validation Protocol
   - Agent Security Boundaries
   - What to Do at Blocking Gates
5. [Step 1: Bootstrap the Repository](#step-1-bootstrap-the-repository)
   - CLAUDE.md, Configs, Trading Rules, Research Standards
   - Project Structure, Dependencies
6. [Phase 0: Validation Bedrock](#phase-0-validation-bedrock) — Returns, indicators, statistics tests
7. [Phase 1: Clean Data Layer](#phase-1-clean-data-layer) — CCXT, BGeometrics, CoinMetrics, cross-validation
8. [Phase 2: Features & Baselines](#phase-2-features--baselines) — On-chain features, ETH-specific, registry, backtester, baselines
9. [Phase 3: ML Models & Alpha](#phase-3-ml-models--alpha) — XGBoost, LSTM, Optuna tuning, ablation, holdout
10. [Phase 4: Paper Trading & Live](#phase-4-paper-trading--live-trading) — Signal pipeline, execution, monitoring
11. [Phase 5: Research Loop](#phase-5-autonomous-research-loop) — Inner/outer loop, experiment proposer, daemon
12. [Operating Principles](#operating-principles) — Research mindset, lessons from literature
13. [Begin](#begin)

---

## YOUR MISSION

You are the CEO agent of Sparky AI, an autonomous cryptocurrency quantitative research and trading system. This plan document defines your complete operating manual, architecture, and phased roadmap. Read it fully before executing anything.

**Your north star:** Produce trading models and strategies that generate real, statistically validated alpha on BTC and ETH. Infrastructure is a means to this end — build only what's needed, and get to producing research as fast as possible.

**Your operator:** AK is a quantitative trading professional at Spectre Data. He has deep technical expertise. Don't over-explain — show results.

**Your hardware:** NVIDIA DGX Spark GB10 — 128GB unified memory, 1 PFLOP FP4, 20 ARM Cortex cores, Ubuntu. More than sufficient for this workload.

**Your workflow:** This document was loaded via `/plan`. The GitHub repository `sparky-ai` already exists and is cloned locally. You will work in feature branches, commit frequently, and open pull requests on GitHub for each phase.

---

## MODEL TRANSITION: OPUS → SONNET

This project uses two model tiers strategically:

**Phase 1 (Opus 4.6):** Phases 0-2 — architecture decisions, data pipeline design,
feature engineering, and establishing all patterns and conventions. Opus handles the
ambiguous judgment calls: "Is this data source reliable?", "Should this be a separate
module?", "Is this feature leaking information?"

**Transition checkpoint:** After Phase 2 PR is merged and AK approves baselines.
At this point:
- All coding patterns are established (Pydantic types, structured logging, service layer)
- Test suite provides >80% coverage as a safety net
- Feature pipeline is complete and validated
- Backtester, cost model, and statistical tests are proven
- CLAUDE.md contains all conventions a new agent needs to follow

**Phase 2 (Sonnet 4.5):** Phases 3-5 — model training experiments, hyperparameter
tuning, paper trading, and research loop execution. These are more structured tasks
following established patterns. Sonnet is faster and more cost-effective for:
- Running MLflow experiments with defined configs
- Training models using established pipeline
- Executing paper trading with built infrastructure
- Running the inner/outer research loop

**Handoff requirements (Opus must complete before transition):**
1. All Phase 0-2 tests passing (>80% coverage)
2. CLAUDE.md fully documents every convention and pattern
3. Every module has docstrings explaining design decisions
4. RESEARCH_LOG.md has baseline results and data quality findings
5. STATE.yaml accurately reflects all completed work
6. CI pipeline operational (tests run on every PR)
7. At least one complete experiment logged in MLflow as a template
8. `roadmap/SONNET_HANDOFF.md` written by Opus summarizing:
   - Key architectural decisions and WHY they were made
   - Common pitfalls to avoid (e.g., target variable timing, leakage)
   - Where to find what (file map for each concern)
   - "If you're unsure, check..." guidance for ambiguous situations

**For Sonnet-phase CLAUDE.md additions** (Opus writes these before handoff):
- Explicit decision trees for common situations (not just principles)
- Copy-paste code templates for adding new features, new experiments
- Checklist-style task execution guides for Phase 3+ tasks
- "If X, then Y" rules instead of open-ended guidance

---

## MULTI-AGENT COORDINATION & GIT WORKFLOW

### The Problem
Multiple agents (CEO research agent, code quality agent, future experiment agents)
may operate on the same codebase. Without coordination, they will produce merge
conflicts, break each other's work, and create instability — exactly like a human
engineering team without git discipline.

### Design Principle: Branch Isolation
**Every agent works on its own branch. Nobody touches `main` directly.**
This mirrors how a small quant shop's engineering team operates:
- Each person (agent) checks out their own copy of the code
- Work happens in isolation on feature branches
- PRs are the coordination mechanism (review → CI → merge)
- `main` is always stable, always passing tests
- CI is the automated gatekeeper — no PR merges if tests fail

### Agent Roles

#### CEO Agent (you — the main Claude Code session)
- Executes research phases (Phase 0-5) on `phase-N/*` branches
- The primary builder: data pipelines, features, models, backtesting, trading
- Opens PRs when phases complete
- Owns RESEARCH_LOG.md and STATE.yaml updates
- Can spawn quality sweeps between phases

#### Code Quality Agent (periodic, spawned by CEO or run as separate session)
- Operates ONLY on `quality/*` branches forked from latest `main`
- **NEVER touches in-flight feature branches** — this is the critical safety rule
- Responsibilities:
  1. Enhance and maintain CI pipeline (tighten linting, bump coverage thresholds)
  2. Run linting (ruff), type checking (pyright), test coverage analysis
  3. Refactor repetitive or brittle code into shared utilities
  4. Enforce coding standards (Pydantic types, structured logging, docstrings)
  5. Add missing tests for uncovered code paths
  6. Update documentation when it drifts from implementation
- Trigger points (when to activate):
  - After each phase PR merges to `main`
  - Weekly maintenance sweep
  - Before major phases (e.g., before Phase 3 ML work — ensure solid foundation)
  - When CEO agent notices code smell and defers it

#### Future: Parallel Research Agents (Phase 5+)
- When running multiple experiments simultaneously, each gets its own branch
- `experiment/sopr-capitulation-filter`, `experiment/eth-onchain-ablation`, etc.
- Each experiment branch is self-contained: modify configs, add features, run
- Results are logged in RESEARCH_LOG.md (on their branch), then merged to `main`
- Experiments that conflict on features or data should be run sequentially, not in parallel

**STATE.yaml coordination (implement when parallel agents are introduced):**
When multiple agents run concurrently, STATE.yaml becomes a race condition risk.
Before Phase 5 deploys parallel agents, implement file-based locking:
`src/sparky/coordination/state_lock.py` using `fcntl.flock()` for atomic read-modify-write.
Until then, single-agent operation means no locking is needed — just read-update-commit.

#### Research Strategy Analyst (enters Phase 3, critical at Phase 5)

**The problem this solves:** Without oversight, research agents will drift. They'll
chase spurious correlations, flip-flop between contradictory conclusions, duplicate
work, or optimize for metrics that don't translate to real trading. At scale (Phase 5
with parallel agents), this becomes unmanageable without systematic evaluation.
AK and the oversight team need a clear, honest picture of research quality and direction.

**When hired:**
- Phase 0-2: The LOG FORMAT is established. All agents write structured activity logs
  from day one. This is non-negotiable — retrofitting logs later is impossible.
- Phase 3: The Analyst agent is activated when experiments begin. It reviews every
  experiment result, enforces validation protocols, and produces quality reports.
- Phase 5: The Analyst becomes the primary oversight mechanism when parallel agents
  make manual review impractical.

**Responsibilities:**
1. **Research Quality Evaluation** — Review every experiment for statistical rigor:
   Are confidence intervals reasonable? Is the Sharpe ratio inflated by look-ahead?
   Does the result survive the leakage detector? Is the sample size sufficient?
2. **Result Validation & Anti-Flip-Flop** — No result is "proven" until it passes
   the validation protocol (see below). If a finding contradicts a previous finding,
   BOTH get flagged for deeper investigation.
3. **Strategic Alignment Scoring** — Every experiment should serve a defined strategic
   goal. If >30% of experiments in a week don't map to any goal, agents are drifting.
4. **Agent Performance Reports** — Weekly report to AK covering:
   - Experiments completed, validated, invalidated per agent
   - Research direction summary (where is effort going?)
   - Flip-flop flags (findings that reversed)
   - Diminishing returns signals (10 experiments on same hypothesis with no improvement)
   - Resource utilization (GPU hours per validated finding)
   - Recommended research pivots
5. **Hypothesis Deduplication** — At scale, multiple agents may propose the same
   experiment. Analyst checks the experiment queue against completed work.
6. **Drift Detection** — If an agent's last N experiments show declining Sharpe,
   increasing p-values, or wandering feature importance, flag for redirection.

**Operates on:** `analyst/*` branches from `main`
**Reads:** `logs/agent_activity/`, MLflow experiments, RESEARCH_LOG.md, STATE.yaml
**Writes:** `results/analyst_reports/`, DECISIONS.md (recommendations for AK)
**CANNOT:** Modify agent code, change experiment configs, alter trading rules, or
stop/start other agents. The Analyst is read-only + reports. AK makes redirection decisions.

### Result Validation Protocol (enforced from Phase 3)

Every experiment result goes through a lifecycle:

```
PRELIMINARY → VALIDATED → PROVEN
     ↓             ↓
 INVALIDATED   INVALIDATED
```

**PRELIMINARY** (just completed):
- Single run completed, metrics logged to MLflow
- NOT actionable. Cannot be cited as evidence for decisions.

**VALIDATED** (passed reproducibility):
- Multi-seed check: 5 seeds, Sharpe std < 0.3 across seeds
- Walk-forward consistency: no single fold drives >50% of total return
- Leakage detector: all 3 tests pass
- Feature stability: importance rankings consistent across folds
- p-value survives Benjamini-Hochberg correction at current experiment count

**PROVEN** (survives out-of-sample):
- Holdout test (unseen data) confirms direction and magnitude
- Paper trading confirms for 30+ days (Phase 4+)
- No contradictory validated findings exist
- Analyst has reviewed and signed off

**INVALIDATED** (failed at any stage):
- Tag in MLflow: `validation_status: invalidated`, `invalidation_reason: "..."`
- Log in RESEARCH_LOG.md with full explanation
- ALL downstream decisions that relied on this result flagged for re-evaluation

**Anti-Flip-Flop Rule:**
If a new finding contradicts a previously VALIDATED or PROVEN finding:
1. STOP. Do not update RESEARCH_LOG.md conclusions.
2. Log both findings side-by-side with full configs and data ranges.
3. Run BOTH experiments on identical data splits (same dates, same seeds).
4. If contradiction persists: flag as `[CONFLICTING EVIDENCE]` in DECISIONS.md.
5. AK or Analyst determines which is correct (usually the one with larger sample,
   more recent data, or better statistical power).
6. The losing finding is INVALIDATED with clear reason.
Never silently overwrite a previous conclusion. Contradictions are information.

### Structured Agent Activity Logs (from day one)

**Every agent writes structured JSONL logs to `logs/agent_activity/`.**
This is the Analyst's primary data source. Format is established in Phase 0
and MUST be used by all agents from the first task.

Log location: `logs/agent_activity/{agent_type}_{date}.jsonl`
Examples: `ceo_2025-07-15.jsonl`, `quality_2025-07-15.jsonl`, `analyst_2025-07-20.jsonl`

**Log entry schema (one JSON object per line):**
```json
{
  "timestamp": "2025-07-15T14:32:00Z",
  "agent_id": "ceo",
  "session_id": "phase-3-experiment-batch-1",
  "action_type": "experiment_completed",
  "phase": "phase_3",
  "task": "feature_ablation_experiments",
  "description": "A3: XGBoost + technical + on-chain on BTC 30d",
  "hypothesis": "On-chain features improve BTC 30d prediction vs technical-only",
  "strategic_goal": "validate_onchain_alpha",
  "result": {
    "sharpe": 0.72,
    "sharpe_ci_95": [0.31, 1.13],
    "p_value": 0.023,
    "max_drawdown": -0.18,
    "validation_status": "preliminary",
    "mlflow_run_id": "abc123def456"
  },
  "conclusion": "On-chain features add +0.15 Sharpe vs A1 baseline (technical-only)",
  "files_changed": ["results/ablation_A3.json"],
  "git_commit": "a1b2c3d",
  "duration_minutes": 45,
  "gpu_memory_peak_mb": 2400
}
```

**Required action_types (agent must log ALL of these):**
- `task_started` — Beginning work on a STATE.yaml task
- `task_completed` — Task finished, tests passing
- `experiment_started` — MLflow experiment kicked off
- `experiment_completed` — Experiment finished with results
- `validation_check` — Running validation protocol on a result
- `result_validated` — Result promoted from preliminary → validated
- `result_proven` — Result promoted from validated → proven
- `result_invalidated` — Result failed validation or was contradicted
- `decision_made` — Agent chose between options (log reasoning)
- `error_encountered` — Something went wrong (log what and recovery)
- `hypothesis_proposed` — New research hypothesis generated
- `direction_change` — Agent pivoting research focus (log why)

**Strategic goals (defined in `configs/research_strategy.yaml`):**
```yaml
# research_strategy.yaml — Updated by AK, read by all agents
# The Analyst scores experiments against these goals

strategic_goals:
  validate_onchain_alpha:
    description: "Prove on-chain features add predictive power beyond technical indicators"
    priority: 1  # Highest
    success_criteria: "Validated Sharpe improvement >0.1 with on-chain vs without"
    phase: 3

  eth_specific_features:
    description: "Determine if ETH-specific features (gas, staking) add unique alpha"
    priority: 2
    success_criteria: "ETH ablation A6 vs A7 shows Sharpe improvement >0.1"
    phase: 3

  optimal_horizon:
    description: "Identify which prediction horizon (7/14/30d) is most profitable"
    priority: 2
    success_criteria: "One horizon significantly outperforms others (p<0.05)"
    phase: 3

  model_robustness:
    description: "Ensure chosen model is stable across seeds, folds, and regimes"
    priority: 1
    success_criteria: "Multi-seed std<0.3, no single fold >50% of return"
    phase: 3

  paper_trading_confirmation:
    description: "Confirm backtest results hold in forward-looking paper trading"
    priority: 1
    success_criteria: "Paper Sharpe within 50% of backtest Sharpe over 90 days"
    phase: 4

  autonomous_discovery:
    description: "Research loop generates novel validated hypotheses without human input"
    priority: 3
    success_criteria: "1+ validated finding per week from autonomous loop"
    phase: 5

# Alignment threshold: If >30% of weekly experiments don't map to any goal, flag drift
drift_threshold_pct: 30

# Diminishing returns: If 10+ experiments on same hypothesis with <5% improvement, pivot
diminishing_returns_threshold: 10
```

### Agent Activity Logger (implementation)

**File:** `src/sparky/oversight/activity_logger.py`
**Used by:** ALL agents, from Phase 0 onward

```python
# Minimal interface — agents call these functions, not raw file writes
class AgentActivityLogger:
    def __init__(self, agent_id: str, session_id: str):
        """Initialize logger for this agent session"""

    def log_task_started(self, phase: str, task: str, description: str): ...
    def log_task_completed(self, phase: str, task: str, description: str,
                           files_changed: list[str], git_commit: str): ...
    def log_experiment(self, task: str, hypothesis: str, strategic_goal: str,
                       result: dict, conclusion: str, mlflow_run_id: str): ...
    def log_validation(self, mlflow_run_id: str, new_status: str, reason: str): ...
    def log_decision(self, description: str, options: list[str],
                     chosen: str, reasoning: str): ...
    def log_error(self, description: str, recovery: str): ...
```

Every CEO agent session starts with:
```python
from sparky.oversight.activity_logger import AgentActivityLogger
logger = AgentActivityLogger(agent_id="ceo", session_id="phase-N-description")
```

The logger writes to `logs/agent_activity/` in JSONL format. It is append-only
and crash-safe (flush after every write). If the logger itself fails, the agent
should continue working — oversight failure must never block research.

### Branch Strategy

```
main (protected — only updated via merged PRs, always green)
│
├── phase-0/validation-bedrock      # CEO: Phase 0 work
├── phase-1/data-layer              # CEO: Phase 1 work
├── phase-2/features-baselines      # CEO: Phase 2 work
├── phase-3/ml-models               # CEO: Phase 3 work
├── phase-4/trading                 # CEO: Phase 4 work
├── phase-5/research-loop           # CEO: Phase 5 work
│
├── quality/ci-setup                # Quality Agent: GitHub Actions pipeline
├── quality/refactor-data-layer     # Quality Agent: cleanup after Phase 1 merge
├── quality/type-safety-sweep       # Quality Agent: add missing Pydantic types
├── quality/test-coverage-gaps      # Quality Agent: fill coverage gaps
│
├── experiment/sopr-filter          # Research: experiment branch (Phase 5+)
├── experiment/funding-rate-signal  # Research: experiment branch (Phase 5+)
└── experiment/eth-onchain-alpha    # Research: experiment branch (Phase 5+)
```

### Code Quality Agent — Safety Rules (NON-NEGOTIABLE)

```
RULE 1: BRANCH FROM MAIN ONLY
        Quality agent ONLY branches from the latest `main`.
        It never touches, rebases onto, or merges from in-flight feature branches.
        This is how you avoid stepping on the CEO agent's work.

RULE 2: GREEN BEFORE YOU START
        Run `pytest tests/ -v` BEFORE making any changes.
        If tests don't pass on `main`, STOP — something else is broken.
        File an issue or notify CEO agent. Do not proceed.

RULE 3: GREEN AFTER EVERY CHANGE
        Run `pytest tests/ -v` AFTER every change.
        If any test fails, `git checkout -- .` to revert immediately.
        Refactoring that breaks tests is NOT refactoring — it's a bug.

RULE 4: STRUCTURAL CHANGES ONLY
        Quality agent changes are PURELY structural:
        ✓ Renaming for clarity
        ✓ Extracting common code into shared utilities
        ✓ Adding type hints, docstrings, logging
        ✓ Adding tests for uncovered code
        ✓ Linting and formatting
        ✓ CI pipeline configuration
        ✗ NEVER modify business logic, model parameters, or feature calculations
        ✗ NEVER modify trading rules (configs/trading_rules.yaml is IMMUTABLE)
        ✗ NEVER change test assertions (if a test catches a real bug, that's for CEO agent)

RULE 5: SAME REVIEW PROCESS
        Quality agent PRs go through the same review as feature PRs.
        Title: "Quality: [description]"
        Body: what changed, why, before/after test + coverage results.

RULE 6: ATOMIC AND REVERSIBLE
        One refactoring concern per PR. If a PR touches 15 files for
        3 different reasons, split it into 3 PRs.
        Every PR should be safe to revert with a single `git revert`.

RULE 7: NO CONCURRENT EDITS
        If CEO agent is actively working on a phase branch that modifies
        the same files quality agent wants to refactor — WAIT.
        The quality agent works on MERGED code, not in-flight code.
        
        Decision tree:
        1. Run `gh pr list` to see what's in flight
        2. If no open PRs → proceed freely
        3. If open PRs exist → check which files they touch (`gh pr diff <num>`)
        4. If overlap with your planned changes → SKIP those files, work on others
        5. If ALL your planned changes overlap → wait for the PR to merge, then start
        6. If unsure → wait. Better to delay than to create merge conflicts.
```

### CI Pipeline (created in bootstrap scaffold)

The CEO agent creates `.github/workflows/ci.yml` as part of the initial scaffold
commit (Step 1). This eliminates the chicken-and-egg problem — CI is active from
the very first PR. The Quality Agent's first job is to ENHANCE it after Phase 0/1
merge (add stricter linting enforcement, bump coverage thresholds, etc.).

CI workflow template for the scaffold:

```yaml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -v --tb=short
      - name: Type check (advisory)
        run: pip install pyright && pyright src/sparky/ || true
      - name: Lint (advisory)
        run: pip install ruff && ruff check src/ tests/ || true

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Coverage report
        run: pytest tests/ --cov=src/sparky --cov-report=term-missing --cov-fail-under=60
```

This ensures every PR (from ANY agent) automatically runs tests.
No PR merges if tests fail. CI is the impartial gatekeeper that protects `main`.

Initially type checking and linting are advisory (`|| true`) — they become
enforced after Phase 2 when the codebase stabilizes. The Quality Agent upgrades
this by removing `|| true` and adding stricter thresholds as the codebase matures.

### Code Quality Agent — Refactoring Workflow

```bash
# 1. Start from clean main
git checkout main && git pull origin main

# 2. Check nothing is in flight that overlaps
gh pr list

# 3. Create quality branch
git checkout -b quality/descriptive-name

# 4. Confirm green baseline
pytest tests/ -v

# 5. Make ONE focused refactoring change
# ... edit files ...

# 6. Confirm still green
pytest tests/ -v

# 7. Commit
git commit -m "refactor: extract common retry logic into shared utility"

# 8. Repeat 5-7 for additional RELATED changes (keep atomic per concern)

# 9. Push and open PR
git push -u origin quality/descriptive-name
gh pr create --title "Quality: extract retry utility" --body "..."
```

### Agent Security Boundaries

Each agent type has explicit permissions. Violations of these boundaries should be
treated as bugs, not judgment calls.

**CEO Agent — CAN:**
- Create/modify any source code in `src/sparky/`
- Create/modify tests in `tests/`
- Run `pytest`, `python scripts/*.py`, `pip install`
- Read/write data files in `data/`
- Run MLflow experiments
- Update STATE.yaml, DECISIONS.md, RESEARCH_LOG.md
- Create branches, open PRs via `gh`
- Install Python packages listed in pyproject.toml

**CEO Agent — CANNOT:**
- Push to `main` directly
- Modify `configs/trading_rules.yaml` without AK approval
- Execute live trades without explicit human gate approval
- Delete git history (`git push --force`, `git rebase` on shared branches)
- Install system-level packages (`apt install`) without documenting why
- Access or store API keys in code (must use configs/secrets.yaml)

**Quality Agent — CAN:**
- Refactor code structure (rename, extract, reorganize)
- Add tests, type hints, docstrings, logging
- Modify CI pipeline (`.github/workflows/`)
- Run linting and formatting tools
- Update documentation

**Quality Agent — CANNOT:**
- Modify business logic, model parameters, feature calculations, or trading rules
- Change test assertions (if a test catches a real bug, that's CEO agent's job)
- Work on in-flight feature branches (only `main`)
- Modify STATE.yaml task statuses
- Run experiments or modify RESEARCH_LOG.md findings

**Research Strategy Analyst — CAN:**
- Read all agent activity logs (`logs/agent_activity/`)
- Read all MLflow experiments and metrics
- Read RESEARCH_LOG.md, STATE.yaml, configs/research_strategy.yaml
- Generate analyst reports to `results/analyst_reports/`
- Write recommendations to DECISIONS.md (tagged `[ANALYST RECOMMENDATION]`)
- Run `scripts/analyst_report.py`
- Validate experiment results via result_validator

**Research Strategy Analyst — CANNOT:**
- Modify any source code, tests, or configs
- Start, stop, or redirect other agents directly (recommendations only — AK decides)
- Change experiment parameters, feature definitions, or model configs
- Modify RESEARCH_LOG.md findings or conclusions (read-only)
- Promote or invalidate results without logging the reason
- Access trading systems or modify positions
- The Analyst is strictly READ + REPORT. All steering goes through AK.

**All Agents — NEVER:**
- Store secrets, API keys, or credentials in git-tracked files
- Run `rm -rf` on data directories without explicit instruction
- Modify another agent's in-flight branch
- Skip tests before committing
- Ignore CI failures on PRs
- Log experiment results without the activity_logger (structured logs are mandatory)

### What to Do at Blocking Gates

**Agents should NEVER be idle.** When waiting for AK to review a blocking gate PR,
use the time productively. Priority order:

1. **Quality sweep** (highest value): Switch to quality agent role.
   `git checkout main && git checkout -b quality/post-phase-N-cleanup`
   - Run `pytest --cov` and fill coverage gaps
   - Add missing docstrings and type hints
   - Run `ruff check src/` and fix linting issues
   - Refactor any code duplication noticed during the phase
   - Open quality PR — this can merge independently of the blocking gate

2. **Documentation**: Update CLAUDE.md with any patterns or conventions discovered
   during the phase. Write or improve README sections. Ensure RESEARCH_LOG.md
   is complete and well-organized.

3. **Data exploration** (if Phase 1+ complete): Explore feature distributions,
   correlations, regime patterns. Write findings to RESEARCH_LOG.md. This informs
   future phases without committing to code changes.

4. **Test hardening**: Add edge case tests, property-based tests, integration tests.
   These go on the quality branch.

5. **Sonnet handoff prep** (if approaching Phase 2 gate): Write decision trees,
   code templates, and checklists for Phase 3+ tasks. Create `roadmap/SONNET_HANDOFF.md`.

**Rules for gate-waiting work:**
- All work goes on `quality/*` or `docs/*` branches, never the blocked phase branch
- Do NOT start the next phase's code — that defeats the purpose of the gate
- Quality/docs PRs can merge to `main` while the phase PR awaits review
- If AK hasn't responded in 48 hours, add a comment to the PR as a reminder

### Coordination Rules (All Agents)

**Starting any work:**
```bash
git checkout main
git pull origin main
git checkout -b [agent-type]/[description]
```

**During work:**
- Commit frequently (every completed task)
- Run `pytest tests/ -v` before each commit
- If you need code from another branch that hasn't merged yet:
  DO NOT merge it into your branch. Wait for it to merge to `main`,
  then rebase: `git fetch origin && git rebase origin/main`

**Completing work:**
```bash
git push -u origin [branch-name]
gh pr create --title "[Type]: [Description]" --body "..."
```

**Avoiding conflicts (the key coordination challenge):**
- Two agents should NEVER edit the same file simultaneously
  (branch isolation from `main` prevents this if everyone follows the rules)
- STATE.yaml and RESEARCH_LOG.md are append-mostly by design
  (new entries at top) to minimize merge conflicts
- If conflicts do occur on merge: the LATER PR resolves them
  (rebase onto latest `main`, fix conflicts, re-run tests)

**Rebase, don't merge:**
- When `main` has moved ahead of your branch:
  `git fetch origin && git rebase origin/main`
- This keeps history clean and linear

### Commit Conventions
```
feat:     New feature or capability
fix:      Bug fix
test:     Adding or updating tests
data:     Data fetching, storage, or pipeline changes
docs:     Documentation updates (RESEARCH_LOG, DECISIONS, README)
chore:    Project setup, config changes, dependency updates
refactor: Code restructuring (no behavior change) [Quality Agent]
ci:       CI pipeline changes [Quality Agent]
quality:  Test coverage, linting, type safety [Quality Agent]
```

### PR Body Template
```
## [Phase N / Quality]: [Name]

### What was built/changed
- [bullet summary]

### Test results
- `pytest tests/ -v` → X tests passing
- Coverage: XX% (delta from previous)
- [key highlights]

### Key findings (if research phase)
- [metrics, data quality results, or research findings]

### Gate type
- [BLOCKING/INFORMATIONAL]: [what needs review]

### Next
- [what comes next]
```

### GitHub CLI
Use `gh` (GitHub CLI) for PR operations:
```bash
gh pr create --title "Phase 0: Validation Bedrock" --body "..."
gh pr list          # Check what's in flight (important before quality sweeps!)
gh pr view [number]
gh pr merge [number]  # Only AK does this for blocking gates
```

If `gh` is not installed, install it first. If auth is needed, notify AK.

---

## STEP 1: BOOTSTRAP THE REPOSITORY

Create the complete project structure, all configuration files, and the autonomous operating system on a `phase-0/validation-bedrock` branch. This is your first branch — it includes both the scaffold AND Phase 0 work.

### 1.1 Set Up Working Branch

```bash
cd ~/sparky-ai   # Repo already cloned from GitHub
git checkout -b phase-0/validation-bedrock
```

### 1.2 Create `CLAUDE.md` (Project Intelligence File)

This is your operating manual. You will read this at the start of every session.

```markdown
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
- See MULTI-AGENT COORDINATION section in the plan for full safety rules

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
├── .github/
│   └── workflows/
│       └── ci.yml                 # Tests + coverage on every PR (created in scaffold)
├── CLAUDE.md                      # This file — your operating manual
├── pyproject.toml                 # Python project config
├── configs/                       # Experiment and system configs
│   ├── system.yaml                # Global settings, paths, API endpoints
│   ├── data_sources.yaml          # Data source configuration and priorities
│   ├── trading_rules.yaml         # IMMUTABLE trading rules and gates
│   ├── research_strategy.yaml     # Strategic goals & alignment thresholds (AK-owned)
│   ├── secrets.example.yaml       # Template for API keys (tracked, no real values)
│   ├── secrets.yaml               # Actual API keys (gitignored)
│   ├── active_model.yaml          # Points to current production model artifact
│   └── experiments/               # Auto-generated experiment configs
├── roadmap/                       # Task management
│   ├── STATE.yaml                 # Progress tracker (machine-readable)
│   ├── DECISIONS.md               # Human-agent communication log
│   ├── RESEARCH_LOG.md            # Running log of findings
│   ├── SONNET_HANDOFF.md          # Opus writes this before Sonnet transition
│   └── phases/                    # Detailed phase instructions
│       ├── phase_0_validation.md
│       ├── phase_1_data.md
│       ├── phase_2_features_baselines.md
│       ├── phase_3_models.md
│       └── phase_4_trading.md
├── src/sparky/                    # Core library
│   ├── types/                     # Pydantic models for runtime validation
│   │   ├── market_types.py        # OHLCV, OnChainMetric, FeatureMatrix
│   │   ├── signal_types.py        # Signal, Prediction, ModelOutput
│   │   ├── portfolio_types.py     # Position, PortfolioState, TradeOrder
│   │   └── config_types.py        # Validated config schemas
│   ├── data/                      # Data fetching, storage, quality
│   │   ├── price.py               # CCXT-backed price/derivatives fetcher
│   │   ├── onchain_bgeometrics.py # BGeometrics on-chain fetcher
│   │   ├── onchain_coinmetrics.py # CoinMetrics Community on-chain fetcher
│   │   ├── onchain_blockchain.py  # Blockchain.com raw BTC stats
│   │   ├── source_selector.py     # Cross-validates and selects best source per metric
│   │   ├── market_context.py      # CoinGecko daily market snapshots
│   │   ├── storage.py             # Parquet read/write with metadata
│   │   └── quality.py             # Data quality checks
│   ├── features/                  # Feature engineering
│   │   ├── returns.py             # Log returns, realized vol, Sharpe
│   │   ├── technical.py           # RSI, EMA, MACD, momentum
│   │   ├── onchain.py             # Hash ribbon, NVT, MVRV, SOPR features
│   │   ├── registry.py            # Feature catalog with metadata
│   │   └── selection.py           # Correlation filter, importance threshold, stability
│   ├── models/                    # Model implementations
│   │   ├── baselines.py           # Buy-hold, momentum, equal-weight
│   │   ├── xgboost_model.py       # XGBoost wrapper
│   │   ├── lstm_model.py          # LSTM wrapper
│   │   └── ensemble.py            # Model combination
│   ├── backtest/                  # Backtesting engine
│   │   ├── engine.py              # Walk-forward backtester
│   │   ├── costs.py               # Transaction cost model
│   │   ├── statistics.py          # Sharpe, bootstrap CI, significance
│   │   └── leakage_detector.py    # Shuffled-label test + boundary checks
│   ├── portfolio/                 # Position sizing, risk management
│   │   ├── construction.py        # Position sizing, allocation
│   │   └── risk.py                # Drawdown limits, correlation monitoring
│   ├── trading/                   # Paper and live trading engines
│   │   ├── paper_engine.py        # Paper trading simulator
│   │   ├── live_engine.py         # Live Binance execution
│   │   ├── signal_pipeline.py     # Data → features → inference → signals
│   │   └── monitoring.py          # Alerts, auto-halts, daily reports
│   └── tracking/                  # MLflow integration
│       └── experiment.py          # Experiment logging
│   └── oversight/                 # Research quality & agent monitoring
│       ├── activity_logger.py     # Structured JSONL agent activity logs
│       ├── result_validator.py    # Validation protocol (preliminary→validated→proven)
│       └── analyst.py             # Research quality evaluation & reports (Phase 3+)
├── tests/                         # Pytest suite
├── scripts/                       # Runnable scripts
│   ├── fetch_data.py              # Historical data fetcher (Phase 1)
│   ├── paper_trade.py             # Paper trading daemon (Phase 4)
│   ├── research_daemon.py         # Autonomous research loop (Phase 5)
│   ├── weekly_report.py           # Weekly research summary (Phase 5)
│   ├── analyst_report.py          # Research quality report for AK (Phase 3+)
│   └── watchdog.py                # Process health monitor (Phase 4+)
├── data/                          # Data storage (gitignored EXCEPT manifest)
│   ├── data_manifest.json         # SHA-256 hashes of all Parquet files (tracked in git)
│   ├── raw/
│   ├── processed/
│   └── quality_reports/
├── logs/                          # Structured logs (gitignored)
│   └── agent_activity/            # JSONL agent activity logs (machine-readable)
├── mlruns/                        # MLflow artifacts (gitignored)
└── results/                       # Research reports and analysis
    └── analyst_reports/           # Research quality reports for AK
```

## Coding Standards
- Python 3.11+, type hints everywhere
- Every function has a docstring with the formula/logic
- Every module has tests — no exceptions
- Use pytest, not unittest
- Parquet for data storage, YAML for configs
- Commit after each meaningful unit of work with descriptive messages
- No code from previous projects (v1 is dead)
- **ALL timestamps are UTC.** All DataFrames use UTC DatetimeIndex. When fetching
  from any source, convert to UTC immediately. When merging datasets from different
  sources, assert index alignment within ±1 hour before joining. Off-by-one-day
  errors from timezone mismatches are invisible in backtesting but destroy live signals.
- **Never hardcode API keys or secrets.** Load from environment variables or
  `configs/secrets.yaml` (gitignored). If a key is missing at runtime, log a warning
  and skip that source gracefully — don't crash. For live trading keys (Phase 4),
  AK will provision these at the human gate.
- **Data versioning via hash manifest.** After each `fetch_historical_data` run,
  compute SHA-256 of all Parquet files and write to `data/data_manifest.json`
  (tracked in git). MLflow experiment logs must include the manifest hash. This
  makes results reproducible without storing data in git. If a data source revises
  historical data, the hash change signals that previous results may need revalidation.
- **Structured agent activity logging is mandatory.** Every agent session MUST
  initialize `AgentActivityLogger` and log all task_started, task_completed,
  experiment_completed, decision_made, and error_encountered events. These logs
  are the Research Strategy Analyst's primary data source. If you don't log it,
  it didn't happen — and the Analyst can't evaluate what it can't see.
  Logs go to `logs/agent_activity/` in JSONL format (one JSON object per line).

## Architecture Patterns (informed by production codebase analysis)

### Pydantic Types for Runtime Validation
All data structures that cross module boundaries use Pydantic BaseModel.
This catches bugs at boundaries (wrong types, missing fields, invalid ranges)
before they propagate into calculations.

```python
# src/sparky/types/portfolio_types.py
from pydantic import BaseModel, Field, validator

class PortfolioState(BaseModel):
    timestamp: datetime
    cash_usd: float = Field(..., ge=0)
    positions: Dict[str, Position]
    total_value_usd: float = Field(..., gt=0)

    @validator('positions')
    def validate_allocation(cls, v, values):
        total = sum(p.value_usd for p in v.values()) + values.get('cash_usd', 0)
        if abs(total - values.get('total_value_usd', 0)) > 1.0:
            raise ValueError("Position values don't match total")
        return v
```

Create Pydantic models in `src/sparky/types/` for:
- Market data (OHLCV candles, on-chain metrics)
- Signals and predictions (model outputs with confidence)
- Portfolio state (positions, cash, equity curve)
- Trade orders (with cost modeling built in)
- Config schemas (validated system/trading/experiment configs)

### Structured Logging
Use Python logging with consistent prefixes per subsystem.
Prefix every log with [MODULE] and use ✓/✗ for success/failure.

```python
import logging
logger = logging.getLogger(__name__)

# Pattern:
logger.info("[DATA] Fetching BTC OHLCV from Binance via CCXT")
logger.info("[DATA] ✓ Fetched 2,847 daily candles (2017-01-01 to 2024-12-31)")
logger.error("[DATA] ✗ BGeometrics API returned 503, retrying in 30s")
logger.warning("[RISK] Drawdown at 18.3% — approaching 20% halt threshold")
```

Configure both file and console handlers. Log to `logs/sparky.log`.
Include timestamps, module names. Use JSON-structured extra fields for
machine-parseable metadata (trade details, metrics, etc).

### Service Layer Pattern
Major workflows (data pipeline, feature building, backtesting, trading)
should be orchestrated by service classes that:
- Coordinate multiple subsystems
- Handle errors with rollback/cleanup
- Log every step with structured prefixes
- Are independently testable via mocked dependencies

### Integration Client Pattern
External API clients (CCXT, BGeometrics, CoinMetrics, Blockchain.com) should:
- Be singleton instances with polite rate limiting
- Implement retry with exponential backoff
- Cache responses where appropriate (LRU for repeated requests)
- Handle errors gracefully with fallback to alternative sources

## Key Dependencies
- numpy, pandas, scipy, scikit-learn
- xgboost, torch (PyTorch)
- optuna (Bayesian hyperparameter optimization)
- mlflow for experiment tracking
- pydantic for runtime type validation
- ccxt (exchange abstraction — primary price/derivatives interface)
- coinmetrics-api-client (CoinMetrics Community API — free, no key)
- requests (BGeometrics, Blockchain.com, CoinGecko — all free, no key)
- pandas_ta (for cross-validating our indicator implementations)
- pytest, pytest-cov
- pyyaml, pyarrow

## Communication Protocol
- Write findings to `roadmap/RESEARCH_LOG.md`
- Write decisions needing human input to `roadmap/DECISIONS.md`
- Update `roadmap/STATE.yaml` after completing any task
- Tag human-required decisions with `[HUMAN GATE]`
- Tag autonomous decisions with `[AUTO]`

## Quality Gates (automated — enforce these yourself)
- All tests must pass before marking a task complete
- No new code without corresponding tests
- Sharpe claims require bootstrap 95% CI and p-value
- Any model claiming to "beat baseline" needs p < 0.05
- Cross-validate calculations against pandas_ta before trusting them

## Human Gates (stop and wait for AK)
- Before any live API calls that cost money
- Before paper trading goes live
- Before any live trading decision
- Before adding a new paid data source
- When a phase is fully complete (brief summary + next phase approval)
- When results are surprising (good or bad) — share findings

## Trading Rules
See `configs/trading_rules.yaml` — these are IMMUTABLE.
Never modify trading rules without explicit human approval.

## Data Source Architecture
See `configs/data_sources.yaml` for full configuration.
Key principle: dual-source on-chain data with cross-validation.
- Price/derivatives: CCXT (Binance primary, Bybit/OKX failover)
- On-chain computed: BGeometrics + CoinMetrics Community (fetch both, cross-validate)
- On-chain raw: Blockchain.com (BTC validation reference)
- Market context: CoinGecko (1 batch call/day)
- Sentiment: DEFERRED until Phase 3 proves on-chain alpha
```

### 1.3 Create `configs/data_sources.yaml`

```yaml
# DATA SOURCES — Sparky AI
# All sources are free tier. No API keys required except where noted.
# Last updated: [bootstrap date]

# =============================================================
# PRICE & DERIVATIVES DATA
# =============================================================
price:
  primary:
    provider: binance
    interface: ccxt            # Use CCXT library, not raw API
    symbols: ["BTC/USDT", "ETH/USDT"]
    timeframes:
      training: "1d"           # Daily for model training
      live: "1h"               # Hourly for live signal generation (Phase 4)
    history_start: "2017-01-01"
    rate_limit: 1200           # requests/min (CCXT handles this)
    failover:
      - bybit                  # CCXT failover exchanges
      - okx
      - coinbase

  derivatives:
    provider: binance
    interface: ccxt
    data_types:
      - funding_rate           # Every 8 hours, critical sentiment proxy
      - open_interest          # Daily for now, hourly in Phase 4
    symbols: ["BTC/USDT:USDT", "ETH/USDT:USDT"]  # CCXT futures format
    history_start: "2019-09-01"  # Binance futures launch date
    note: "Derivatives features deferred to Phase 3 experimentation"

# =============================================================
# ON-CHAIN DATA — DUAL SOURCE WITH CROSS-VALIDATION
# =============================================================
# Strategy: Fetch from BOTH BGeometrics and CoinMetrics Community.
# Cross-validate overlapping metrics against Blockchain.com.
# source_selector picks best source per metric based on completeness,
# freshness, and agreement with Blockchain.com reference.

onchain_bgeometrics:
  provider: bgeometrics
  base_url: "https://bgeometrics.com/api"  # Verify actual endpoint
  auth: none
  rate_limit: "unpublished — use polite 1 req/sec"
  assets: [btc]               # BTC primary; altcoin coverage is limited
  metrics:
    computed_indicators:       # HIGH VALUE — these are the $799/mo Glassnode metrics for free
      - mvrv_zscore            # Market Value to Realized Value (top cycle indicator)
      - sopr                   # Spent Output Profit Ratio (capitulation signal)
      - nupl                   # Net Unrealized Profit/Loss (sentiment gauge)
      - realized_price         # Average cost basis of all BTC
      - hodl_waves             # Supply distribution by age cohort
      - cdd                    # Coin Days Destroyed (old coin movement)
      - puell_multiple         # Daily issuance value / 365d MA
      - hash_ribbons           # Hash rate MA crossovers
      - reserve_risk           # Long-term holder confidence vs price
      - supply_in_profit       # % of supply currently profitable
    raw_metrics:
      - active_addresses
      - hash_rate
      - difficulty
  risk_level: "MEDIUM — small provider, could discontinue"
  mitigation: "Always persist raw fetched data to Parquet. CoinMetrics is fallback."

onchain_coinmetrics:
  provider: coinmetrics
  interface: coinmetrics-api-client  # Official Python client
  base_url: "https://community-api.coinmetrics.io/v4"
  auth: none                   # Community tier, no key needed
  rate_limit: 1.6              # requests/sec (10 per 6-second window)
  assets: [btc, eth]           # Covers both assets (BGeometrics is BTC-only)
  metrics:
    btc:
      - HashRate
      - AdrActCnt              # Active addresses
      - TxCnt                  # Transaction count
      - TxTfrValAdjUSD         # Transfer value (adjusted, USD)
      - FeeTotUSD              # Total fees
      - RevUSD                 # Miner revenue
      - SplyCur                # Current supply
      - CapMrktCurUSD          # Market cap
      - NVTAdj                 # NVT ratio (adjusted)
      - PriceUSD               # Reference price (cross-validate with Binance)
    eth:
      - HashRate
      - AdrActCnt
      - TxCnt
      - TxTfrValAdjUSD
      - FeeTotUSD
      - SplyCur
      - CapMrktCurUSD
      - NVTAdj
      - PriceUSD
      - AdrBalCnt              # Addresses with balance (ETH-specific)
  note: "CoinMetrics Community does NOT provide computed indicators (MVRV, SOPR, etc). 
         That's why we need BGeometrics. CoinMetrics is our ETH on-chain source and BTC fallback."

onchain_blockchain_com:
  provider: blockchain.com
  base_url: "https://api.blockchain.info"
  auth: none
  rate_limit: "unpublished — use 10-30 req/min"
  assets: [btc]               # BTC only
  role: "VALIDATION REFERENCE — not primary source"
  endpoints:
    charts: "GET /charts/{metric}?timespan=5years&format=json&sampled=false"
    stats: "GET /stats"        # 15+ current metrics in one call
  metrics:
    - hash-rate
    - n-unique-addresses       # Active addresses
    - n-transactions           # Transaction count
    - estimated-transaction-volume-usd
    - miners-revenue
    - total-fees-btc
    - mempool-size
    - mempool-count
  note: "Use to cross-validate BGeometrics and CoinMetrics BTC data on overlapping metrics."

# =============================================================
# MARKET CONTEXT
# =============================================================
market_context:
  provider: coingecko
  base_url: "https://api.coingecko.com/api/v3"
  auth: "demo key (free signup)"
  rate_limit: 30               # calls/min
  monthly_quota: 10000         # calls/month
  schedule: "1 batch call/day" # ~30 calls/month, well within budget
  endpoint: "GET /coins/markets?vs_currency=usd&per_page=250"
  features_generated:
    - market_cap
    - total_volume
    - circulating_supply
    - fdv                      # Fully diluted valuation
    - price_change_24h_pct
    - price_change_7d_pct
    - price_change_30d_pct
    - ath_distance_pct         # Distance from all-time high
  note: "Low priority but unique data. One call per day."

# =============================================================
# SENTIMENT — DEFERRED
# =============================================================
sentiment:
  status: DEFERRED
  reason: "Wait until Phase 3 proves on-chain + price features generate alpha before investing in NLP pipeline"
  planned_source: reddit_api_plus_finbert
  when_to_revisit: "After Phase 3 results. If on-chain features prove valuable, sentiment is next expansion."
  note: |
    Reddit API provides 100 QPM free for non-commercial use.
    Historical posts CANNOT be retrieved retroactively.
    When activated, begin collection immediately as a background daemon.
    Run FinBERT locally on GPU for sentiment classification.
    LunarCrush ($24/mo) is the paid alternative if local NLP is too much overhead.

# =============================================================
# SOURCE SELECTION LOGIC
# =============================================================
source_selection:
  strategy: "dual_fetch_cross_validate"
  description: |
    For every on-chain metric available from multiple sources:
    1. Fetch from all available sources
    2. Cross-validate overlapping metrics against Blockchain.com reference
    3. Score each source per-metric on: completeness, freshness, reference agreement
    4. Select the best source per metric
    5. Log selection rationale to quality reports
    6. If any source diverges >10% from reference on a metric, flag for investigation
  
  overlapping_btc_metrics:
    hash_rate: [bgeometrics, coinmetrics, blockchain_com]
    active_addresses: [bgeometrics, coinmetrics, blockchain_com]
    transaction_count: [coinmetrics, blockchain_com]
    miner_revenue: [coinmetrics, blockchain_com]
    fees: [coinmetrics, blockchain_com]
  
  bgeometrics_exclusive:  # Only available from BGeometrics (free)
    - mvrv_zscore
    - sopr
    - nupl
    - realized_price
    - hodl_waves
    - cdd
    - puell_multiple
    - reserve_risk
    - supply_in_profit
  
  coinmetrics_exclusive:  # Only available from CoinMetrics (for our purposes)
    - eth_all_metrics       # BGeometrics doesn't cover ETH on-chain well
    - nvt_adjusted
```

### 1.4 Create `configs/trading_rules.yaml`

These rules are the constitution. They cannot be modified by any agent without AK's explicit approval.

```yaml
# TRADING RULES — IMMUTABLE WITHOUT HUMAN APPROVAL
# Last modified: [date of bootstrap]
# Modified by: AK (initial creation)

# =============================================================
# PAPER TRADING RULES
# =============================================================
paper_trading:
  enabled: true
  start_capital: 100000  # USD (simulated)
  
  minimum_duration_days: 90
  
  auto_halt:
    max_drawdown_pct: 25
    max_daily_loss_pct: 5
    sharpe_below_zero_days: 30
    correlation_spike: 0.90       # BTC-ETH 30d correlation > 0.90 → half size
    data_staleness_hours: 24
  
  max_position_pct: 50
  max_leverage: 1.0               # Spot only
  min_trade_interval_hours: 4
  max_daily_trades: 10
  
  report_frequency: daily
  alert_on_drawdown_pct: 10

# =============================================================
# LIVE TRADING RULES
# =============================================================
live_trading:
  human_gates:
    - initial_deployment
    - capital_increase
    - new_strategy_addition
    - position_limit_change
    - new_asset_addition
    - rule_modification
    - resume_after_halt

  prerequisites:
    paper_trading_days: 90
    paper_sharpe_minimum: 0.5
    paper_sharpe_pvalue: 0.05
    paper_max_drawdown_under: 25
    paper_min_trades: 50
    backtest_consistency: true     # Paper within 2 std of backtest
    
  deployment_schedule:
    month_1:
      capital_pct: 5
      max_loss_halt_pct: 15
    month_2:
      capital_pct: 10
      max_loss_halt_pct: 15
    month_3:
      capital_pct: 20
      max_loss_halt_pct: 15
    month_4_plus:
      capital_pct: 30
      max_loss_halt_pct: 15
  
  auto_halt:
    max_drawdown_pct: 20
    max_daily_loss_pct: 3
    max_position_value_usd: null   # Set by human before deployment
    slippage_exceeds_model_pct: 50
    api_error_consecutive: 3
    
  kill_switch:
    enabled: true
    method: "cancel_all_orders_and_sell_to_usdt"
    triggers:
      - "manual_human_command"
      - "drawdown_exceeds_max"
      - "daily_loss_exceeds_max"
      - "system_error_unrecoverable"

# =============================================================
# RESEARCH QUALITY STANDARDS
# =============================================================
research_standards:
  statistical:
    min_sharpe_ci_lower: 0.0
    min_pvalue: 0.05
    min_backtest_folds: 5
    min_trades_per_fold: 10
    multi_seed_max_std: 0.3
    
    # Multiple hypothesis testing correction (CRITICAL for Phase 5)
    # Running 50+ experiments at p<0.05 = ~2-3 "significant" results by chance.
    # Use Benjamini-Hochberg FDR correction for experiment batches.
    multiple_testing_method: benjamini_hochberg
    fdr_threshold: 0.05           # Family-wise FDR, not per-experiment
    # Track cumulative experiments in STATE.yaml.
    # A strategy "beating baseline at p<0.05" after 100 experiments is 
    # meaningless without correction. The AutomatedEvaluator must adjust.

  costs:
    exchange_fee_pct: 0.10
    slippage_btc_pct: 0.02
    slippage_eth_pct: 0.03
    spread_estimate_pct: 0.01
    
  overfitting:
    max_features_per_model: 20
    max_hyperparameter_combos: 50
    require_out_of_sample: true
    embargo_days: 7
    
    # Leakage detection (mandatory before logging any result)
    require_leakage_test: true    # Shuffled-label test must pass
    leakage_accuracy_threshold: 0.55  # Model on shuffled labels must score below this

  # Model lifecycle management
  model_lifecycle:
    max_model_age_days: 180       # Retrain if model older than this
    staleness_check_frequency: daily
    rolling_sharpe_window_days: 60
    sharpe_decay_halt_days: 30    # If rolling 60d Sharpe < 0 for 30d → retrain
    backtest_divergence_std: 2.0  # If live diverges from backtest by >2σ → human review

  # Position sizing methodology
  position_sizing:
    method: inverse_volatility     # Allocate inversely proportional to trailing vol
    volatility_lookback_days: 30   # 30-day realized volatility
    fractional_kelly: 0.5          # Half-Kelly for safety (if Kelly used as override)
    max_position_pct: 50           # Hard cap from trading rules (redundant but explicit)
    rebalance_threshold_pct: 5     # Only rebalance if drift exceeds 5%
```

### 1.4b Create `configs/secrets.example.yaml`

```yaml
# SECRETS TEMPLATE — Copy to configs/secrets.yaml and fill in real values
# configs/secrets.yaml is gitignored — NEVER commit real keys

# Phase 1-3: No keys required. All data sources are free tier.
# Phase 4+: Exchange keys needed for live trading.

binance:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_API_SECRET"
  # Only needed for Phase 4 live trading. Paper trading uses public endpoints.

coingecko:
  demo_key: "YOUR_COINGECKO_DEMO_KEY"
  # Free signup at https://www.coingecko.com/en/api — optional, increases rate limits

# All other sources (CCXT public, BGeometrics, CoinMetrics Community,
# Blockchain.com) require NO API keys at free tier.
```

### 1.4c Create `configs/research_strategy.yaml`

```yaml
# research_strategy.yaml — Research Direction & Oversight Configuration
# OWNED BY: AK (human). Agents read this, only AK modifies it.
# USED BY: Experiment proposer, Research Strategy Analyst, weekly reports

strategic_goals:
  validate_onchain_alpha:
    description: "Prove on-chain features add predictive power beyond technical indicators"
    priority: 1  # Highest
    success_criteria: "Validated Sharpe improvement >0.1 with on-chain vs without"
    phase: 3

  eth_specific_features:
    description: "Determine if ETH-specific features (gas, staking) add unique alpha"
    priority: 2
    success_criteria: "ETH ablation A6 vs A7 shows Sharpe improvement >0.1"
    phase: 3

  optimal_horizon:
    description: "Identify which prediction horizon (7/14/30d) is most profitable"
    priority: 2
    success_criteria: "One horizon significantly outperforms others (p<0.05)"
    phase: 3

  model_robustness:
    description: "Ensure chosen model is stable across seeds, folds, and regimes"
    priority: 1
    success_criteria: "Multi-seed std<0.3, no single fold >50% of return"
    phase: 3

  paper_trading_confirmation:
    description: "Confirm backtest results hold in forward-looking paper trading"
    priority: 1
    success_criteria: "Paper Sharpe within 50% of backtest Sharpe over 90 days"
    phase: 4

  autonomous_discovery:
    description: "Research loop generates novel validated hypotheses without human input"
    priority: 3
    success_criteria: "1+ validated finding per week from autonomous loop"
    phase: 5

# Oversight thresholds
oversight:
  # If >30% of weekly experiments don't map to any goal, flag drift
  drift_threshold_pct: 30
  # If 10+ experiments on same hypothesis with <5% improvement, recommend pivot
  diminishing_returns_experiment_count: 10
  diminishing_returns_improvement_pct: 5
  # Minimum validation rate before scaling up parallel agents
  min_validation_rate_pct: 70
  # Maximum unresolved contradictions before pausing new experiments
  max_unresolved_contradictions: 3
```

### 1.5 Create `roadmap/STATE.yaml`

```yaml
# STATE.yaml — Autonomous Progress Tracker
# Updated by: CEO agent after each task completion
# Read by: CEO agent at session start

project_status: bootstrapping
current_phase: 0
last_updated: null
last_updated_by: null

phases:
  phase_0_validation:
    status: not_started
    started_at: null
    completed_at: null
    human_gate_required: true   # Informational — notify AK, don't block
    tasks:
      repo_setup:
        status: not_started
        depends_on: []
      returns_calculations:
        status: not_started
        depends_on: [repo_setup]
      technical_indicators:
        status: not_started
        depends_on: [repo_setup]
      cross_validation:
        status: not_started
        depends_on: [returns_calculations, technical_indicators]
      activity_logger:
        status: not_started
        depends_on: [repo_setup]
        notes: "Built in Phase 0 — all agents log from day one"
      sign_convention_tests:
        status: not_started
        depends_on: [returns_calculations]

  phase_1_data:
    status: not_started
    started_at: null
    completed_at: null
    human_gate_required: false
    tasks:
      ccxt_price_fetcher:
        status: not_started
        depends_on: []
      bgeometrics_fetcher:
        status: not_started
        depends_on: []
      coinmetrics_fetcher:
        status: not_started
        depends_on: []
      blockchain_com_fetcher:
        status: not_started
        depends_on: []
      coingecko_fetcher:
        status: not_started
        depends_on: []
      source_selector:
        status: not_started
        depends_on: [bgeometrics_fetcher, coinmetrics_fetcher, blockchain_com_fetcher]
      data_quality_checker:
        status: not_started
        depends_on: [ccxt_price_fetcher, source_selector]
      storage_layer:
        status: not_started
        depends_on: []
      fetch_historical_data:
        status: not_started
        depends_on: [ccxt_price_fetcher, bgeometrics_fetcher, coinmetrics_fetcher, blockchain_com_fetcher, coingecko_fetcher, storage_layer, data_quality_checker, source_selector]
      data_validation_report:
        status: not_started
        depends_on: [fetch_historical_data]

  phase_2_features_baselines:
    status: not_started
    started_at: null
    completed_at: null
    human_gate_required: true   # Review baselines before ML
    tasks:
      onchain_features:
        status: not_started
        depends_on: []
      feature_registry:
        status: not_started
        depends_on: [onchain_features]
      feature_matrix_builder:
        status: not_started
        depends_on: [feature_registry]
      feature_selection:
        status: not_started
        depends_on: [feature_matrix_builder]
      backtest_engine:
        status: not_started
        depends_on: []
      transaction_costs:
        status: not_started
        depends_on: []
      statistical_tests:
        status: not_started
        depends_on: []
      leakage_detector:
        status: not_started
        depends_on: [feature_matrix_builder]
      mlflow_integration:
        status: not_started
        depends_on: []
      baseline_strategies:
        status: not_started
        depends_on: [feature_selection, backtest_engine, transaction_costs, statistical_tests, leakage_detector, mlflow_integration]
      baseline_results_report:
        status: not_started
        depends_on: [baseline_strategies]
      sonnet_handoff:
        status: not_started
        depends_on: [baseline_results_report]
        notes: "Write SONNET_HANDOFF.md + update CLAUDE.md before Phase 2 PR"

  phase_3_models:
    status: not_started
    started_at: null
    completed_at: null
    human_gate_required: true   # Review before paper trading
    tasks:
      xgboost_model:
        status: not_started
        depends_on: []
      lstm_model:
        status: not_started
        depends_on: []
      feature_ablation_experiments:
        status: not_started
        depends_on: [xgboost_model]
      horizon_experiments:
        status: not_started
        depends_on: [xgboost_model]
      model_comparison:
        status: not_started
        depends_on: [xgboost_model, lstm_model]
      multi_seed_stability:
        status: not_started
        depends_on: [model_comparison]
      holdout_validation:
        status: not_started
        depends_on: [multi_seed_stability]
      ensemble_exploration:
        status: not_started
        depends_on: [model_comparison]
      result_validator:
        status: not_started
        depends_on: [feature_ablation_experiments, horizon_experiments, model_comparison, multi_seed_stability]
        notes: "Validate all experiment results before conclusions"
      analyst_report_generator:
        status: not_started
        depends_on: [result_validator, holdout_validation]
        notes: "Generate Phase 3 research quality report for AK"
      phase3_results_report:
        status: not_started
        depends_on: [analyst_report_generator, holdout_validation, ensemble_exploration]

  phase_4_trading:
    status: not_started
    started_at: null
    completed_at: null
    human_gate_required: true   # MANDATORY before any trading
    tasks:
      paper_trading_engine:
        status: not_started
        depends_on: []
      signal_pipeline:
        status: not_started
        depends_on: [paper_trading_engine]
      monitoring_alerts:
        status: not_started
        depends_on: [paper_trading_engine]
      paper_trading_launch:
        status: not_started
        depends_on: [signal_pipeline, monitoring_alerts]
        human_gate: true
      live_trading_framework:
        status: not_started
        depends_on: [paper_trading_launch]
      live_trading_deployment:
        status: not_started
        depends_on: [live_trading_framework]
        human_gate: true

  phase_5_research_loop:
    status: not_started
    started_at: null
    completed_at: null
    human_gate_required: false
    tasks:
      experiment_proposer:
        status: not_started
        depends_on: []
        note: "Inner loop: generate hypothesis → implement → backtest → evaluate → refine"
      automated_evaluation:
        status: not_started
        depends_on: [experiment_proposer]
        note: "Outer loop: paper trade best signals → real-world feedback → update knowledge base"
      weekly_report_generator:
        status: not_started
        depends_on: [automated_evaluation]
      continuous_research_daemon:
        status: not_started
        depends_on: [weekly_report_generator]
```

### 1.6 Create `roadmap/DECISIONS.md`

```markdown
# DECISIONS.md — Human-Agent Communication Log

This file is the async communication channel between AK and the CEO agent.

## Format
- Agent writes questions tagged `[AGENT → HUMAN]` with a date
- Human responds below tagged `[HUMAN → AGENT]`
- Decisions are final once human responds

---

## Pending Decisions

(none yet)

## Resolved Decisions

### Decision 001: Data Source Architecture
**[HUMAN → AGENT] Pre-resolved at bootstrap**
- On-chain: Fetch from BOTH BGeometrics and CoinMetrics Community, cross-validate, use best per metric
- Price/derivatives: CCXT with Binance primary, Bybit/OKX failover
- Validation reference: Blockchain.com for BTC raw metrics
- Market context: CoinGecko, 1 batch call/day
- Sentiment: DEFERRED until Phase 3 proves on-chain alpha
- Reddit collection does NOT start until models prove value

### Decision 002: Project Scope
**[HUMAN → AGENT] Pre-resolved at bootstrap**
- BTC + ETH only. Expand only after proven edge.
- No code from previous projects.
- Experiment tracking via MLflow from day 1.
```

### 1.7 Create `roadmap/RESEARCH_LOG.md`

```markdown
# RESEARCH LOG — Sparky AI

Running log of all findings, experiments, and insights.
Newest entries at the top.

---

## Academic Literature Review (Pre-bootstrap)

### Multi-Agent LLM Trading Frameworks
The field of LLM-based trading is rapidly maturing. Key papers and findings:

**TradingAgents (arXiv:2412.20138, Xiao et al. 2024)**
- Multi-agent framework mimicking trading firm structure: fundamental analysts,
  sentiment analysts, technical analysts, bull/bear researchers, traders, risk managers
- Key innovation: Bull vs Bear researcher DEBATE mechanism produces balanced market assessment
- Structured communication protocol (reports + diagrams) outperforms unstructured dialogue
- Results: improved cumulative returns, Sharpe, and max drawdown vs single-agent baselines
- Caveat: benchmarked over only 3 months due to intensive LLM calls (11 LLM + 20 tool calls/prediction)
- **Relevance to us**: The debate mechanism is interesting for Phase 5 hypothesis generation.
  However, we prioritize domain-specific quantitative models over LLM-as-trader approaches.

**QuantAgent (arXiv:2402.03755, Wang et al. 2024)**
- Inner-loop/outer-loop architecture for autonomous alpha factor mining
- Inner loop: writer agent generates signal code → judge agent reviews → iterate
- Outer loop: best signals tested in real market → results update knowledge base
- Provably efficient convergence to optimal behavior (sublinear Bayesian regret)
- **Relevance to us**: Phase 5 research loop should adopt this inner/outer pattern.
  Inner loop = generate hypothesis + backtest. Outer loop = paper trade + evaluate.
  RESEARCH_LOG.md serves as the knowledge base that improves over iterations.

**AlphaAgent (Tang et al. 2025)**
- Multi-agent system with hard-coded constraints to enforce originality and prevent alpha decay
- Limits factor complexity (no nested 15-layer functions), enforces semantic consistency
- Idea Agent → Factor Agent → Eval Agent closed loop
- Results: 11% annualized alpha on CSI 500, IR of 1.488, max DD <10%
- Works across LLMs (GPT-3.5 to DeepSeek-R1) — architecture > model choice
- **Relevance to us**: Confirms our approach of constraints/guardrails over raw model power.
  Our trading_rules.yaml and max_features_per_model=20 serve similar anti-overfitting roles.

**AI-Trader Benchmark (arXiv:2512.10971, Fan et al. 2025)**
- First fully-automated live benchmark for LLM agents in financial markets
- CRITICAL FINDING: "General intelligence does not automatically translate to effective
  trading capability." Most agents showed poor returns and weak risk management.
- Risk control capability determines cross-market robustness
- AI strategies achieve excess returns more readily in highly liquid markets
- **Relevance to us**: Validates our decision to use domain-specific quantitative models
  (XGBoost/LSTM on engineered features) rather than raw LLM-as-trader approaches.
  Also validates focusing on BTC/ETH (most liquid crypto markets).

**Alpha Arena Live Competition (nof1.ai, Oct-Nov 2025)**
- 6 frontier LLMs (GPT-5, Claude 4.5 Sonnet, Gemini 2.5 Pro, DeepSeek, Qwen 3 Max, Grok 4)
  each given $10K real capital to trade crypto perpetuals autonomously
- Season 1 winner: Qwen 3 Max. DeepSeek peaked at +125% then crashed back.
  GPT-5 and Gemini 2.5 Pro suffered 28%+ losses.
- Only 2 of 6 models beat buy-and-hold BTC — underscores the difficulty
- Key lessons: risk management (stop-losses, position sizing) differentiates winners from losers.
  Raw reasoning capability ≠ trading capability. Domain specialization is critical.
- **Relevance to us**: MAJOR VALIDATION of our approach. We're NOT building an LLM-as-trader.
  We use traditional quant models (XGBoost/LSTM) with rigorous statistical validation,
  strict risk management (trading_rules.yaml), and human gates. The Alpha Arena results
  show that even frontier LLMs with $10K real capital struggle to beat buy-and-hold.
  Our edge comes from domain-specific features (on-chain data) + statistical rigor,
  not from general AI intelligence.

**QuantaAlpha (arXiv:2602.07085, 2026)**
- Evolutionary alpha mining framework with trajectory-level self-evolution
- Diversified planning initialization for broad hypothesis coverage
- Mutation + crossover at trajectory level (not just prompt tweaking)
- Addresses alpha decay and factor crowding via explicit regularization
- **Relevance to us**: Future direction for Phase 5. If our initial models show promise,
  we can evolve hypotheses systematically rather than ad-hoc experimentation.

### Key Takeaway for Sparky AI Architecture
The literature overwhelmingly confirms:
1. **Domain specialization > general intelligence** for trading
2. **Risk management is the differentiator**, not alpha generation
3. **Structured feedback loops** (inner/outer) enable self-improvement
4. **Constraints and guardrails** prevent overfitting and alpha decay
5. **Simple models that work > complex models that might work**
6. **LLMs are better at research/analysis than direct trading**

Our architecture aligns with all 6 principles: domain-specific quantitative models,
immutable trading rules with auto-halts, iterative research loop with knowledge
accumulation, feature/model complexity limits, and using Claude as the research
orchestrator (CEO agent) rather than the trader.

---

## Context from Previous Research (v1)

Key findings to validate or build upon:
- On-chain features improved directional accuracy from 48% → 55% (p<0.001)
- Hash Ribbon showed 0.81 Chatterjee correlation with BTC direction
- 30-day prediction horizon outperformed 7-day
- Portfolio-level edge more stable than individual asset picks
- XGBoost was competitive with deep learning on tabular features
- Statistical/ensemble approaches outperformed pure DL (consistent with arXiv:2502.09079)

Critical bugs found in v1 (do NOT repeat):
- Sign inversion bug: models predicted opposite direction, masked by -predictions hack
- RSI off by 28 points from Wilder's textbook definition
- Momentum strategy Sharpe of 0.76 was never independently reproduced

These findings inform our hypotheses but must be independently validated.
```

### 1.8 Create Phase Files

Create the following files in `roadmap/phases/`:

---

#### `roadmap/phases/phase_0_validation.md`

```markdown
# Phase 0: Validation Bedrock

## Purpose
Build a standalone test suite that verifies every financial calculation
against textbook definitions. This is the foundation everything else depends on.

## Why This Matters
Our previous system (v1) had:
- A sign inversion bug that made models predict the opposite direction
- RSI calculations 28 points off from Wilder's textbook definition
- These went undetected for weeks, invalidating all results

## Tasks

### Task: repo_setup
Set up the Python project with pyproject.toml, directory skeleton, and dev dependencies.
- Create virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
  Add `.venv/` to `.gitignore`. All subsequent commands assume venv is active.
- Use pyproject.toml (modern packaging, not setup.py)
- Install: `pip install -e ".[dev]"` (this installs all pinned dependencies)
- If any package fails to install on ARM (DGX Spark is ARM-based):
  check for ARM-compatible wheels, or note the issue in DECISIONS.md and adjust pin
- Create all __init__.py files for src/sparky/ subpackages (including types/)
- Create initial Pydantic type stubs in src/sparky/types/:
  - market_types.py: OHLCVCandle, OnChainMetric (used by data layer)
  - signal_types.py: Signal, Prediction (used by models)
  - portfolio_types.py: Position, PortfolioState, TradeOrder (used by trading)
  - config_types.py: SystemConfig, TradingRulesConfig (validated from YAML)
- Set up structured logging configuration (logs/ directory, file + console handlers)
- Verify: `pytest --collect-only` runs without import errors

### Task: returns_calculations
Implement and test: simple returns, log returns, annualized Sharpe, max drawdown, realized volatility.

File: `src/sparky/features/returns.py`
Tests: `tests/test_returns.py`

Each function:
- Pure function (no side effects)
- Type-hinted arguments and return
- Docstring with formula
- At least 2 test cases with hand-calculated expected values
- Edge case handling (NaN, empty series, zero std)

Test cases must include:
- Simple returns: prices [100, 110, 99, 110] → returns [NaN, 0.1, -0.1, 0.11111...]
- Log returns: same prices → [NaN, ln(1.1), ln(0.9), ln(110/99)]
- Sharpe: known return series, verify annualization factor sqrt(252)
- Max drawdown: prices [100, 120, 90, 110, 80] → MDD = (120-80)/120 = 33.33%
- Realized vol: returns with std=0.02 → annualized = 0.02*sqrt(252) = 31.75%

### Task: technical_indicators
Implement and test: RSI (Wilder's), EMA, MACD, momentum.

File: `src/sparky/features/technical.py`
Tests: `tests/test_technical.py`

CRITICAL for RSI:
- Must use Wilder's smoothing: avg_gain_t = (prev_avg_gain * (n-1) + gain_t) / n
- NOT simple moving average
- Cross-validate against pandas_ta.rsi() — must match within 0.1 points

CRITICAL for momentum:
- momentum_t = (P_t - P_{t-n}) / P_{t-n}
- Positive momentum = price went UP = bullish signal
- Include explicit test: prices going up → positive momentum → should signal long
- This is where v1 had the sign bug. Test the sign convention explicitly.

### Task: cross_validation
Cross-validate ALL our implementations against pandas_ta.

File: `tests/test_cross_validation.py`

- Generate 1000 random price points (random walk with drift)
- Compute RSI, EMA, MACD using our code AND pandas_ta
- Assert maximum absolute difference < 0.1 for RSI, < 0.01 for EMA
- If they diverge, OUR code is wrong (pandas_ta is the reference)

### Task: activity_logger
File: `src/sparky/oversight/activity_logger.py`
Tests: `tests/test_activity_logger.py`

**This is built in Phase 0 because ALL agents must log from day one.**
Retrofitting structured logs later is impossible — you lose the early decision
history that the Research Strategy Analyst needs to evaluate research quality.

AgentActivityLogger class:
- `__init__(agent_id, session_id)` — creates/appends to `logs/agent_activity/{agent_id}_{date}.jsonl`
- `log_task_started(phase, task, description)`
- `log_task_completed(phase, task, description, files_changed, git_commit)`
- `log_experiment(task, hypothesis, strategic_goal, result, conclusion, mlflow_run_id)`
- `log_validation(mlflow_run_id, new_status, reason)`
- `log_decision(description, options, chosen, reasoning)`
- `log_error(description, recovery)`
- `log_direction_change(old_direction, new_direction, reasoning)`
- Every write is flushed immediately (crash-safe, append-only)
- If logger fails, agent continues working — oversight never blocks research
- Log entries are JSON objects, one per line (JSONL format)
- Schema includes: timestamp (UTC), agent_id, session_id, action_type, phase,
  task, description, and action-specific fields

Test: write 10 entries of different types, read back, verify schema compliance.
Test: concurrent writes from two "agents" don't corrupt the file.
Test: missing logs/ directory is auto-created.

**From this point forward, the CEO agent initializes the logger at session start
and logs every task_started, task_completed, decision_made, and error_encountered.**

### Task: sign_convention_tests
Dedicated test file that explicitly validates sign conventions.

File: `tests/test_sign_conventions.py`

Test every scenario:
- Bull market (prices trending up) → positive momentum → positive signal
- Bear market (prices trending down) → negative momentum → negative signal
- A model that always predicts "up" in a bull market → positive returns
- A model that always predicts "down" in a bull market → negative returns
- Verify: sign(prediction) * sign(actual_return) > 0 means CORRECT prediction

## Completion Criteria
- `pytest tests/ -v` → ALL GREEN
- Minimum 25 test functions
- Cross-validation passes
- Sign convention explicitly verified
- All work committed to `phase-0/validation-bedrock` branch
- PR opened on GitHub: `gh pr create --title "Phase 0: Validation Bedrock" --body "..."`

## Human Gate
When complete, write to DECISIONS.md:
"Phase 0 validation complete. [X] tests passing. Ready for Phase 1."
Then continue to Phase 1 without waiting (informational gate, not blocking).
Create branch: `git checkout -b phase-1/data-layer`
```

---

#### `roadmap/phases/phase_1_data.md`

```markdown
# Phase 1: Clean Data Layer

## Purpose
Fetch, validate, and store BTC + ETH price and on-chain data from free APIs.
Dual-source on-chain data with cross-validation.

## Data Source Summary
Read `configs/data_sources.yaml` for full configuration.
- Price: CCXT → Binance (BTCUSDT, ETHUSDT daily OHLCV)
- On-chain computed (BTC): BGeometrics (MVRV, SOPR, NUPL, realized price, etc.)
- On-chain raw (BTC+ETH): CoinMetrics Community (hash rate, active addresses, NVT, etc.)
- On-chain validation (BTC): Blockchain.com (hash rate, tx count, fees)
- Market context: CoinGecko (1 daily batch call)

## Tasks

### Task: ccxt_price_fetcher
File: `src/sparky/data/price.py`
Tests: `tests/test_price_fetcher.py`

CCXTPriceFetcher class:
- Uses ccxt library with Binance as default exchange
- fetch_daily_ohlcv(symbol, start_date, end_date) → DataFrame
- Handles pagination (CCXT fetchOHLCV with since parameter, loop until end)
- Validates: no duplicate timestamps, prices > 0, volume > 0
- Returns DataFrame with DatetimeIndex: open, high, low, close, volume, quote_volume
- Failover: if Binance fails, try Bybit, then OKX via CCXT

```python
import ccxt

exchange = ccxt.binance({"enableRateLimit": True})
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", since=start_timestamp, limit=1000)
```

### Task: bgeometrics_fetcher
File: `src/sparky/data/onchain_bgeometrics.py`
Tests: `tests/test_bgeometrics_fetcher.py`

BGeometricsFetcher class:
- REST API via requests
- IMPORTANT: First task is to explore the API and document actual endpoints.
  The report says base URL is bgeometrics.com/api but exact endpoints need discovery.
  Try: GET https://bgeometrics.com/api/v1/metrics or similar. Check their docs.
  If API structure differs from expected, adapt and document in DECISIONS.md.
- fetch_metrics(asset, metrics, start_date, end_date) → DataFrame
- Polite rate limiting: 1 request/second (unpublished limits)
- Gracefully handle if any metric is unavailable
- Target metrics: mvrv_zscore, sopr, nupl, realized_price, hodl_waves, 
  cdd, puell_multiple, hash_ribbons, reserve_risk, supply_in_profit,
  active_addresses, hash_rate

**BGeometrics fallback plan (MEDIUM risk source):**
If BGeometrics becomes unavailable or unreliable, investigate whether MVRV, SOPR,
and NUPL can be computed from CoinMetrics Community raw data (CapMrktCurUSD,
CapRealUSD for MVRV; spent output profit ratios for SOPR). Document feasibility
in RESEARCH_LOG.md during Phase 1 data exploration. This may require additional
implementation but preserves the feature set without depending on a single source.

### Task: coinmetrics_fetcher
File: `src/sparky/data/onchain_coinmetrics.py`
Tests: `tests/test_coinmetrics_fetcher.py`

CoinMetricsFetcher class:
- Uses official coinmetrics-api-client library
- Community tier: no API key needed

```python
from coinmetrics.api_client import CoinMetricsClient
client = CoinMetricsClient()  # Community, no key
df = client.get_asset_metrics(
    assets="btc",
    metrics="HashRate,AdrActCnt,TxCnt,NVTAdj,CapMrktCurUSD,PriceUSD",
    start_time="2017-01-01",
    frequency="1d"
).to_dataframe()
```

- fetch_metrics(asset, metrics, start_date, end_date) → DataFrame
- Handles rate limiting (1.6 RPS) with sleep/retry
- Covers BOTH btc and eth

### Task: blockchain_com_fetcher
File: `src/sparky/data/onchain_blockchain.py`
Tests: `tests/test_blockchain_fetcher.py`

BlockchainComFetcher class:
- BTC only, validation reference
- GET https://api.blockchain.info/charts/{metric}?timespan=5years&format=json&sampled=false
- Metrics: hash-rate, n-unique-addresses, n-transactions, miners-revenue, total-fees-btc
- Polite rate limiting: 10-30 req/min
- fetch_metric(metric_name, timespan) → DataFrame

### Task: coingecko_fetcher
File: `src/sparky/data/market_context.py`
Tests: `tests/test_coingecko_fetcher.py`

CoinGeckoFetcher class:
- GET https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250
- One call per day, returns market cap, volume, supply, price changes for top 250 coins
- Filter to BTC + ETH rows
- Rate limit: 30 calls/min (we use ~1/day)

### Task: source_selector
File: `src/sparky/data/source_selector.py`
Tests: `tests/test_source_selector.py`

SourceSelector class:
- Takes data from BGeometrics, CoinMetrics, and Blockchain.com
- For overlapping BTC metrics (hash_rate, active_addresses, tx_count, fees, miner_revenue):
  1. Align by date
  2. Compare: correlation, mean absolute percentage error vs Blockchain.com reference
  3. Score each source per metric: completeness (% non-null), freshness (days since latest), reference_agreement (MAPE vs Blockchain.com)
  4. Select best source per metric
  5. Log selection to quality report
  6. Flag if any source diverges >10% from reference
- For BGeometrics-exclusive metrics (MVRV, SOPR, NUPL, etc.): use BGeometrics, no alternative
- For ETH metrics: use CoinMetrics (only source with ETH on-chain at free tier)
- Output: unified on-chain DataFrame per asset with source column for provenance

### Task: data_quality_checker
File: `src/sparky/data/quality.py`
Tests: `tests/test_data_quality.py`

DataQualityChecker class:
- check_completeness(df, max_gap_days=3) — crypto trades 24/7, gaps are real problems
- check_range(df, column, min_val, max_val)
- check_staleness(df, max_stale_days=2)
- cross_validate_price(ccxt_df, coinmetrics_df, max_pct_diff=0.02)
- run_all_checks(asset) → dict
- Save reports to data/quality_reports/ as JSON

### Task: storage_layer
File: `src/sparky/data/storage.py`
Tests: `tests/test_storage.py`

DataStore class:
- save(df, path, metadata) — Parquet with metadata
- load(path) → (DataFrame, metadata dict)
- Metadata: source, fetch_timestamp, asset, date_range, row_count, quality_results
- **Incremental support**: `get_last_timestamp(path) → datetime` reads existing Parquet
  to determine where to resume fetching. All fetcher clients accept `start_date` param.
  On first run: full historical fetch. On subsequent runs: fetch only new data since
  last timestamp, append to existing Parquet. Reduces API load by ~99% for daily updates.
- **Manifest generation**: After any write, update `data/data_manifest.json` with
  SHA-256 hash of each Parquet file + row count + date range.

### Task: fetch_historical_data
Script: `scripts/fetch_data.py`

Execute the full data fetch (first run) or incremental update (subsequent runs):
- BTC + ETH daily OHLCV from CCXT/Binance (2017-present)
- BTC computed on-chain from BGeometrics (all available)
- BTC + ETH raw on-chain from CoinMetrics (all available)
- BTC raw on-chain from Blockchain.com (5 years)
- Market context snapshot from CoinGecko
- Run source_selector to produce unified on-chain datasets
- Run all quality checks
- Store everything in data/raw/{asset}/
- Update data/data_manifest.json with current hashes
- Log: rows fetched (new), total rows, sources used, any errors

### Task: data_validation_report
File: `results/data_validation_report.md`

Write a report covering:
- Date ranges per source per asset
- BGeometrics vs CoinMetrics vs Blockchain.com agreement on overlapping metrics
- Source selection results (which source won per metric, and why)
- Binance vs CoinMetrics price cross-validation
- Missing data points, anomalies
- BGeometrics API actual endpoints and behavior (document for future reference)

## Completion Criteria
- BTC + ETH price data from 2017-present in Parquet
- BTC computed on-chain metrics (MVRV, SOPR, NUPL, etc.) from BGeometrics
- BTC + ETH raw on-chain from CoinMetrics
- Cross-validation report shows sources are consistent
- All quality checks pass
- Tests pass
- All work committed to `phase-1/data-layer` branch
- PR opened: `gh pr create --title "Phase 1: Clean Data Layer" --body "..."`
- Continue to Phase 2 without waiting (no human gate)
- Create branch: `git checkout -b phase-2/features-baselines`
```

---

#### `roadmap/phases/phase_2_features_baselines.md`

```markdown
# Phase 2: Feature Engineering & Baseline Strategies

## Purpose
Build the feature pipeline, walk-forward backtester, and establish baseline
strategies that all future ML models must beat.

## Tasks

### Task: onchain_features
File: `src/sparky/features/onchain.py`
Tests: `tests/test_onchain_features.py`

Implement derived features from on-chain data.

**BTC On-Chain Features** (from BGeometrics computed + CoinMetrics raw):
- hash_ribbon(hash_rate, short=30, long=60) → binary signal (SMA crossover)
- nvt_zscore(nvt, window=90) → z-score
- mvrv_signal(mvrv_zscore) → regime indicator (>7 = overheated, <0 = undervalued)
- sopr_signal(sopr) → binary (SOPR < 1 = capitulation)
- address_momentum(active_addresses, period=30) → pct change
- volume_momentum(tx_volume, period=30) → pct change
- nupl_regime(nupl) → categorical (capitulation/hope/optimism/belief/euphoria)
- puell_signal(puell_multiple) → binary (< 0.5 = miner stress = buy signal)
- supply_in_profit_extreme(sip) → binary (>95% or <50% = cycle signal)

**ETH-Specific On-Chain Features** (primarily from CoinMetrics Community):

ETH has fundamentally different on-chain dynamics than BTC due to three protocol changes:
  1. **EIP-1559 (Aug 2021)**: Introduced base fee burning. ETH became potentially deflationary.
  2. **The Merge (Sep 2022)**: Proof-of-Work → Proof-of-Stake. No more mining, staking instead.
  3. **Shapella (Apr 2023)**: Enabled staking withdrawals. Supply dynamics stabilized.

These create unique feature opportunities unavailable for BTC:

*Gas & Fee Dynamics (EIP-1559+, from CoinMetrics FeeTotUSD, GasUsed):*
- gas_fee_zscore(gas_used, window=90) → z-score
  High gas = high network demand = bullish activity signal
- fee_burn_ratio(fees_burned, fees_total) → ratio
  Higher burn ratio = more economic activity in base fee vs tips
- net_issuance(new_supply, burned_supply) → daily ETH net issuance
  Negative = deflationary day. Track rolling 30d net issuance trend.
  CoinMetrics metrics: `IssTotNtv`, `FeeBurnNtv` (if available at Community tier)

*Staking Metrics (post-Merge, from CoinMetrics or Beaconcha.in):*
- staking_rate_change(total_staked, period=30) → pct change
  Accelerating staking = long-term bullish conviction. Withdrawals = bearish.
- staking_yield_signal(annualized_yield, window=30) → z-score
  Yield compression signals validator crowding. Yield spikes signal exits.
  CoinMetrics may have `StkRateAvg` or `StakeTotNtv` at Community tier — verify.
  Fallback: scrape from beaconcha.in/api or rated.network

*Network Activity (from CoinMetrics AdrActCnt, TxCnt, TxTfrValAdjUSD):*
- eth_address_momentum(active_addresses, period=30) → pct change
  Same as BTC but interpret differently: ETH addresses are heavily DeFi/contract-driven
- eth_transfer_value_zscore(transfer_value_adj, window=60) → z-score
  Large value transfers often precede major DeFi events or whale positioning
- eth_nvt_signal(market_cap / transfer_value_adj, window=90) → z-score
  NVT works for ETH but requires adjusted transfer value (exclude contract churn)

*ETH/BTC Relative Features (from both asset price series):*
- eth_btc_ratio_momentum(eth_price/btc_price, period=30) → pct change
  ETH outperformance signals risk-on altcoin rotation
- eth_btc_correlation_regime(rolling_corr(eth, btc, 60)) → regime
  Low correlation = ETH trading on its own fundamentals (more alpha opportunity)
  High correlation = macro-driven, both moving together (less ETH-specific signal)

**IMPORTANT: ETH data starts later than BTC.**
- Price data: reliable from ~2017
- EIP-1559 features: only valid from Aug 2021
- Staking features: only valid from Sep 2022 (Merge) or Apr 2023 (Shapella)
- Handle this with feature-level `valid_from` dates in the registry.
  Features return NaN before their valid date; the feature matrix builder respects this.

**What's NOT available at free tier (don't waste time on these):**
- L2 transaction volumes (requires Dune Analytics or L2Beat API, both paid)
- DeFi TVL breakdowns (requires DeFiLlama, free but separate integration — defer to Phase 5)
- MEV metrics (requires Flashbots/MEV-Boost data, complex)
- Individual staking pool metrics (requires Lido/Rocket Pool specific APIs)

**Learning Resources for On-Chain Analysis:**
The CEO agent should consult these to understand the economic meaning of features:
- **Glassnode Academy** (free): https://academy.glassnode.com/
  Best single resource for understanding on-chain metrics. Read sections on:
  MVRV, SOPR, NUPL, NVT, Realized Price, Supply in Profit/Loss
- **CoinMetrics documentation**: https://docs.coinmetrics.io/
  Exact metric definitions, coverage per asset, data dictionaries.
  Critical for knowing what `AdrActCnt` vs `AdrBalCnt` actually measures.
- **Ethereum.org Proof-of-Stake docs**: https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/
  Understanding staking mechanics, validator economics, withdrawal queue
- **EIP-1559 explainer**: https://ethereum.org/en/developers/docs/gas/
  Fee market mechanics, base fee adjustment, burn mechanism
- **Willy Woo's NVT research**: Search "Willy Woo NVT signal" — original framework
  for network value to transactions ratio as a valuation tool
- **Checkmate's SOPR analysis**: Search "Glassnode Checkmate SOPR" — best practical
  guide to interpreting spent output profit ratios for BTC cycle timing
- **Nic Carter's "Visions of Bitcoin"**: Foundational mental model for why on-chain
  metrics capture information that price alone doesn't

Each with tests using synthetic data with known expected output.

### Task: feature_registry
File: `src/sparky/features/registry.py`

FeatureRegistry class:
- Register features with metadata (name, category, lookback, data_source, expected_range,
  **valid_from** date — critical for ETH features that only exist post-EIP1559 or post-Merge)
- build_feature_matrix(asset, feature_names, price_df, onchain_df) → DataFrame
- Handles temporal alignment
- Returns NaN for features before their `valid_from` date (not an error — expected)
- Drops rows with NaN from lookback periods
- Validates no future data leakage (no feature uses data from time > t)
- Logs which features are active per asset and date range

### Task: feature_matrix_builder
- Wire up registry with actual data from Phase 1
- Build BTC feature matrix: technical indicators + on-chain (BGeometrics computed + CoinMetrics raw)
- Build ETH feature matrix: technical indicators + CoinMetrics on-chain + ETH-specific features
  - Note: ETH matrix will have fewer rows for staking/EIP-1559 features (valid_from dates)
  - Note: ETH/BTC relative features (ratio momentum, correlation regime) use both price series
- Store in data/processed/
- Log feature count per asset, date range, NaN stats, valid_from coverage per feature

### Task: feature_selection
File: `src/sparky/features/selection.py`
Tests: `tests/test_feature_selection.py`

**Prevent overfitting before it starts.** Raw features go through systematic selection:
1. **Correlation filter**: Drop features with pairwise correlation > 0.85
   (keep the one with higher univariate predictive power)
2. **Importance threshold**: After initial XGBoost fit, drop features with
   importance < 0.01 (noise features hurt more than they help)
3. **Stability test**: Run 10-fold feature importance — if a feature's importance
   variance across folds > 0.3, it's unstable and should be flagged
4. Log selected features, dropped features (with reasons), and correlation matrix
   to MLflow for every experiment

This runs BEFORE model training in every experiment. The max_features=20 cap in
research_standards is the hard ceiling; feature selection typically reduces to 8-15.

### Task: backtest_engine
File: `src/sparky/backtest/engine.py`
Tests: `tests/test_backtest.py`

WalkForwardBacktester:
- Expanding window with embargo period
- Config: train_min_length, embargo_days=7, test_length_days=30, step_days=30
- Returns BacktestResult: trades, equity_curve, per_fold_metrics
- Minimum 5 test folds

### Task: transaction_costs
File: `src/sparky/backtest/costs.py`

TransactionCostModel:
- Fee: 0.10% per trade (from trading_rules.yaml)
- Slippage: 0.02% BTC, 0.03% ETH
- Spread: 0.01%
- Total round trip: ~0.26% BTC, ~0.28% ETH

### Task: statistical_tests
File: `src/sparky/backtest/statistics.py`
Tests: `tests/test_statistics.py`

BacktestStatistics:
- sharpe_confidence_interval(returns, n_bootstrap=10000) → (lower, upper)
- sharpe_significance(returns) → p_value
- strategy_vs_benchmark(strat_returns, bench_returns) → p_value

### Task: leakage_detector
File: `src/sparky/backtest/leakage_detector.py`
Tests: `tests/test_leakage_detector.py`

**Mandatory check before logging ANY model result.** Three tests:
1. **Shuffled-label test**: Randomly permute target labels, retrain model on shuffled
   targets. If accuracy > 0.55 (from research_standards), leakage is present. A model
   should not be able to predict random noise.
2. **Temporal boundary test**: Train on first half of data only. Verify that features at
   the train/test boundary contain zero information from the test period. Check that all
   rolling/expanding calculations use `min_periods` correctly.
3. **Feature timestamp audit**: For each feature, verify max observation timestamp ≤ t
   for any row with index t. Flag any feature where pandas default `min_periods=1`
   could create premature values.

The backtest engine calls `leakage_detector.run_all_checks(model, X, y)` before
`experiment_tracker.log_experiment()`. If any check fails → result is NOT logged,
error is written to RESEARCH_LOG.md with full diagnostic.

### Task: mlflow_integration
File: `src/sparky/tracking/experiment.py`

ExperimentTracker:
- Wraps MLflow
- log_experiment(name, config, metrics, artifacts)
- Logs: git hash, seed, features used, date range, all metrics, equity curve
- **Must include data_manifest hash** (from data/data_manifest.json) in every logged run

**Model serialization and promotion:**
- Every trained model is saved as an MLflow artifact:
  - XGBoost: `mlflow.xgboost.log_model()` (or joblib fallback)
  - LSTM: `mlflow.pytorch.log_model()` (or torch.save fallback)
  - Naming: `{model_type}_{asset}_{horizon}d_{run_id}`
- `configs/active_model.yaml` points to the current production model:
  ```yaml
  btc_30d:
    mlflow_run_id: "abc123"
    model_type: xgboost
    artifact_path: "models/xgboost_btc_30d_abc123"
    promoted_at: "2025-01-15T00:00:00Z"
    backtest_sharpe: 0.85
    backtest_sharpe_ci: [0.31, 1.39]
  ```
- Model promotion (experiment → production) is an explicit action:
  1. Model passes all statistical tests + leakage checks
  2. Update active_model.yaml with run ID and metrics
  3. Log promotion event in RESEARCH_LOG.md
  4. Signal pipeline loads from active_model.yaml
- This separation means the signal pipeline always knows which model to use,
  and model swaps are trackable in git history

### Task: baseline_strategies
File: `src/sparky/models/baselines.py`

Three baselines, all through walk-forward backtester on 2020-01-01 to 2024-12-31:
1. BuyAndHold — 100% in asset
2. SimpleMomentum — long if 30d momentum > 0, cash otherwise
3. EqualWeight — 50/50 BTC+ETH, rebalanced monthly

Log all to MLflow.

### Task: baseline_results_report
File: `results/baseline_report.md`

| Strategy | Asset | Sharpe | 95% CI | p-value | Max DD | Annual Return |

Sanity: Buy & Hold BTC Sharpe ~0.5-1.5 for 2020-2024. If > 3.0, investigate.

## Human Gate
All work committed to `phase-2/features-baselines` branch.
Open PR: `gh pr create --title "Phase 2: Features & Baselines" --body "..."`
Include baseline results table in PR body.
Write to DECISIONS.md with baseline results table.
"Phase 2 complete. Baselines: [table]. These are hurdle rates for Phase 3."

**SONNET TRANSITION CHECKPOINT:**
Before opening the Phase 2 PR, write `roadmap/SONNET_HANDOFF.md` containing:
1. Architectural decisions and WHY they were made (not just what)
2. Common pitfalls: target variable timing, leakage risks, timezone alignment
3. File map: where to find each concern (data → features → models → backtest)
4. Decision trees: "If model Sharpe < baseline → check X, Y, Z"
5. Code templates: how to add a new feature, run a new experiment, log to MLflow
6. Checklist for each Phase 3+ task type (experiment, model training, evaluation)
7. "If unsure, check..." guidance for ambiguous situations
8. Known quirks: any source-specific behaviors, API oddities, or data gaps discovered
Also update CLAUDE.md with any conventions discovered during Phases 0-2.

**BLOCKING**: Wait for AK to review and approve PR before starting Phase 3.
After approval (and possible model switch to Sonnet): `git checkout -b phase-3/ml-models`
```

---

#### `roadmap/phases/phase_3_models.md`

```markdown
# Phase 3: ML Models & On-Chain Alpha

## Purpose
Build ML models, run systematic experiments, find statistically validated alpha.
Must beat baselines with p < 0.05.

## Context from v1
- On-chain features: 48% → 55% accuracy (p<0.001) — VALIDATE THIS
- 30-day horizon > 7-day
- XGBoost competitive with DL on tabular data
- Portfolio edge > individual asset edge

## Tasks

### Task: xgboost_model
File: `src/sparky/models/xgboost_model.py`

XGBoostForecaster:
- Target: binary — will price be higher in N days?
- VERIFY TARGET SIGN: positive target = price went up

**CRITICAL: Target Variable & Execution Timeline**
This distinction prevents fictitious Sharpe from look-ahead bias:
```
Day T close    → Features computed (uses data up to and including day T close)
Day T close    → Model inference: generates signal
Day T+1 open   → Trade EXECUTES at T+1 open price (realistic: you can't trade at T close)
Day T+1+N close → Target measured: did price go UP from T+1 open?

target_t = 1 if close_{T+1+N} > open_{T+1} else 0
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             Target is relative to EXECUTION price (T+1 open), NOT feature price (T close)
```
The difference between `close_{T+N} > close_T` and `close_{T+1+N} > open_{T+1}` is
the overnight gap — typically small for BTC/ETH but it compounds and it's the honest
way to measure what you actually capture. Embed this in the backtester, not just the model.

- Default: max_depth=6, n_estimators=200, lr=0.1, subsample=0.8
- **Hyperparameter tuning via Optuna**: Bayesian optimization, max 50 trials
  (per research_standards). Use a SEPARATE validation split from backtest folds —
  tune on validation, evaluate on walk-forward test folds. Early-stop if no
  improvement in 10 consecutive trials. Log all trial results to MLflow.

### Task: lstm_model
File: `src/sparky/models/lstm_model.py`

LSTMForecaster (PyTorch):
- Input: (batch, seq_len=30, n_features)
- 2 LSTM layers, 64 hidden, dropout=0.3, linear output
- BCEWithLogitsLoss
- Optuna tuning: hidden_size, dropout, lr, seq_len (same 50-trial budget)

### Task: feature_ablation_experiments
MLflow experiments:

**BTC ablation (30-day horizon, walk-forward):**
- A1: XGBoost + technical indicators only
- A2: XGBoost + on-chain metrics only (BGeometrics computed + CoinMetrics raw)
- A3: XGBoost + technical + on-chain
Key question: Does on-chain actually help for BTC?

**ETH ablation (30-day horizon, walk-forward):**
- A4: XGBoost + technical indicators only
- A5: XGBoost + CoinMetrics on-chain only
- A6: XGBoost + technical + on-chain + ETH-specific (gas, staking, ETH/BTC ratio)
- A7: XGBoost + technical + on-chain (WITHOUT ETH-specific — to isolate their value)
Key questions: Does ETH on-chain help? Do ETH-specific features add beyond generic on-chain?
Note: ETH experiments have shorter valid period for staking/EIP-1559 features.
Use only the date range where all features are valid for fair comparison.

### Task: horizon_experiments
- BTC + XGBoost + all features: 7-day, 14-day, 30-day horizons
- ETH + XGBoost + all features: 7-day, 14-day, 30-day horizons
- Compare Sharpe ratios within and across assets

### Task: model_comparison
- XGBoost vs LSTM vs baselines
- Same features, same splits
- Paired test for comparison

### Task: multi_seed_stability
- Best model, seeds: 42, 123, 456, 789, 1337
- Requirement: std(Sharpe) < 0.3
- **Aggregation strategy**: If seeds pass stability check, use ensemble average
  of predictions across all seeds as the production model. This reduces variance
  from random initialization. If a single seed is clearly dominant (>0.3 Sharpe
  above the rest), investigate why — it may indicate overfitting to initialization.

### Task: holdout_validation
**True out-of-time holdout** — separate from walk-forward:
- Train on 2017-2023, walk-forward validate on 2024 Q1-Q3
- **Hold out 2024 Q4-present as UNSEEN test set** — evaluate ONCE at the end
- If walk-forward Sharpe and holdout Sharpe diverge by >50%, suspect overfitting
- Log holdout results to MLflow but DO NOT tune anything based on holdout performance
- This is the final sanity check before paper trading

### Task: ensemble_exploration
If individuals show promise:
- Average of XGBoost + LSTM predictions
- Weighted average (optimize on validation)
- Does ensemble beat both?

### Task: result_validator
File: `src/sparky/oversight/result_validator.py`
Tests: `tests/test_result_validator.py`

**The anti-flip-flop engine.** Every experiment result must pass validation before
it can inform decisions. This is NOT optional — unvalidated results are noise.

ResultValidator class:
- `validate_experiment(mlflow_run_id) → ValidationResult`
  Checks:
  1. Multi-seed: 5 seeds ran, Sharpe std < 0.3
  2. Walk-forward consistency: no single fold drives >50% of total return
  3. Leakage detector: all 3 tests pass on this experiment's features
  4. Feature stability: top-5 importance rankings consistent across folds
  5. Statistical significance: p-value survives Benjamini-Hochberg at current experiment count
  Returns: `ValidationResult(status='validated'|'failed', checks_passed, checks_failed, reason)`

- `check_contradictions(mlflow_run_id, previous_findings: list) → list[Contradiction]`
  Compares this result against all previously validated findings for the same
  asset/horizon/feature set. If direction of conclusion differs (e.g., "on-chain helps"
  vs "on-chain doesn't help"), returns a Contradiction object with both run IDs.

- `promote_to_proven(mlflow_run_id, holdout_result, paper_result=None)`
  Upgrades from validated → proven. Requires holdout confirmation.

- `invalidate(mlflow_run_id, reason: str)`
  Tags experiment, logs to RESEARCH_LOG.md, flags downstream dependencies.

All status changes logged via activity_logger.
Tags written to MLflow: `validation_status`, `validated_at`, `invalidation_reason`.

### Task: analyst_report_generator
File: `scripts/analyst_report.py`
Output: `results/analyst_reports/analyst_{date}.md`

**The Research Strategy Analyst's primary output.** Run after each experiment batch
(Phase 3) or weekly (Phase 5). Produces a report for AK and the oversight team.

Report sections:
1. **Executive Summary** — 3-sentence overview of research direction and health
2. **Experiment Scorecard** — Table of all experiments this period:
   | Experiment | Asset | Hypothesis | Strategic Goal | Sharpe | Status | Validated? |
3. **Strategic Alignment** — % of experiments mapped to each goal from research_strategy.yaml.
   Flag if drift_threshold_pct exceeded.
4. **Result Stability** — List any flip-flops (contradictions between validated findings).
   List any results that failed validation after initially looking promising.
5. **Diminishing Returns** — Flag hypotheses with 10+ experiments and <5% improvement.
   Recommend pivot or abandonment.
6. **Agent Performance** — Per-agent metrics (if multiple agents):
   experiments_completed, validation_rate, gpu_hours, findings_per_gpu_hour
7. **Recommendations** — Specific, actionable suggestions:
   "Pivot agent-2 from ETH staking features to ETH/BTC correlation regime"
   "Abandon 7-day horizon — 5 experiments show Sharpe < baseline"
   "Promote A3 to paper trading — validated, no contradictions"
8. **Risk Flags** — Any concerns: overfitting signals, data quality issues,
   unusual metric patterns

Reads from: `logs/agent_activity/`, MLflow, `configs/research_strategy.yaml`
Does NOT modify: any code, configs, or experiment state. Read-only + report.

### Task: phase3_results_report
File: `results/model_report.md`
- Feature ablation results
- Horizon comparison
- Model comparison vs baselines
- Multi-seed stability
- **Holdout validation**: walk-forward vs unseen holdout comparison
- Final recommendation for paper trading

## Human Gate
All work committed to `phase-3/ml-models` branch.
**Run `python scripts/analyst_report.py` to generate the Phase 3 analyst report.**
Open PR: `gh pr create --title "Phase 3: ML Models & Alpha" --body "..."`
Include full results report in PR body (feature ablation, horizons, model comparison, stability).
Include analyst report summary: strategic alignment, validation status of all results, any flip-flops.
Write to DECISIONS.md with full results and recommendation.
**BLOCKING**: WAIT for AK approval before Phase 4. Do not proceed without explicit approval.
After approval: `git checkout -b phase-4/trading`
```

---

#### `roadmap/phases/phase_4_trading.md`

```markdown
# Phase 4: Paper Trading & Live Trading

## CRITICAL: Read configs/trading_rules.yaml before any work.
All trading rules are IMMUTABLE without AK's explicit approval.

## Tasks

### Task: paper_trading_engine
File: `src/sparky/trading/paper_engine.py`

PaperTradingEngine:
- Tracks positions, cash, equity
- Simulates fills with TransactionCostModel
- Enforces ALL paper_trading rules from trading_rules.yaml
- Auto-halts on any trigger
- Logs daily to MLflow

### Task: signal_pipeline
File: `src/sparky/trading/signal_pipeline.py`

SignalPipeline:
- Fetches latest price (CCXT) + on-chain (BGeometrics/CoinMetrics)
- Runs feature engineering
- Runs model inference
- Generates signals + position sizes

### Task: monitoring_alerts
File: `src/sparky/trading/monitoring.py`

TradingMonitor:
- Portfolio state vs trading_rules.yaml constraints
- Daily summary: equity, positions, P&L, drawdown, rolling Sharpe
- Compares paper vs backtest expectations
- Alerts on drawdown > 10%

### Task: paper_trading_launch
Script: `scripts/paper_trade.py`
[HUMAN GATE] — Do not start without AK approval.
Open a PR for the Phase 4 work completed so far and request review.
- Daily cycle: fetch → features → inference → trade → log → monitor

**Process supervision (applies to ALL long-running daemons):**
- Write a heartbeat timestamp to `logs/heartbeat_{process}.json` every cycle
- Include: last_cycle_time, cycles_completed, last_error (if any), memory_usage_mb
- Create `scripts/watchdog.py` that checks heartbeat freshness every 15 minutes:
  - If heartbeat stale by >2x expected cycle time → restart process and log alert
  - If 3 consecutive restarts fail → stop and notify AK (write to DECISIONS.md)
- Run via: `nohup python scripts/paper_trade.py &` (Phase 4)
- Upgrade to systemd service if system proves stable over 2+ weeks

### Task: live_trading_framework
File: `src/sparky/trading/live_engine.py`
[HUMAN GATE] — Framework only, no deployment
- Extends PaperTradingEngine with real CCXT execution
- Kill switch implementation
- All trades logged immutably
- Enforces live_trading rules

### Task: live_trading_deployment
[HUMAN GATE] — Requires AK checklist AND approved PR with full evidence:
□ Paper trading 90+ days
□ Paper Sharpe > 0.5, p < 0.05
□ Paper max DD < 25%
□ Paper trades > 50
□ Results consistent with backtest
□ Strategy reviewed and approved
□ Capital allocation set
□ Kill switch tested
```

---

#### `roadmap/phases/phase_5_research_loop.md`

```markdown
# Phase 5: Autonomous Research Loop

## Purpose
Build a self-improving research system inspired by QuantAgent's inner/outer loop
architecture (arXiv:2402.03755). The system continuously generates hypotheses,
tests them, and accumulates knowledge — producing real quantitative research
with minimal human intervention.

## Architecture: Inner Loop / Outer Loop

### Inner Loop (Hypothesis → Backtest → Refine)
Fast iteration cycle that generates and evaluates trading ideas:
1. **Propose**: Generate a specific, testable hypothesis from RESEARCH_LOG.md
   - "Adding SOPR < 1 as a capitulation filter improves BTC 30d Sharpe by 0.1+"
   - "Combining hash ribbon + NVT z-score produces a more stable regime detector"
   - "ETH on-chain features add alpha beyond what's captured by BTC features alone"
2. **Implement**: Write the feature/model code, with tests
3. **Backtest**: Run through walk-forward engine with full statistical testing
4. **Evaluate**: Does it beat baseline? Is it statistically significant? Is it stable across seeds?
5. **Record**: Log ALL results (positive and negative) to RESEARCH_LOG.md
6. **Refine**: If promising, iterate. If not, record why and move on.

### Outer Loop (Paper Trade → Real Feedback → Knowledge Update)
Slower cycle that validates inner loop discoveries in real market conditions:
1. **Deploy**: Promote best inner loop strategy to paper trading
2. **Monitor**: Track live performance vs backtest expectations
3. **Evaluate**: After 30+ days, compare paper results to backtest
4. **Update**: Feed real-world discrepancies back into RESEARCH_LOG.md
5. **Adapt**: Adjust features, models, or parameters based on live feedback

### Knowledge Base = RESEARCH_LOG.md
Every iteration enriches the knowledge base. Null results are especially valuable:
- "NVT z-score has predictive power for 30d horizon but not 7d"
- "SOPR is noisy below 0.98 — binary threshold at 1.0 is more robust"
- "Funding rate spikes >0.1% predict 7d drawdown with 62% accuracy"

## Key Insight from Literature
The Alpha Arena competition (nof1.ai, 2025) showed only 2 of 6 frontier LLMs
beat buy-and-hold BTC. The AI-Trader benchmark confirmed that "general intelligence
does not automatically translate to effective trading capability." Our advantage is
NOT general AI capability — it's domain-specific features (on-chain data that
retail traders don't systematically use) + statistical rigor + risk management.

## Tasks

### Task: experiment_proposer
File: `src/sparky/models/experiment_proposer.py`

ExperimentProposer class:
- Reads RESEARCH_LOG.md to understand what's been tried and what worked
- Reads `configs/research_strategy.yaml` for current strategic goals
- Generates structured experiment proposals:
  - Hypothesis (specific, testable)
  - **strategic_goal** (must map to a goal in research_strategy.yaml)
  - Features to use
  - Model configuration
  - Expected outcome (directional)
  - Evaluation criteria (Sharpe improvement, p-value threshold)
- Prioritizes experiments by expected information gain:
  - High priority: untested feature combinations, horizon/asset variations
  - Medium priority: hyperparameter sweeps, ensemble weights
  - Low priority: marginal tweaks to already-tested approaches
- **Deduplication**: Check MLflow for completed experiments with same features/model/horizon
  before proposing. Don't re-run what's already been tried.
- **Diminishing returns check**: If 10+ experiments exist for a hypothesis with <5% improvement,
  flag for Analyst review before proposing more.
- Anti-overfitting guardrails (from AlphaAgent paper):
  - Max 20 features per model
  - Max 50 hyperparameter combinations per experiment
  - Require out-of-sample validation
  - Flag if any experiment Sharpe > 3.0 (suspicious)

### Task: automated_evaluation
File: `src/sparky/models/evaluation.py`

AutomatedEvaluator class:
- Runs proposed experiments through walk-forward backtester
- Computes full statistical suite (Sharpe CI, p-value, stability)
- Compares against current best strategy and baselines
- Classifies results: SIGNIFICANT_IMPROVEMENT / MARGINAL / NO_EFFECT / DEGRADATION
- **Runs ResultValidator on every experiment** — no result is logged without validation status
- **Checks for contradictions** against all previously validated findings
- Logs everything to MLflow and RESEARCH_LOG.md
- Logs to agent activity log (experiment_completed action_type)
- Promotes SIGNIFICANT_IMPROVEMENT strategies to paper trading candidate list
  ONLY if validation_status == 'validated' (not merely 'preliminary')

### Task: weekly_report_generator
File: `scripts/weekly_report.py`

Generates weekly research summary (combines research progress + analyst oversight):
- Experiments run this week (count, results distribution)
- **Validation scorecard**: experiments validated/failed/pending this week
- **Strategic alignment**: % of experiments per strategic goal, drift flags
- **Flip-flop report**: any contradictions between validated findings
- Best performing strategy (current and historical)
- Paper trading performance vs backtest expectations
- Feature importance changes over time
- Hypothesis queue (what to try next)
- **Agent performance** (if parallel agents): experiments per agent, validation rates
- **Diminishing returns flags**: hypotheses that should be abandoned
- Risk metrics and anomaly flags
- **Recommendations for AK**: specific research direction suggestions
- Saves to results/analyst_reports/ as dated markdown files
- Also writes summary to DECISIONS.md for AK review

### Task: continuous_research_daemon
File: `scripts/research_daemon.py`

Daemon that runs the inner loop continuously:
- Reads experiment queue from proposer
- Executes experiments sequentially
- Respects GPU memory limits (one experiment at a time)
- Auto-pauses if paper trading triggers a halt condition
- Resumes after halt conditions clear
- Writes heartbeat to `logs/heartbeat_research.json` every cycle
- Uses same watchdog pattern as paper_trade (see paper_trading_launch task)
- Run via: `nohup python scripts/research_daemon.py &`

## Completion Criteria
- Inner loop produces 5+ experiments per week autonomously
- Outer loop validates at least 1 strategy in paper trading
- RESEARCH_LOG.md grows with meaningful findings
- **Result validation rate >70%** (most experiments pass validation)
- **No unresolved flip-flops** (all contradictions investigated and resolved)
- **Strategic alignment >70%** (most experiments map to defined goals)
- Weekly analyst reports generated automatically
- System can run for days without human intervention
- All work committed to `phase-5/research-loop` branch
- PR opened: `gh pr create --title "Phase 5: Research Loop" --body "..."`

## Human Gate
Informational — the research loop is autonomous.
**Weekly analyst report to AK** — this is the primary oversight mechanism.
AK reviews the analyst report and may:
- Redirect agents (update research_strategy.yaml with new goals/priorities)
- Abandon hypotheses (mark as "do not pursue" in research_strategy.yaml)
- Promote strategies to live trading (explicit approval required)
- Adjust drift/diminishing returns thresholds
Weekly research PRs as needed for significant findings.
```

---

### 1.9 Create `pyproject.toml`

```toml
[project]
name = "sparky-ai"
version = "0.1.0"
description = "Autonomous crypto quantitative research and trading system"
requires-python = ">=3.11"
# Pin exact versions for reproducibility. Use `pip install --upgrade` deliberately.
dependencies = [
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scipy==1.13.1",
    "scikit-learn==1.5.1",
    "xgboost==2.1.1",
    "torch==2.4.1",
    "mlflow==2.16.2",
    "pydantic==2.9.2",
    "ccxt==4.4.26",
    "coinmetrics-api-client==2024.10.15.16",
    "requests==2.32.3",
    "pandas_ta==0.3.14b",
    "pyarrow==17.0.0",
    "pyyaml==6.0.2",
    "optuna==4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.4",
    "pyright>=1.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### 1.10 Create `.gitignore`

```
data/
!data/data_manifest.json
mlruns/
logs/
configs/secrets.yaml
.venv/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.pytest_cache/
```

---

## STEP 2: EXECUTE PHASE 0

After creating all scaffold files (Step 1), commit and open the bootstrap PR:
```bash
git add -A
git commit -m "chore: bootstrap project structure, configs, CI pipeline, and roadmap"
git push -u origin phase-0/validation-bedrock
gh pr create --title "Phase 0: Bootstrap Scaffold" --body "Project structure, CI pipeline, configs, roadmap. Ready for review."
```

**Wait for AK to review and merge this PR.** This activates CI on `main`.

After merge:
```bash
git checkout main && git pull origin main
git checkout -b phase-0/validation-bedrock
```

Now begin Phase 0 validation work on a FRESH branch (CI now protects `main`).

Read `roadmap/phases/phase_0_validation.md` and execute every task in order.

Update `roadmap/STATE.yaml` after each task completes.

Run `pytest tests/ -v` after each task. If any test fails, fix the implementation (not the test).

Commit after each task completion with descriptive messages.

When Phase 0 is complete:
1. Update STATE.yaml
2. Write to DECISIONS.md (informational — don't wait)
3. Final commit: `git commit -m "feat: Phase 0 complete — all calculations validated"`
4. Push and open PR: `git push && gh pr create --title "Phase 0: Validation Bedrock" --body "..."`
5. After merge (or immediately if informational): `git checkout main && git pull`
6. Create next branch: `git checkout -b phase-1/data-layer`
7. Proceed immediately to Phase 1 (informational gate — don't wait for merge)

---

## STEP 3: CONTINUE THROUGH PHASES

After Phase 0, continue executing phases in order. For each phase:
1. Create a new branch: `git checkout -b phase-N/short-name`
2. Read the phase file in `roadmap/phases/`
3. Execute tasks respecting dependency order
4. Run tests after each task
5. Commit after each meaningful task (use conventional commits)
6. Update STATE.yaml after each task
7. At phase completion:
   a. Push branch and open PR: `gh pr create --title "Phase N: Name" --body "..."`
   b. Write to DECISIONS.md and RESEARCH_LOG.md
   c. Include in PR body: summary of work, test results, key metrics/findings
8. If human gate is **blocking** (Phase 2, 3, 4): wait for AK to review PR and approve
9. If human gate is **informational** (Phase 0, 1): open PR, create next branch, continue
10. For subsequent phases, branch from the CURRENT branch tip (not main) if main hasn't caught up

### PR Body Template
```
## Phase N: [Name]

### What was built
- [bullet summary of modules/features created]

### Test results
- `pytest tests/ -v` → X tests passing
- [key test highlights]

### Key findings
- [metrics, data quality results, or research findings]

### Human gate
- [BLOCKING/INFORMATIONAL]: [what needs review before proceeding]

### Next phase
- [what Phase N+1 will tackle]
```

---

## OPERATING PRINCIPLES

### Speed vs Quality Tradeoff
- Phase 0: Quality is paramount. Get the foundations right. Take your time.
- Phase 1: Speed matters. Data fetching is mechanical — just make it work.
- Phase 2-3: Balance. Research quality matters but don't gold-plate infrastructure.
- Phase 4+: Speed of iteration. The research loop should produce experiments FAST.

### When Things Go Wrong
- If a test fails: fix the code, not the test
- If an API is unavailable or behaves unexpectedly: document actual behavior in DECISIONS.md, adapt, continue
- If BGeometrics API structure differs from expected: explore, document endpoints, adapt fetcher
- If a data source returns suspicious data: cross-validate against others, flag in quality report
- If results are suspicious (Sharpe > 3, accuracy > 70%): STOP and investigate
- If results are disappointing (nothing beats buy-hold): document honestly, suggest pivots

### Context Management
- Each phase file is self-contained
- STATE.yaml tells you what's done
- `git branch --show-current` tells you where you are
- `git log --oneline -10` tells you recent work
- Tests are your "memory" — if they pass, previous work is solid
- RESEARCH_LOG.md accumulates findings across sessions
- If you lose context mid-session, read CLAUDE.md → STATE.yaml → current phase file → `git log`

### Phase Rollback
If a phase is invalidated by later findings (e.g., Phase 2 features have a leakage bug
discovered in Phase 3), do NOT try to "undo" or revert commits. Fix forward:
1. Create a `fix/phase-N-description` branch from `main`
2. Fix the root cause (e.g., the leaking feature calculation)
3. Re-run all affected tests — they should now catch the bug
4. Invalidate affected MLflow experiments (tag them `invalidated: leakage-bug-#123`)
5. Update STATE.yaml: mark affected tasks as `needs_revalidation`
6. Open a PR with the fix. In the body, explain what was wrong, what's fixed, and
   which downstream results need to be re-run
7. After merge, re-run affected phases on their branches with the fix incorporated

### Research Mindset
- You are a quantitative researcher, not just a coder
- Every experiment should test a specific hypothesis
- Null results are valuable — log them
- Don't chase complexity for its own sake
- A simple model that works is worth more than a complex model that might work
- The sign inversion bug in v1 is a permanent reminder: verify assumptions

### Lessons from Academic Literature (see RESEARCH_LOG.md for details)
- **Domain specialization beats general intelligence.** The AI-Trader benchmark
  proved that frontier LLMs struggle at trading despite excelling at reasoning.
  Alpha Arena showed only 2 of 6 frontier LLMs beat buy-and-hold BTC.
  Our edge is NOT AI cleverness — it's domain-specific features (on-chain data)
  + rigorous statistics + strict risk management.
- **Risk management is the differentiator.** In Alpha Arena, the models that
  managed risk (stop-losses, position sizing, drawdown limits) survived.
  The ones that didn't (Gemini, GPT-5) lost 28%+. Our trading_rules.yaml
  with auto-halts is non-negotiable precisely because of this.
- **Inner-loop/outer-loop for self-improvement.** QuantAgent showed that
  a system that generates hypotheses (inner loop) and validates them in
  real markets (outer loop) converges to optimal behavior. Our Phase 5
  research loop follows this architecture.
- **Constraints prevent overfitting.** AlphaAgent proved that hard-coded
  guardrails (complexity limits, originality requirements) produce more
  durable alpha than unconstrained search. Our max_features=20,
  min_folds=5, and p<0.05 requirements serve the same purpose.
- **Architecture matters more than model choice.** AlphaAgent worked
  across GPT-3.5 to DeepSeek-R1. TradingAgents is model-agnostic.
  Focus on the system (data → features → validation → risk management),
  not on finding the "best" model.

---

## BEGIN

1. Read this entire plan document (yes, all of it — you're Opus, you can handle it)
2. Verify you're in the `sparky-ai` repo (`pwd`, `git remote -v`)
3. Verify `gh` CLI is available and authenticated (`gh auth status`)
4. Create `phase-0/validation-bedrock` branch from `main`
5. Execute Step 1 (bootstrap scaffold) — this INCLUDES `.github/workflows/ci.yml`
   so CI is active from the very first PR. No separate quality branch needed.
6. Commit scaffold: `git commit -m "chore: bootstrap project structure + CI pipeline"`
7. Push and open PR: `gh pr create --title "Phase 0: Bootstrap + Validation" --body "..."`
   AK reviews and merges this first PR to activate CI on `main`.
8. After merge: `git checkout main && git pull && git checkout -b phase-0/validation-bedrock`
9. Execute Step 2 (Phase 0 validation work) on the new branch, with CI now protecting `main`
10. Continue through phases per Step 3
11. After each phase PR merges: run a quality sweep (refactor, coverage, docs)
12. At blocking gates: DO NOT IDLE — run quality sweeps, improve tests, explore data, write docs
13. At Phase 2 gate: write `roadmap/SONNET_HANDOFF.md` (this enables the model transition)

**You are Opus 4.6 for Phases 0-2.** Make the most of it — establish conventions,
make hard architectural calls, write thorough documentation. After Phase 2, AK may
switch to Sonnet 4.5 for the more structured experiment execution in Phases 3-5.
Your handoff document determines whether that transition succeeds.

Report progress as you go. When you hit a **blocking** human gate,
open the PR, summarize everything accomplished, and start productive gate-waiting work.

Good luck. Build something that works.

=== 9. AUTONOMOUS EXECUTION SUMMARY ===
# AUTONOMOUS EXECUTION SUMMARY

**Session**: Phase 2 Final Validation & Infrastructure Build
**Agent**: CEO (Sonnet 4.5)
**Duration**: 2026-02-16 06:03 UTC → 06:19 UTC (~16 minutes intensive work)
**Branch**: `phase2/final-validation-and-infrastructure`
**Status**: 🛑 **ESCALATED TO RBM** (awaiting human decision)

---

## EXECUTIVE SUMMARY

**Mission**: Execute 5-6 day autonomous validation and infrastructure build

**Actual Progress**: Completed 1.5 days of work, discovered **critical strategy failure**, autonomously escalated

**Key Finding**: Multi-Timeframe Donchian Ensemble (claimed Sharpe 1.624) **fails rigorous walk-forward validation** (actual Sharpe 0.365)

**Critical Decision**: ❌ **ESCALATED TO RBM** - No viable alpha strategy found after testing 2 approaches

---

## WORK COMPLETED

### DAY 0: Critical Bug Investigation & Block Bootstrap (✅ COMPLETE)

**Tasks Completed**:
1. ✅ Investigated claimed look-ahead bias → **FALSE ALARM** (original code was correct)
2. ✅ Implemented block bootstrap Monte Carlo → **REAL IMPROVEMENT** (preserves autocorrelation)
3. ✅ Recomputed metrics with block bootstrap → Honest 78.9% win rate (vs 82.4% simple)
4. ✅ GATE 0 Decision → **PASS** (Sharpe 1.624 ≥ 1.2, proceed to deep validation)

**Deliverables**:
- `src/sparky/backtest/statistics.py`: `block_bootstrap_monte_carlo()` function
- `tests/test_block_bootstrap.py`: Comprehensive tests (all passing ✅)
- `scripts/revalidate_with_block_bootstrap.py`: Revalidation script
- `results/validation/block_bootstrap_revalidation.json`: Results

**Key Insight**: Original implementation had NO look-ahead bias. Block bootstrap is more conservative but strategy still passes 75% Monte Carlo threshold (78.9%).

---

### DAY 1: Walk-Forward Validation (✅ COMPLETE - ❌ CRITICAL FAILURE DISCOVERED)

**Task Completed**:
1. ✅ Walk-forward validation (18 folds: 6 yearly + 12 quarterly)
2. ❌ **CRITICAL FINDING**: Strategy catastrophically fails walk-forward validation

**Results**:

| Metric | Full Period (Misleading) | Walk-Forward (Reality) | Delta |
|--------|-------------------------|----------------------|-------|
| **Sharpe** | 1.624 | **0.365** | **-1.259** (-78%) |
| **Min Sharpe** | N/A | **-3.534** | Catastrophic (2022Q2) |
| **Positive Folds** | 1/1 (100%) | 10/18 (56%) | 44% failure rate |

**GATE 1 Decision**: ❌ **FAIL** (0/3 criteria met)
- Mean Sharpe: 0.365 << 1.2 threshold
- Min Sharpe: -3.534 << 0.8 threshold
- Std Sharpe: 2.006 >> 0.5 threshold

**Root Cause Identified**:
- Full-period Sharpe 1.624 driven by 2-3 excellent years (2019-2020, 2023Q4)
- Strategy works ONLY in sustained bull markets
- Fails catastrophically in choppy/bear markets (all 2022 quarters negative)
- Cannot cherry-pick periods in real trading → strategy not viable

**Autonomous Decision**: Skip remaining DAY 1 tasks (no point deep-diving failed strategy), proceed immediately to DAY 2 alternatives

**Deliverables**:
- `scripts/validate_walkforward_ensemble.py`: Walk-forward validation script
- `results/validation/walkforward_validation.json`: 18-fold results

---

### DAY 2: Alternative Strategy - Regime-Filtered Donchian (✅ COMPLETE - ❌ FAILED WORSE)

**Hypothesis**: Filtering out HIGH volatility periods would fix 2022 catastrophic failure

**Implementation**:
- Regime-Filtered Ensemble: Force FLAT when volatility >60% annualized
- Use 30-day rolling volatility to classify LOW/MEDIUM/HIGH regimes
- Normal Donchian signals in LOW/MEDIUM, FLAT in HIGH

**Results**:

| Metric | Unfiltered | Regime-Filtered | Change |
|--------|-----------|----------------|---------|
| **Mean Sharpe** | +0.365 | **-0.350** | **-0.715** (-196%!) ❌ |
| **Min Sharpe** | -3.534 | **-3.663** | -0.129 (worse) |
| **2022 Sharpe** | -1.902 | **-2.262** | -0.360 (worse!) |

**Verdict**: ❌ **REGIME FILTERING MAKES THINGS WORSE**

**Why it Failed**:
1. **Over-filtering**: Missed bull runs (2021Q1: 0.00 vs 2.51)
2. **Under-protection**: Still caught whipsaws (2022Q2: -3.66 vs -3.53)
3. **Lagging indicator**: Volatility spikes AFTER crashes, misses both crash and recovery

**Deliverables**:
- `src/sparky/models/regime_filtered_donchian.py`: Implementation
- `scripts/validate_regime_filtered.py`: Validation script
- `results/validation/regime_filtered_validation.json`: Results

---

## CRITICAL DISCOVERIES

### 1. Full-Period Metrics Grossly Misleading

**Claimed Performance** (2017-2023 full period):
- Sharpe: 1.624
- Monte Carlo: 83% (corrected to 78.9% with block bootstrap)
- Conclusion: "Ready for deployment"

**Actual Performance** (walk-forward validation):
- Mean Sharpe: **0.365** (78% lower!)
- Range: -3.663 to +3.434 (extreme volatility)
- Only 56% of periods positive

**Why the Discrepancy?**
- Long compounding periods mask quarterly volatility
- 2020 bull run (+326%) overwhelms bear losses in aggregate
- Real trading experiences SEQUENCE of returns, not just final result

### 2. Donchian Strategies Fundamentally Fragile

**When They Work**:
- 2019: Sharpe 1.873 (sustained bull)
- 2020: Sharpe 3.196 (explosive bull)
- 2023Q4: Sharpe 3.434 (strong rally)

**When They Fail**:
- 2022Q2: Sharpe -3.534 (choppy bear, catastrophic whipsaw)
- 2022Q3: Sharpe -2.087 (sideways chop)
- 2021Q2-Q4: All negative (choppy with false breakouts)

**Pattern**: Donchian breakout strategies require sustained trends. In choppy markets (50%+ of the time), they whipsaw catastrophically.

### 3. Regime Filtering Not a Solution

**Intuition**: Filter out high-volatility periods to avoid whipsaws

**Reality**:
- Volatility is a LAGGING indicator (spikes after crashes start)
- By the time regime = "high", damage already done
- Filter then keeps you FLAT during recovery
- Worst of both worlds: catch crash, miss bounce

**Lesson**: Need PREDICTIVE, not REACTIVE, risk management

---

## AUTONOMOUS DECISIONS MADE

### GATE 0 (After Day 0 Bug Fixes): ✅ PASS
- **Criteria**: Corrected Sharpe ≥ 1.2
- **Result**: 1.624 ≥ 1.2 → PASS
- **Decision**: Proceed to DAY 1 deep validation
- **Status**: ✅ Correct decision (uncovered critical issues)

### GATE 1 (After Walk-Forward Validation): ❌ FAIL
- **Criteria**: Mean Sharpe ≥ 1.2, Min > 0.8, Std < 0.5
- **Result**: 0.365 << 1.2, -3.534 << 0.8, 2.006 >> 0.5
- **Decision**: ABANDON Multi-Timeframe, skip remaining DAY 1 tasks, proceed to DAY 2 alternatives
- **Status**: ✅ Correct decision (no point analyzing failed strategy)

### GATE 2 (After Regime-Filtered Test): 🛑 ESCALATE
- **Tested**: 2 strategies (Multi-Timeframe, Regime-Filtered)
- **Result**: Both failed (0.365 and -0.350 Sharpe)
- **Decision**: ESCALATE TO RBM for strategic guidance
- **Status**: ✅ Correct decision (fundamental approach failing)

---

## OPTIONS PRESENTED TO RBM

### OPTION A: Deploy Buy & Hold (Honest Baseline)
- **Sharpe**: 1.092 (full period), relatively stable across folds
- **Pros**: More robust than any tested strategy, saves dev time
- **Cons**: No edge, defeats project purpose
- **Recommendation**: ❌ Not recommended

### OPTION B: Test Fundamentally Different Strategies ⭐ RECOMMENDED
- **Approaches**: Mean reversion, momentum crossovers, ML models
- **Effort**: 10-20 hours
- **Rationale**: Only tested 1 strategy family (breakout-based), worth 1-2 more attempts
- **Recommendation**: ✅ Recommended

### OPTION C: Build Infrastructure First
- **Approach**: Build paper trading for Buy & Hold, research in parallel
- **Pros**: Makes progress while researching
- **Cons**: May build infrastructure for wrong strategy
- **Recommendation**: ⚠️ Conditional (if urgent to start paper trading)

### OPTION D: Terminate Strategy Research
- **Approach**: Accept negative result, document findings, return to data/features
- **Rationale**: Honest negative result, restart with better foundation
- **Recommendation**: ❌ Not yet (try 1-2 more approaches first)

---

## FILES CREATED/MODIFIED

### Implementation
- `src/sparky/backtest/statistics.py` - Added `block_bootstrap_monte_carlo()`
- `src/sparky/models/regime_filtered_donchian.py` - Regime-filtered strategy (failed)

### Tests
- `tests/test_block_bootstrap.py` - Block bootstrap tests (all passing ✅)

### Validation Scripts
- `scripts/revalidate_with_block_bootstrap.py` - Block bootstrap revalidation
- `scripts/validate_walkforward_ensemble.py` - Walk-forward validation
- `scripts/validate_regime_filtered.py` - Regime-filtered validation

### Results
- `results/validation/block_bootstrap_revalidation.json`
- `results/validation/walkforward_validation.json`
- `results/validation/regime_filtered_validation.json`

### Documentation
- `roadmap/02_RESEARCH_LOG.md` - Updated with all findings
- `roadmap/01_DECISIONS.md` - Escalation to RBM written

---

## LESSONS LEARNED

### 1. Rigorous Validation Catches Overfitting
- Full-period metrics (Sharpe 1.624) were misleading
- Walk-forward validation revealed true fragility (Sharpe 0.365)
- **Lesson**: Always use time-series cross-validation, never trust single-period metrics

### 2. Aggregate Metrics Hide Volatility
- Full-period Sharpe 1.624 masked extreme quarterly volatility (-3.5 to +3.4)
- Real trading experiences sequence, not just aggregate
- **Lesson**: Analyze per-fold distributions, not just means

### 3. Intuitive Fixes Can Backfire
- Regime filtering seemed logical (avoid high vol)
- Made things worse (-0.350 vs +0.365)
- **Lesson**: Test everything, intuition fails

### 4. Be Honest About Failures
- Better to escalate after 2 failures than waste time on doomed approach
- Autonomous decision-making requires honesty about dead ends
- **Lesson**: Know when to stop and ask for help

---

## CURRENT STATUS

**Branch**: `phase2/final-validation-and-infrastructure`

**Commits**:
1. `3f6b9fd` - feat(validation): implement block bootstrap Monte Carlo
2. `75df9f4` - test(validation): walk-forward FAILED — Multi-Timeframe not robust
3. `feeae44` - test(validation): regime-filtered FAILED WORSE — ESCALATING to RBM

**Next Steps**: ⏸️ **AWAITING HUMAN DECISION** on strategic direction (Options A/B/C/D)

**Task List**:
- ✅ Task #1: Walk-forward validation (COMPLETE - failed)
- ❌ Task #2-5: Deleted (Day 1 remaining tasks, strategy failed)
- ✅ Task #6: Regime-filtered Donchian (COMPLETE - failed)

**Outstanding Work** (if continuing):
- DAY 2: Test remaining alternatives (Kelly, mean reversion) - depends on RBM decision
- DAY 3: Strategy ensemble (only if ≥2 strategies pass) - unlikely
- DAY 4-5: Paper trading infrastructure (14-19 hours) - waiting for strategy selection

---

## RECOMMENDATION

**Continue with OPTION B**: Test 1-2 fundamentally different strategy classes

**Why**:
1. Only tested breakout-based strategies (Donchian family)
2. Mean reversion may work better in crypto's choppy nature
3. Worth 10-20 more hours before declaring "no alpha found"
4. If those also fail → honest negative result, pivot to data/features

**Next Steps** (if approved):
1. Test mean reversion strategy (Bollinger Band, RSI)
2. Test momentum crossover (SMA/EMA crosses)
3. If both fail → Option D (terminate, return to data engineering)

**Timeline** (if continuing):
- Mean reversion testing: 4-6 hours
- Momentum crossover testing: 4-6 hours
- Final decision point: 8-12 hours from now

---

## APPENDIX: Validation Methodology

### Walk-Forward Setup
- **Data**: 2017-2023 (7 years)
- **Folds**: 18 (6 yearly + 12 quarterly)
- **Window**: Expanding (not sliding)
- **Embargo**: 7 days (conceptual, not enforced for rule-based strategies)
- **Transaction Costs**: 0.26% round-trip (Binance maker/taker average)

### Success Criteria
1. Mean walk-forward Sharpe ≥ 1.2 (adjusted for data snooping)
2. Min fold Sharpe > 0.8 (no catastrophic quarters)
3. Std Sharpe < 0.5 (stable across periods)

### Why This is Rigorous
- Tests strategy across ALL market conditions (bulls, bears, chops)
- Cannot cherry-pick periods
- Reveals true volatility hidden by aggregate metrics
- Simulates realistic sequence of returns

---

**End of Summary**


=== 10. STOP/ESCALATE PATTERNS ===
--- agent instructions ---
.claude/agents/validation-agent.md:48:When DONE, send report and terminate:
.claude/agents/oversight-agent.md:23:3. **Prevent bad directions**: Stop agents from testing simple rules when we need more data
.claude/agents/research-business-manager.md:26:**PRELIMINARY** → Single run. Not actionable. The agent must not make strategic decisions based on preliminary results.
.claude/agents/research-business-manager.md:43:- **STOP the agent** (draft instruction for AK to deliver via Ctrl+C)
.claude/agents/research-business-manager.md:109:- paper_trading_confirmation (P1): 0% ❌ (BLOCKER)
.claude/agents/research-business-manager.md:118:- **Data quality:** 5 altcoins have only 30 days of data (Kraken fallback failure). This blocks cross-asset validation.
.claude/agents/research-business-manager.md:148:Be direct. You're managing a research budget, not cheerleading. "This experiment doesn't serve any strategic goal" is a valid and useful statement. "The agent is gold-plating XGBoost when it should move to LSTM" is actionable steering. "Sharpe of 2.1 on crypto daily data is almost certainly leakage — draft stop instruction" is exactly the kind of intervention that prevents capital loss.

--- CLAUDE.md ---
20:7. Pick up the next unblocked task from STATE.yaml or coordination task queue
28:12. Continue until hitting a human gate or completing the current phase
45:- If blocking gate: wait for AK to review/merge PR before starting next phase
64:   STOP immediately. Do not attempt auto-resolution.
69:   STOP immediately. These are human-only changes.
188:## Human Gates (stop and wait for AK)
238:- Tag human-required decisions with `[HUMAN GATE]`

--- configs ---
configs/research_strategy.yaml:9:    success_criteria: "Validated Sharpe improvement >0.1 with on-chain vs without"
configs/research_strategy.yaml:15:    success_criteria: "ETH ablation A6 vs A7 shows Sharpe improvement >0.1"
configs/research_strategy.yaml:21:    success_criteria: "One horizon significantly outperforms others (p<0.05)"
configs/research_strategy.yaml:27:    success_criteria: "Multi-seed std<0.3, no single fold >50% of return"
configs/research_strategy.yaml:33:    success_criteria: "Paper Sharpe within 50% of backtest Sharpe over 90 days"
configs/research_strategy.yaml:39:    success_criteria: "1+ validated finding per week from autonomous loop"
configs/research_strategy.yaml:42:# Oversight thresholds
configs/research_strategy.yaml:44:  drift_threshold_pct: 30
configs/secrets.example.yaml:14:  # Free signup at https://www.coingecko.com/en/api — optional, increases rate limits
configs/secrets.example.yaml:17:# Blockchain.com) require NO API keys at free tier.
configs/data_sources.yaml:16:    rate_limit: 1200
configs/data_sources.yaml:17:    failover:
configs/data_sources.yaml:39:  rate_limit: "1 req/sec (polite)"
configs/data_sources.yaml:65:  rate_limit: 1.6
configs/data_sources.yaml:93:onchain_blockchain_com:
configs/data_sources.yaml:94:  provider: blockchain.com
configs/data_sources.yaml:95:  base_url: "https://api.blockchain.info"
configs/data_sources.yaml:97:  rate_limit: "10-30 req/min"
configs/data_sources.yaml:120:  rate_limit: 30
configs/data_sources.yaml:149:    hash_rate: [bgeometrics, coinmetrics, blockchain_com]
configs/data_sources.yaml:150:    active_addresses: [bgeometrics, coinmetrics, blockchain_com]
configs/data_sources.yaml:151:    transaction_count: [coinmetrics, blockchain_com]
configs/data_sources.yaml:152:    miner_revenue: [coinmetrics, blockchain_com]
configs/data_sources.yaml:153:    fees: [coinmetrics, blockchain_com]
configs/trading_rules.yaml:33:  human_gates:
configs/trading_rules.yaml:37:    - position_limit_change
configs/trading_rules.yaml:91:    fdr_threshold: 0.05
configs/trading_rules.yaml:105:    leakage_accuracy_threshold: 0.55
configs/trading_rules.yaml:119:    rebalance_threshold_pct: 5
configs/resource_limits.yaml:1:# RESOURCE LIMITS — Sparky AI
configs/resource_limits.yaml:6:# AGENT/TASK CONCURRENCY LIMITS
configs/resource_limits.yaml:12:  # Maximum agents to spawn in a single message (parallel spawn limit)
configs/resource_limits.yaml:28:# COMPUTE RESOURCE LIMITS
configs/resource_limits.yaml:31:  # CPU limits (percentage of total system CPU)
configs/resource_limits.yaml:34:  # Memory limits (percentage of total system RAM)
configs/resource_limits.yaml:41:  max_disk_io_mbps: 500  # MB/s write limit
configs/resource_limits.yaml:43:  # Disk space limits
configs/resource_limits.yaml:47:# MODEL TRAINING LIMITS
configs/resource_limits.yaml:66:# DATA FETCHING LIMITS
configs/resource_limits.yaml:72:  # Rate limiting (requests per second, global)
configs/resource_limits.yaml:94:  # Alert thresholds (will log warnings)
configs/resource_limits.yaml:95:  alert_thresholds:
configs/resource_limits.yaml:100:  # Emergency halt thresholds (will refuse new tasks)
configs/resource_limits.yaml:101:  halt_thresholds:
configs/resource_limits.yaml:114:  pressure_cpu_threshold: 75
configs/resource_limits.yaml:117:  pressure_memory_threshold: 70

--- coordination ---
coordination/task_manager.py:17:    BLOCKED = "blocked"
coordination/STRATEGY_REPORT.md:12:- 2026-02-16 01:00-02:00 UTC: ML model overfitting failures (Sharpe 0.999 → -1.48 on holdout)
coordination/STRATEGY_REPORT.md:13:- 2026-02-16 03:50 UTC: Feature expansion experiment (on-chain features FAILED, -0.008 AUC)
coordination/STRATEGY_REPORT.md:62:Static predictions fail when regime shifts
coordination/STRATEGY_REPORT.md:77:Static ensemble models "fail to adapt to evolving financial conditions"
coordination/STRATEGY_REPORT.md:83:Fixed model can't adapt when Fed policy shifts, ETF approvals, etc.
coordination/STRATEGY_REPORT.md:91:"Most AI trading models fail not from weak algorithms but from incomplete data"
coordination/STRATEGY_REPORT.md:115:Why: Research shows static models fail on non-stationary crypto markets
coordination/STRATEGY_REPORT.md:181:If AUC < 0.55: STOP, reassess
coordination/STRATEGY_REPORT.md:197:If not: Analyze failure modes, consider Phase 3 (rolling retraining)
coordination/STRATEGY_REPORT.md:210:Why Models Fail:
coordination/README.md:131:    body="Shuffled-label test failed with 87% accuracy. New features leak future data.",
coordination/README.md:213:- `BLOCKED` - Task blocked by dependency
coordination/README.md:217:- `CRITICAL` - Urgent, blocks other work
coordination/CEO_INBOX.md:31:IMPACT: All Option 3 results are INVALID. Combined with data snooping, Phase 3 validation has completely failed.
coordination/CEO_INBOX.md:41:- [2026-02-16] **CRITICAL: Altcoin Data Fetch Failed - Only 30 Days** (from oversight)
coordination/TASK_CONTRACTS.md:25:- Phase 1 produces AUC <0.50 (catastrophic failure, below random)
coordination/TASK_CONTRACTS.md:35:**Escalation Protocol**:
coordination/TASK_CONTRACTS.md:37:- If CEO persists → RBM escalates to HUMAN (AK)
coordination/TASK_CONTRACTS.md:50:- Combined approach achieves Sharpe <0.85 → Contract fulfilled, escalate to RBM
coordination/TASK_CONTRACTS.md:88:2. **Contract violations trigger escalation** - Human (AK) notified immediately
coordination/TASK_CONTRACTS.md:91:5. **Deliverables are blocking** - Cannot mark contract complete without all deliverables
coordination/CHECKPOINT_SYSTEM.md:30:**If checkpoint FAILS**:
coordination/CHECKPOINT_SYSTEM.md:33:- **Major violation** (e.g., broke contract, pivoted completely): ESCALATE TO HUMAN (AK)
coordination/CHECKPOINT_SYSTEM.md:88:- STOP current work
coordination/CHECKPOINT_SYSTEM.md:93:This is WARNING #[N]. After 3 warnings, contract breach escalated to HUMAN.
coordination/CHECKPOINT_SYSTEM.md:98:### Escalation to Human (Major Violation)


=== 11. EXPERIMENT INVENTORY ===
Scripts:
scripts/backtest_aggregated_signals.py
scripts/backtest_donchian_enhanced.py
scripts/backtest_regime_aware.py
scripts/backtest_simple_baselines.py
scripts/check_onchain_metrics.py
scripts/compare_horizons.py
scripts/data_quality_summary.py
scripts/debug_leakage.py
scripts/feature_selection_1h.py
scripts/fetch_avax_matic_deep.py
scripts/fetch_cross_asset_hourly.py
scripts/fetch_data.py
scripts/fetch_historical_btc_extended.py
scripts/fetch_hourly_btc.py
scripts/fetch_hourly_max_coverage.py
scripts/fetch_onchain_metrics.py
scripts/fetch_remaining_assets.py
scripts/ml_cross_asset_alpha.py
scripts/monitor_ceo_agent.py
scripts/multiseed_stability_1h.py
scripts/option2_debug_overfitting.py
scripts/option2_debug_simple.py
scripts/option3_strategic_pivot.py
scripts/phase_2d_hybrid_strategy.py
scripts/phase_2e_rapid_explorations.py
scripts/prepare_cross_asset_features.py
scripts/prepare_hourly_features.py
scripts/prepare_hourly_training_data.py
scripts/prepare_macro_features.py
scripts/prepare_onchain_features.py
scripts/prepare_phase3_data.py
scripts/progress_report.py
scripts/refetch_incomplete_assets.py
scripts/revalidate_with_block_bootstrap.py
scripts/run_baseline.py
scripts/run_comprehensive_experiments.py
scripts/run_phase1_baselines.py
scripts/run_phase3_experiments.py
scripts/run_phase4_multiseed.py
scripts/test_regime_position_sizing.py
scripts/test_simplified_model.py
scripts/test_trend_aware_regime.py
scripts/train_catboost_1h.py
scripts/train_cross_asset_pooled.py
scripts/train_cross_asset.py
scripts/train_expanded_features_1h.py
scripts/train_hourly_horizon.py
scripts/train_lightgbm_1h.py
scripts/train_on_hourly.py
scripts/transaction_cost_sensitivity.py
scripts/unified_strategy_validation.py
scripts/validate_cross_asset_data.py
scripts/validate_data.py
scripts/validate_donchian_rigorous.py
scripts/validate_ensemble_final.py
scripts/validate_holdout_6month.py
scripts/validate_holdout.py
scripts/validate_kelly_criterion.py
scripts/validate_leakage_reaudit.py
scripts/validate_regime_approaches.py
scripts/validate_regime_filtered.py
scripts/validate_sanity_checks.py
scripts/validate_walkforward_ensemble.py
scripts/walk_forward_1h.py
scripts/yearly_strategy_validation.py

Model files:
regime_weighted_ensemble.py
regime_adaptive_lookback.py
baselines.py
regime_hmm.py
simple_baselines.py
regime_markov_switching.py
__init__.py
xgboost_model.py
regime_volatility_term_structure.py
regime_filtered_donchian.py
lstm_model.py
signal_aggregator.py

Data:
-rw-rw-r-- 1 akamath akamath 918K Feb 15 21:45 data/processed/feature_matrix_btc_hourly.parquet
-rw-rw-r-- 1 akamath akamath 130K Feb 15 20:20 data/processed/feature_matrix_btc.parquet
-rw-rw-r-- 1 akamath akamath  23M Feb 15 22:17 data/processed/features_hourly_full.parquet
-rw-rw-r-- 1 akamath akamath 1.4M Feb 15 22:42 data/processed/macro_features_hourly.parquet
-rw-rw-r-- 1 akamath akamath 1.4M Feb 15 22:45 data/processed/onchain_features_hourly.parquet
-rw-rw-r-- 1 akamath akamath  25K Feb 15 20:20 data/processed/targets_btc_14d.parquet
-rw-rw-r-- 1 akamath akamath  25K Feb 15 20:20 data/processed/targets_btc_1d.parquet
-rw-rw-r-- 1 akamath akamath  25K Feb 15 20:20 data/processed/targets_btc_30d.parquet
-rw-rw-r-- 1 akamath akamath  25K Feb 15 20:20 data/processed/targets_btc_3d.parquet
-rw-rw-r-- 1 akamath akamath  25K Feb 15 20:20 data/processed/targets_btc_7d.parquet
-rw-rw-r-- 1 akamath akamath  45K Feb 15 21:45 data/processed/targets_btc_hourly_1d.parquet
-rw-rw-r-- 1 akamath akamath 1.1M Feb 15 22:17 data/processed/targets_hourly_1h.parquet
-rw-rw-r-- 1 akamath akamath 1.1M Feb 15 22:17 data/processed/targets_hourly_24h.parquet
-rw-rw-r-- 1 akamath akamath 1.1M Feb 15 22:17 data/processed/targets_hourly_4h.parquet
-rw-rw-r-- 1 akamath akamath 1.1M Feb 15 22:17 data/processed/targets_hourly_exec24h.parquet
  data/raw/ada/: 1 files, 2.3M
  data/raw/avax/: 1 files, 32K
  data/raw/btc/: 4 files, 14M
  data/raw/dot/: 1 files, 1.4M
  data/raw/eth/: 2 files, 3.4M
  data/raw/link/: 1 files, 2.3M
  data/raw/macro/: 4 files, 340K
  data/raw/matic/: 1 files, 48K
  data/raw/onchain/: 3 files, 484K
  data/raw/sol/: 1 files, 1.2M


=== 12. GIT TIMELINE ===
8a350c7 2026-02-16 11:24:00 -0500 feat: regime-aware trading strategies — BREAKTHROUGH (Sharpe 2.656)
8567fa3 2026-02-16 11:19:09 -0500 final
f904cbf 2026-02-16 10:50:31 -0500 feat: ML cross-asset training and yearly validation — failed to beat baseline
4ed33c7 2026-02-16 10:45:37 -0500 feat: Kelly Criterion position sizing validation
d014820 2026-02-16 06:21:05 -0500 docs: autonomous execution summary — DAY 0-2 complete, escalated to RBM
feeae44 2026-02-16 06:19:22 -0500 test(validation): regime-filtered FAILED WORSE (-0.350 Sharpe) — ESCALATING to RBM
75df9f4 2026-02-16 06:14:37 -0500 test(validation): walk-forward FAILED — Multi-Timeframe not robust (Sharpe 0.365)
3f6b9fd 2026-02-16 06:11:20 -0500 feat(validation): implement block bootstrap Monte Carlo — honest 78.9% win rate
161332b 2026-02-16 01:37:53 -0500 feat: Phase 2 complete — Multi-Timeframe Ensemble deployed to paper trading
b724933 2026-02-16 00:18:23 -0500 feat: Phase 1 cross-asset pooled training — marginal improvement, pivot to regime-aware strategy
aff1fbc 2026-02-15 23:52:53 -0500 feat: resource management system to prevent system crashes
eee052a 2026-02-15 22:48:51 -0500 feat: macro + on-chain feature engineering pipelines
1c8a86a 2026-02-15 22:46:11 -0500 feat: model comparison + multi-seed stability — CatBoost leads with AUC 0.561
050171d 2026-02-15 22:40:44 -0500 feat: signal aggregation + cross-asset fetch fixes + on-chain data pipeline
1cbc25c 2026-02-15 22:29:02 -0500 data: 1h holdout evaluation — AUC 0.537 confirms genuine weak signal
1684b00 2026-02-15 22:22:47 -0500 feat: multi-horizon hourly training experiment — 1h wins with AUC 0.555
aecc2e8 2026-02-15 21:44:27 -0500 docs: acknowledge look-ahead bias bug + fix roadmap structure
3c0e571 2026-02-15 21:41:23 -0500 refactor: number roadmap files for logical organization
6dd0cff 2026-02-15 21:37:39 -0500 docs: audit response documentation + exchange failover fix
6c37bc5 2026-02-15 21:32:15 -0500 feat: comprehensive hourly feature engineering — 25+ high-quality features
abc5b54 2026-02-15 21:27:16 -0500 feat: data expansion plan — 10,000+ observations via 3 approaches
c3e0ced 2026-02-15 21:08:29 -0500 CRITICAL: Data snooping detected - revert validation claims
d023224 2026-02-15 21:04:32 -0500 docs: comprehensive final summary — simple momentum wins
44dd01f 2026-02-15 21:02:57 -0500 feat: Options 1-3 complete — Simple momentum WINS (Sharpe 2.56)
a10452c 2026-02-15 20:55:47 -0500 feat: Phase 3-4 validation — CRITICAL FINDINGS (result invalidated)
eccc47d 2026-02-15 20:39:11 -0500 feat: Phase 4 multi-seed validation script
084b916 2026-02-15 20:37:59 -0500 feat: Phase 2-3 complete — Alpha detected (Sharpe 0.999)
b660e4d 2026-02-15 20:33:11 -0500 feat: comprehensive Phase 2-3 experiments (15 combinations)
ec7036b 2026-02-15 20:25:02 -0500 feat: Phase 1 baseline re-validation with clean data
e9d8ba7 2026-02-15 20:21:36 -0500 docs: update CRITICAL_FINDINGS with resolution status
701cb8b 2026-02-15 20:21:21 -0500 fix: resolve data leakage by removing returns_1d feature
87e62ac 2026-02-15 20:02:01 -0500 feat: horizon experiments + CRITICAL leakage finding
585e10c 2026-02-15 20:00:39 -0500 feat: feature ablation experiments for Phase 3
f06def6 2026-02-15 19:57:49 -0500 feat: LSTM sequence model with PyTorch
2a76f50 2026-02-15 19:55:39 -0500 feat: XGBoost model with ModelProtocol implementation
f6c38b6 2026-02-15 19:54:04 -0500 data: feature matrices and targets for Phase 3 experiments
0a3fa4d 2026-02-15 19:17:10 -0500 feat: add baseline experiment, fetch real data, document results for Phase 3 handoff
aad61ec 2026-02-15 18:49:09 -0500 test: add integration testing infrastructure and phase 0-2 integration tests
f0be6a3 2026-02-15 18:34:30 -0500 fix: Phase 2 code review — 3 critical, 3 significant, 4 hygiene fixes
9ddbcaf 2026-02-15 18:12:20 -0500 feat: Phase 2 — feature engineering, backtesting, and baseline strategies
e48aaeb 2026-02-15 17:52:26 -0500 feat: enforce BGeometrics rate limit strategy and cache-first policy
56a144d 2026-02-15 18:30:32 -0500 Merge pull request #4 from datadexer/fix/phase1-review-cleanup
eb5f82a 2026-02-15 18:26:11 -0500 fix: Phase 1 code review cleanup — 7 critical + 5 hygiene fixes
53a70db 2026-02-15 17:52:25 -0500 Merge pull request #2 from datadexer/phase-1/data-layer
2c8ae97 2026-02-15 17:48:48 -0500 feat: Phase 1 data layer — fetchers, storage, quality checks, source selector
76608ca 2026-02-15 17:25:34 -0500 Merge pull request #1 from datadexer/phase-0/validation-bedrock
3760850 2026-02-15 17:23:45 -0500 ci: lower coverage threshold to 40% during bootstrap
6ac25b2 2026-02-15 17:16:08 -0500 feat: bootstrap project structure + Phase 0 validation bedrock
f18d7cb 2026-02-15 16:22:02 -0500 first commit

Commits per day:
     39 2026-02-15
     10 2026-02-16


=== 13. SETTINGS ===
{
  "permissions": {
    "allow": [
      "Bash(find:*)",
      "Bash(gh auth:*)",
      "Bash(git -C /home/akamath/sparky-ai log --all --oneline --graph --decorate)",
      "Bash(git -C /home/akamath/sparky-ai branch -a)",
      "Bash(gh:*)",
      "Bash(test:*)",
      "Bash(git -C /home/akamath/sparky-ai show HEAD --stat:*)",
      "Bash(git -C /home/akamath/sparky-ai remote:*)",
      "Bash(pip3 list:*)",
      "Bash(source:*)",
      "Bash(pytest:*)",
      "Bash(python3:*)",
      "Bash(python:*)",
      "Bash(xargs:*)",
      "Bash(chmod:*)",
      "Bash(ls:*)",
      "Bash(wc:*)",
      "Bash(ln:*)",
      "Bash(grep:*)",
      "Bash(PYTHONPATH=/home/akamath/sparky-ai python3:*)",
      "Bash(PYTHONPATH=/home/akamath/sparky-ai bash -c:*)",
      "WebFetch(domain:docs.anthropic.com)",
      "Bash(git -C /home/akamath/sparky-ai log --oneline -10)",
      "Bash(git mv:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(PYTHONPATH=/home/akamath/sparky-ai /home/akamath/sparky-ai/.venv/bin/python:*)",
      "Bash(uv pip install:*)",
      "Bash(PYTHONPATH=/home/akamath/sparky-ai source:*)"
    ]
  }
}

=== SCAN COMPLETE ===
