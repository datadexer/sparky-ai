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