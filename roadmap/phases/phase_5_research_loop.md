# Phase 5: Autonomous Research Loop

## Purpose
Build a self-improving research system with an inner loop (run experiments,
evaluate results) and an outer loop (propose new hypotheses, expand the
search space). Human stays in the loop via weekly reports and approval gates.

## Tasks

| Task | Description |
|------|-------------|
| `experiment_proposer` | LLM-powered module that generates new experiment configs from past results |
| `automated_evaluation` | Auto-score experiments, rank by risk-adjusted performance, flag anomalies |
| `weekly_report_generator` | Summarize week's experiments, top findings, and proposed next steps |
| `continuous_research_daemon` | Orchestrator that runs inner/outer loop on a schedule with resource limits |

## Completion Criteria
- Experiment proposer generates valid, non-duplicate experiment configs
- Automated evaluation correctly ranks experiments and catches overfitting
- Weekly report is clear enough for human to make approve/reject decisions
- Daemon respects compute budgets and stops gracefully on failure
- At least one full inner/outer loop cycle completes end-to-end

## Human Gate
**Type: Weekly Review**
Human reviews weekly report, approves or rejects proposed experiments,
and can pause/redirect the research loop at any time.
