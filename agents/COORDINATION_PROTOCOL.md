# Multi-Agent Coordination Protocol

**Version**: 1.0
**Last Updated**: 2026-02-16
**Purpose**: Define how multiple agents coordinate to prevent conflicts and ensure efficient collaboration

---

## Table of Contents

1. [Overview](#overview)
2. [Agent Roles](#agent-roles)
3. [Communication Channels](#communication-channels)
4. [Work Assignment Flow](#work-assignment-flow)
5. [Conflict Prevention](#conflict-prevention)
6. [Handoff Protocol](#handoff-protocol)
7. [Activity Logging](#activity-logging)
8. [Emergency Procedures](#emergency-procedures)

---

## Overview

### The Challenge

With multiple agents working on the Sparky AI project, we need clear coordination to prevent:
- **Duplicate work**: Two agents working on the same task
- **Conflicts**: Agents making contradictory changes
- **Lost context**: Work happening without proper documentation
- **Miscommunication**: Agents not aware of each other's findings

### The Solution

A structured coordination system with:
- **Clear ownership**: One task = one agent at a time
- **Central communication**: All coordination via CEO agent
- **State tracking**: TASK_ASSIGNMENTS.md shows who's doing what
- **Activity logging**: All work recorded in JSONL logs
- **Inbox system**: CEO_INBOX.md for reports and alerts

### Design Principles

1. **CEO is the orchestrator**: All coordination flows through CEO
2. **Explicit over implicit**: Write it down, don't assume
3. **Single source of truth**: TASK_ASSIGNMENTS.md owns task state
4. **Audit trail**: Log everything for reproducibility
5. **Fail-safe**: When in doubt, ask CEO

---

## Agent Roles

### CEO Agent (Primary Orchestrator)

**Responsibilities**:
- Execute roadmap
- Assign tasks to sub-agents
- Review reports
- Make strategic decisions
- Prevent duplicate work

**Unique Constraint**: Only ONE CEO agent exists at a time

**Communication**:
- Reads: CEO_INBOX.md (from sub-agents)
- Writes: TASK_ASSIGNMENTS.md (to sub-agents)

### Sub-Agents (On-Demand Workers)

**Types**:
- Validation agents (audit work)
- Data engineer agents (collect/process data)
- Research agents (literature review, method exploration)

**Lifecycle**: Spawned → Work → Report → Terminate

**Communication**:
- Reads: TASK_ASSIGNMENTS.md (from CEO)
- Writes: CEO_INBOX.md (to CEO)

**No Direct Communication**: Sub-agents do NOT coordinate with each other directly

---

## Communication Channels

### File-Based Communication

All agent communication happens through files (no other channels):

| File | Purpose | Written By | Read By |
|------|---------|------------|---------|
| CEO_INBOX.md | Reports to CEO | Sub-agents, Humans | CEO |
| TASK_ASSIGNMENTS.md | Task tracking | CEO, Sub-agents | All agents |
| VALIDATION_DIRECTIVE.md | Blocking requirements | Humans | CEO |
| RESEARCH_LOG.md | Experiment findings | CEO | All agents |
| Activity logs (*.jsonl) | Audit trail | All agents | All agents |

### Message Flow

```
Human → VALIDATION_DIRECTIVE.md → CEO reads
Human → CEO_INBOX.md → CEO reads
Sub-Agent → CEO_INBOX.md → CEO reads
CEO → TASK_ASSIGNMENTS.md → Sub-Agent reads
CEO → RESEARCH_LOG.md → All read
```

### No Side Channels

**NOT ALLOWED**:
- Sub-agents messaging each other directly
- Out-of-band communication (Slack, email, etc.)
- Undocumented coordination
- Verbal agreements

**WHY**: All communication must be in files for:
- Reproducibility
- Audit trail
- Future agent awareness
- Human oversight

---

## Work Assignment Flow

### Step 1: CEO Identifies Work

CEO determines a task needs to be done:
- From roadmap phases
- From validation directives
- From sub-agent recommendations
- From experimental findings

### Step 2: CEO Checks for Existing Work

**CRITICAL**: Before assigning, check TASK_ASSIGNMENTS.md:

```python
# Pseudo-code for CEO logic
def before_starting_task(task_name):
    # Read current assignments
    active_tasks = read_task_assignments()

    # Check if anyone is already working on this
    for task in active_tasks['currently_active']:
        if task.name == task_name:
            print(f"CONFLICT: {task.agent} already working on {task_name}")
            return False  # DO NOT START

    # Check if recently completed (might be duplicate)
    for task in active_tasks['completed']:
        if task.name == task_name and task.completed_within_hours(24):
            print(f"WARNING: {task_name} completed {task.completed} ago")
            return False  # Probably don't need to redo

    return True  # Safe to start
```

### Step 3: CEO Creates Task Assignment

For **CEO's own work**:

```python
# Update TASK_ASSIGNMENTS.md
# Move task from "Queued" to "Currently Active"

| Agent | Task | Status | Started | Last Update | ETA |
|-------|------|--------|---------|-------------|-----|
| CEO | Run holdout test | 🔄 IN PROGRESS | 2026-02-16 10:00 | 2026-02-16 10:00 | 10:30 |

# Log activity
log_agent_activity(
    agent_id="ceo",
    action_type="task_started",
    task="holdout_test",
    description="Running holdout test on technical-only 30d model"
)
```

For **sub-agent work**:

```python
# Create detailed task assignment in TASK_ASSIGNMENTS.md

### Task: Audit Phase 3 Results

**Priority**: HIGH
**Assigned To**: validation-sub-agent-001
**Created**: 2026-02-16 10:00 UTC
**Deadline**: 2026-02-16 12:00 UTC
**Estimated Time**: 1h 30m

**Objective**:
Audit PHASE_3_VALIDATION_SUMMARY.md for errors, data leakage, and overfitting.

**Deliverables**:
1. Audit report in CEO_INBOX.md
2. List of issues (prioritized: CRITICAL, HIGH, MEDIUM, LOW)
3. Recommendation: PROCEED or DEBUG

**Pass Criteria**:
- All claimed results verified OR issues clearly documented
- Clear recommendation with rationale

**Resources**:
- /home/akamath/sparky-ai/roadmap/PHASE_3_VALIDATION_SUMMARY.md
- /home/akamath/sparky-ai/results/phase_3/
```

### Step 4: Agent Executes Work

**Sub-agent reads assignment**:
```python
# Read TASK_ASSIGNMENTS.md
# Find task assigned to your agent ID
# Read objective, scope, deliverables, pass criteria

# Update status to IN PROGRESS
# Log task_started
```

**Agent works on task**:
- Follow assigned scope
- Don't deviate without CEO approval
- Log progress periodically
- Document findings

### Step 5: Agent Reports Completion

**Sub-agent writes report**:
```markdown
# In CEO_INBOX.md (Unread Messages section)

### [2026-02-16 11:30] From: validation-sub-agent-001
**Subject**: Phase 3 Audit Complete
**Priority**: HIGH
**Type**: Report

**Summary**: Audited Phase 3 results. Found 5 critical issues.

**Deliverables**:
1. ✅ Audit report (below)
2. ✅ Issues prioritized
3. ✅ Recommendation: 🛑 DEBUG before proceeding

**Critical Issues**:
1. [Issue 1]
2. [Issue 2]
...

**Recommendation**: Do not proceed to multi-seed validation until issues resolved.

**Signed**: validation-sub-agent-001
**Completed**: 2026-02-16 11:30 UTC
```

**Agent updates task status**:
```python
# In TASK_ASSIGNMENTS.md, move from "Currently Active" to "Completed"

| Agent | Task | Completed | Duration | Notes |
|-------|------|-----------|----------|-------|
| validation-sub-agent-001 | Audit Phase 3 results | 2026-02-16 11:30 | 1h 30m | Found 5 critical issues |

# Log task_completed
log_agent_activity(
    agent_id="validation-sub-agent-001",
    action_type="task_completed",
    task="audit_phase3_results",
    files_changed=["roadmap/CEO_INBOX.md", "roadmap/TASK_ASSIGNMENTS.md"]
)
```

**Agent terminates** (sub-agent lifecycle complete)

### Step 6: CEO Reviews Report

**CEO reads inbox**:
```python
# At next session start or during session
# Read CEO_INBOX.md
# See new message from validation-sub-agent-001
# Review findings
# Decide next steps
```

**CEO acts on findings**:
- If issues found: Create tasks to address them
- If validation passes: Proceed to next phase
- If unclear: Spawn another sub-agent for deeper analysis

---

## Conflict Prevention

### Problem: Duplicate Work

**Scenario**: Two agents start the same task simultaneously.

**Prevention**:
1. **Check before start**: Always read TASK_ASSIGNMENTS.md first
2. **Update immediately**: Mark task IN PROGRESS as soon as you start
3. **Single source of truth**: TASK_ASSIGNMENTS.md owns state
4. **Atomic updates**: Update file atomically (not incrementally)

**Example**:
```python
# WRONG: Check and update in separate steps
active_tasks = read_task_assignments()  # Step 1
if task not in active_tasks:
    time.sleep(60)  # Another agent could start here!
    mark_task_in_progress(task)  # Step 2 - TOO LATE

# RIGHT: Check and update atomically
with file_lock("TASK_ASSIGNMENTS.md"):
    active_tasks = read_task_assignments()
    if task not in active_tasks:
        mark_task_in_progress(task)  # Immediate
```

### Problem: Conflicting Changes

**Scenario**: Two agents modify the same file differently.

**Prevention**:
1. **Clear ownership**: One file = one owner at a time
2. **Non-overlapping scopes**: Assign disjoint work to sub-agents
3. **Read-only vs write**: Most agents read, few write
4. **Git for versioning**: Commit frequently, resolve conflicts

**Example**:
```python
# WRONG: Two agents editing RESEARCH_LOG.md simultaneously
agent_1: Append results to RESEARCH_LOG.md
agent_2: Append results to RESEARCH_LOG.md  # CONFLICT

# RIGHT: Clear ownership
CEO: Owns RESEARCH_LOG.md (only writer)
Sub-agents: Write to CEO_INBOX.md (own section)
CEO: Later incorporates findings into RESEARCH_LOG.md
```

### Problem: Lost Context

**Scenario**: Agent B doesn't know what Agent A discovered.

**Prevention**:
1. **Activity logs**: All work logged to JSONL
2. **CEO synthesis**: CEO reads all reports, synthesizes findings
3. **Shared read access**: All agents can read RESEARCH_LOG.md
4. **Task dependencies**: TASK_ASSIGNMENTS.md shows dependency chain

**Example**:
```python
# Agent A discovers leakage issue
validation_agent_001: "Found leakage in 30d model" → CEO_INBOX.md

# CEO reads report, logs to RESEARCH_LOG.md
ceo: Read validation report → Update RESEARCH_LOG.md

# Agent B (later) reads RESEARCH_LOG.md
data_engineer_002: Read RESEARCH_LOG.md → Aware of leakage issue
```

---

## Handoff Protocol

### When to Hand Off

- CEO has too much work (delegate to sub-agent)
- Specialized expertise needed (validation, data engineering)
- Independent review required (audit by validation agent)
- Parallel work possible (multiple data engineers)

### How to Hand Off

**Step 1: Create clear task assignment** (see Work Assignment Flow)

**Step 2: Provide all context**:
```markdown
**Resources**:
- Read these files first: [list]
- Relevant code: [paths]
- Previous work: [references]
```

**Step 3: Set clear deliverables**:
```markdown
**Deliverables**:
1. [Concrete output 1]
2. [Concrete output 2]

**Pass Criteria**:
- [How to know if successful]
```

**Step 4: Specify report location**:
```markdown
**Report To**: CEO_INBOX.md (use message template)
```

### How to Receive Handoff

**Sub-agent**:
1. Read TASK_ASSIGNMENTS.md for your agent ID
2. Read all provided resources
3. Ask questions if unclear (via CEO_INBOX.md)
4. Update status to IN PROGRESS
5. Execute work
6. Report findings to CEO_INBOX.md
7. Update status to COMPLETED
8. Terminate

### Handoff Checklist

**For CEO (handing off)**:
- [ ] Clear task definition written
- [ ] All context provided
- [ ] Deliverables specified
- [ ] Pass criteria defined
- [ ] Deadline set
- [ ] Resources listed
- [ ] Agent ID assigned

**For Sub-Agent (receiving)**:
- [ ] Task assignment read
- [ ] Context understood
- [ ] Resources accessed
- [ ] Status updated to IN PROGRESS
- [ ] Work completed
- [ ] Report written to CEO_INBOX.md
- [ ] Status updated to COMPLETED

---

## Activity Logging

### Why Log Everything

- **Reproducibility**: Recreate what happened
- **Debugging**: Understand what went wrong
- **Audit trail**: Prove scientific rigor
- **Context**: Future agents understand history
- **Metrics**: Measure productivity and quality

### What to Log

**Required for all agents**:
- Task started
- Task completed
- Experiments run
- Decisions made
- Errors encountered

**Format** (JSONL):
```json
{
  "timestamp": "2026-02-16T10:30:00.000000+00:00",
  "agent_id": "ceo",
  "session_id": "phase-4-validation",
  "action_type": "experiment_completed",
  "task": "holdout_test",
  "description": "Ran holdout test on technical-only 30d model",
  "result": {
    "holdout_sharpe": 0.45,
    "train_sharpe": 0.999,
    "interpretation": "Overfitting confirmed - holdout << train"
  },
  "mlflow_run_id": "abc123",
  "files_changed": ["roadmap/RESEARCH_LOG.md"]
}
```

### When to Log

- **Immediately** when action happens (don't batch)
- **Before and after** major actions (task_started, task_completed)
- **On errors** (document failures)
- **On decisions** (record rationale)

### Log Location

```
/home/akamath/sparky-ai/logs/agent_activity/
  ├── ceo_2026-02-16.jsonl
  ├── validation_2026-02-16.jsonl
  ├── data_engineer_2026-02-17.jsonl
  └── ...
```

**Naming**: `{agent_type}_YYYY-MM-DD.jsonl`

### Log Retention

- Keep all logs indefinitely (disk is cheap)
- Critical for reproducibility
- May need for audits or publications

---

## Emergency Procedures

### Agent Stuck or Frozen

**Symptoms**: Agent not responding, task status stale (>24h)

**Action**:
1. CEO checks TASK_ASSIGNMENTS.md
2. See task stuck in IN PROGRESS for 24+ hours
3. Mark task as FAILED in completed tasks
4. Create new task assignment (retry or different approach)
5. Log incident

### Contradictory Findings

**Symptoms**: Agent A says X, Agent B says NOT X

**Action**:
1. CEO reads both reports in CEO_INBOX.md
2. Analyzes evidence from each
3. If unclear, spawn third agent as tiebreaker
4. Makes decision and logs rationale in DECISIONS.md
5. Updates RESEARCH_LOG.md with consensus

### Critical Error Discovered

**Symptoms**: Major bug, data leakage, invalid results

**Action**:
1. Agent discovering error writes CRITICAL priority message to CEO_INBOX.md
2. CEO reads at next session start (or immediately if monitoring)
3. CEO creates VALIDATION_DIRECTIVE.md with BLOCKING status
4. All work stops until issue resolved
5. CEO assigns sub-agent to debug
6. Once fixed, CEO updates VALIDATION_DIRECTIVE.md to PASS
7. Work resumes

### Lost or Corrupted Files

**Symptoms**: File missing, corrupted, or conflicting versions

**Action**:
1. Use Git to recover previous version
2. Check activity logs for who last modified
3. Restore from backup if needed
4. Document incident in logs
5. Add safeguards to prevent recurrence

---

## Best Practices

### For All Agents

1. **Read before write**: Always check state before changing
2. **Update atomically**: Don't leave state inconsistent
3. **Log everything**: Future you will thank you
4. **Be explicit**: Write it down, don't assume
5. **Check inbox/assignments**: Stay coordinated

### For CEO Agent

1. **Read inbox first**: Every session, check CEO_INBOX.md
2. **Prevent duplicates**: Check TASK_ASSIGNMENTS.md before starting
3. **Delegate wisely**: Sub-agents for specialized work
4. **Review thoroughly**: Read all sub-agent reports
5. **Synthesize findings**: Update RESEARCH_LOG.md with insights

### For Sub-Agents

1. **Stay in scope**: Do assigned task, nothing more
2. **Report clearly**: Use message template
3. **Be independent**: Critical analysis, not rubber-stamping
4. **Meet deadlines**: Respect time estimates
5. **Terminate cleanly**: Complete all logging before exit

---

## Common Patterns

### Pattern: CEO Solo Work

```
1. CEO checks CEO_INBOX.md (no new messages)
2. CEO checks TASK_ASSIGNMENTS.md (no conflicts)
3. CEO updates TASK_ASSIGNMENTS.md (mark IN PROGRESS)
4. CEO logs task_started
5. CEO executes work
6. CEO logs task_completed
7. CEO updates TASK_ASSIGNMENTS.md (mark COMPLETED)
```

### Pattern: CEO Spawns Sub-Agent

```
1. CEO identifies need for sub-agent (e.g., audit)
2. CEO creates task assignment in TASK_ASSIGNMENTS.md
3. CEO spawns sub-agent (provides agent ID)
4. Sub-agent reads assignment
5. Sub-agent updates status to IN PROGRESS
6. Sub-agent executes work
7. Sub-agent writes report to CEO_INBOX.md
8. Sub-agent updates status to COMPLETED
9. Sub-agent terminates
10. CEO (next session) reads report from CEO_INBOX.md
11. CEO acts on findings
```

### Pattern: Parallel Sub-Agents

```
1. CEO spawns validation-sub-agent-001 to audit Phase 3
2. CEO spawns data-engineer-001 to collect hourly data
3. Both agents work in parallel (non-overlapping scopes)
4. validation-sub-agent-001 reports to CEO_INBOX.md, terminates
5. data-engineer-001 reports to CEO_INBOX.md, terminates
6. CEO reads both reports
7. CEO synthesizes findings
```

---

## Version History

- v1.0 (2026-02-16): Initial coordination protocol
