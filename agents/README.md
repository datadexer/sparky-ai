# Sparky AI Multi-Agent System

**Version**: 1.0
**Created**: 2026-02-16
**Status**: Production Ready

---

## Overview

This directory contains the multi-agent coordination system for Sparky AI, an ML trading project. The system enables multiple AI agents to collaborate on complex tasks while preventing conflicts, duplicate work, and maintaining full audit trail.

---

## Quick Start

### For CEO Agent

**Every session MUST start with**:
```bash
# Read the startup checklist
cat /home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md

# Then follow all 7 steps before starting work
```

**Critical files to check**:
1. `/home/akamath/sparky-ai/roadmap/CEO_INBOX.md` - Messages from sub-agents
2. `/home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md` - Task tracking
3. `/home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md` - Blocking validations

### For Sub-Agents

**On spawn**:
1. Read `/home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md`
2. Find your task assignment
3. Execute work as specified
4. Report to `/home/akamath/sparky-ai/roadmap/CEO_INBOX.md`
5. Terminate

---

## System Architecture

### Agent Types

```
┌─────────────────────────────────────────────────────┐
│                    CEO AGENT                        │
│           (Primary Orchestrator)                    │
│  - Execute roadmap                                  │
│  - Coordinate sub-agents                            │
│  - Make strategic decisions                         │
│  - ONLY ONE CEO EXISTS AT A TIME                    │
└───────────┬──────────────────────────┬──────────────┘
            │                          │
            │                          │
    ┌───────▼────────┐        ┌────────▼───────┐
    │  VALIDATION    │        │ DATA ENGINEER  │
    │   SUB-AGENT    │        │   SUB-AGENT    │
    │                │        │                │
    │ - Audit work   │        │ - Collect data │
    │ - Find errors  │        │ - Process data │
    │ - Report       │        │ - Quality      │
    └────────────────┘        └────────────────┘
    (On-demand)               (On-demand)
```

### Communication Flow

```
Human
  │
  ├─→ VALIDATION_DIRECTIVE.md ──→ CEO reads
  │
  └─→ CEO_INBOX.md ──→ CEO reads

Sub-Agent
  │
  └─→ CEO_INBOX.md ──→ CEO reads

CEO
  │
  ├─→ TASK_ASSIGNMENTS.md ──→ All agents read
  │
  └─→ RESEARCH_LOG.md ──→ All agents read
```

**Key Principle**: All communication goes through files, not side channels.

---

## File Structure

```
/home/akamath/sparky-ai/
├── agents/                          ← Agent definitions (YOU ARE HERE)
│   ├── README.md                    ← This file
│   ├── CEO_AGENT.md                 ← CEO role and responsibilities
│   ├── VALIDATION_AGENT.md          ← Validation sub-agent definition
│   ├── DATA_ENGINEER_AGENT.md       ← Data engineer template (future)
│   ├── COORDINATION_PROTOCOL.md     ← How agents coordinate
│   └── CEO_STARTUP_CHECKLIST.md     ← Mandatory CEO session startup
│
├── roadmap/                         ← Coordination files
│   ├── CEO_INBOX.md                 ← Messages to CEO
│   ├── TASK_ASSIGNMENTS.md          ← Task tracking (who's doing what)
│   ├── VALIDATION_DIRECTIVE.md      ← Blocking requirements
│   ├── RESEARCH_LOG.md              ← Experiment findings
│   └── phases/*.md                  ← Roadmap phase definitions
│
└── logs/
    └── agent_activity/              ← Activity logs
        ├── ceo_YYYY-MM-DD.jsonl
        ├── validation_YYYY-MM-DD.jsonl
        └── data_engineer_YYYY-MM-DD.jsonl
```

---

## Agent Definitions

### 1. CEO Agent

**File**: `CEO_AGENT.md`

**Role**: Primary orchestrator for all ML experiments and roadmap execution

**Unique Constraint**: ONLY ONE CEO AGENT EXISTS AT A TIME

**Key Responsibilities**:
- Execute roadmap phases sequentially
- Coordinate sub-agents
- Make strategic decisions
- Ensure scientific rigor

**Must Read on Every Session Start**:
- `CEO_INBOX.md` - Check for messages
- `TASK_ASSIGNMENTS.md` - Prevent duplicate work
- `VALIDATION_DIRECTIVE.md` - Check for blocking requirements
- `CEO_STARTUP_CHECKLIST.md` - Follow startup procedure

**Activity Log**: `/home/akamath/sparky-ai/logs/agent_activity/ceo_YYYY-MM-DD.jsonl`

### 2. Validation Sub-Agent

**File**: `VALIDATION_AGENT.md`

**Role**: Audit validation results, check for errors/inconsistencies

**Lifecycle**: On-demand (spawned → work → report → terminate)

**Key Responsibilities**:
- Review experimental results
- Identify data leakage, overfitting, bugs
- Suggest fixes
- Write audit reports

**Reports To**: CEO via `CEO_INBOX.md`

**Activity Log**: `/home/akamath/sparky-ai/logs/agent_activity/validation_YYYY-MM-DD.jsonl`

### 3. Data Engineer Sub-Agent (Template)

**File**: `DATA_ENGINEER_AGENT.md`

**Role**: Data collection, feature engineering, data quality

**Status**: Template for future use (not yet active)

**Key Responsibilities**:
- Gather data from external sources
- Clean and transform data
- Ensure data quality
- Create features (no leakage)

**Reports To**: CEO via `CEO_INBOX.md`

**Activity Log**: `/home/akamath/sparky-ai/logs/agent_activity/data_engineer_YYYY-MM-DD.jsonl`

---

## Coordination Files

### CEO_INBOX.md

**Purpose**: Central communication hub for messages to CEO

**Written By**: Sub-agents, humans, system

**Read By**: CEO agent (at every session start)

**Structure**:
- Unread Messages (requires action)
- Read Messages (archived)

**Message Format**:
```markdown
### [YYYY-MM-DD HH:MM] From: [agent-id]
**Subject**: [Brief subject]
**Priority**: CRITICAL | HIGH | MEDIUM | LOW
**Type**: Report | Alert | Directive | Question

**Message**: [Details]
**Action Required**: [Specific actions]
**Signed**: [agent-id]
```

### TASK_ASSIGNMENTS.md

**Purpose**: Track who is working on what to prevent duplicate work

**Written By**: CEO, sub-agents

**Read By**: All agents

**Structure**:
- Currently Active (tasks in progress)
- Queued Tasks (waiting to start)
- Completed Tasks (audit trail)

**Critical Rule**: Check this file BEFORE starting any work

### VALIDATION_DIRECTIVE.md

**Purpose**: Define blocking validation requirements

**Written By**: Humans (project stakeholders)

**Read By**: CEO agent (at every session start)

**Statuses**:
- BLOCKING: Must validate before proceeding
- PASS: Validations complete, proceed normally

**Critical Rule**: Never skip BLOCKING validations

---

## Coordination Protocol

**File**: `COORDINATION_PROTOCOL.md`

**Key Principles**:

1. **CEO is the orchestrator**: All coordination flows through CEO
2. **Explicit over implicit**: Write it down, don't assume
3. **Single source of truth**: TASK_ASSIGNMENTS.md owns task state
4. **Audit trail**: Log everything for reproducibility
5. **Fail-safe**: When in doubt, ask CEO

**Conflict Prevention**:
- Check TASK_ASSIGNMENTS.md before starting work
- Update status to IN PROGRESS immediately
- One task = one agent at a time
- Use git for version control

**Handoff Protocol**:
- CEO creates clear task assignment
- Sub-agent reads assignment
- Sub-agent executes work
- Sub-agent reports to CEO_INBOX.md
- Sub-agent terminates
- CEO reviews report and acts

---

## Common Workflows

### Workflow 1: CEO Solo Work

```
1. Read CEO_STARTUP_CHECKLIST.md
2. Follow all 7 startup steps
3. Check TASK_ASSIGNMENTS.md (no conflicts)
4. Update TASK_ASSIGNMENTS.md (mark IN PROGRESS)
5. Log task_started
6. Execute work
7. Log task_completed
8. Update TASK_ASSIGNMENTS.md (mark COMPLETED)
```

### Workflow 2: CEO Spawns Validation Sub-Agent

```
CEO:
1. Identify need for validation (e.g., suspicious results)
2. Create task assignment in TASK_ASSIGNMENTS.md
3. Spawn validation sub-agent with agent ID

Validation Agent:
4. Read TASK_ASSIGNMENTS.md for assignment
5. Update status to IN PROGRESS
6. Review materials (code, results, data)
7. Identify issues (prioritize: CRITICAL, HIGH, MEDIUM, LOW)
8. Write audit report to CEO_INBOX.md
9. Update TASK_ASSIGNMENTS.md to COMPLETED
10. Terminate

CEO (next session):
11. Read CEO_INBOX.md
12. See validation report
13. Review findings
14. Act on recommendations (debug or proceed)
15. Mark message as read, archive
```

### Workflow 3: Validation Blocking Gate

```
1. Human writes VALIDATION_DIRECTIVE.md with BLOCKING status
2. CEO reads directive at session start (Step 3 of checklist)
3. CEO STOPS planned work
4. CEO executes required validations (in order)
5. CEO reports validation results
6. If validations PASS:
   - Human updates VALIDATION_DIRECTIVE.md to PASS
   - CEO proceeds to next phase
7. If validations FAIL:
   - CEO debugs root cause
   - CEO fixes issues
   - CEO re-runs validations (back to step 4)
```

---

## Activity Logging

All agents log activities to JSONL files:

**Location**: `/home/akamath/sparky-ai/logs/agent_activity/`

**Naming**: `{agent_type}_YYYY-MM-DD.jsonl`

**Format**:
```json
{
  "timestamp": "2026-02-16T10:30:00.000000+00:00",
  "agent_id": "ceo",
  "session_id": "phase-4-validation",
  "action_type": "task_started|task_completed|experiment_completed",
  "task": "task_identifier",
  "description": "Human-readable description",
  "result": {},
  "files_changed": [],
  "git_commit": "sha"
}
```

**Why Log**:
- Reproducibility (recreate what happened)
- Debugging (understand errors)
- Audit trail (prove scientific rigor)
- Context (future agents understand history)

---

## Best Practices

### For All Agents

1. **Read before write**: Always check state before changing
2. **Log everything**: Document all actions
3. **Be explicit**: Write it down, don't assume
4. **Check coordination files**: Stay aligned
5. **Commit frequently**: Track changes in git

### For CEO Agent

1. **Follow startup checklist**: Every session, no exceptions
2. **Read inbox first**: Check for critical messages
3. **Prevent duplicates**: Check TASK_ASSIGNMENTS.md before starting
4. **Never skip validations**: Respect BLOCKING status
5. **Review sub-agent reports**: Act on findings

### For Sub-Agents

1. **Stay in scope**: Do assigned task, nothing more
2. **Report clearly**: Use message templates
3. **Be independent**: Critical analysis, not rubber-stamping
4. **Meet deadlines**: Respect time estimates
5. **Terminate cleanly**: Complete logging before exit

---

## Anti-Patterns to Avoid

### ❌ Starting work without reading inbox
**Why**: Miss critical messages, ignore blocking validations
**Fix**: Always read CEO_INBOX.md first

### ❌ Not checking TASK_ASSIGNMENTS.md
**Why**: Duplicate work, waste time
**Fix**: Always check before starting

### ❌ Skipping validation gates
**Why**: Proceed on invalid results, scientific rigor lost
**Fix**: Respect BLOCKING status, never skip

### ❌ Poor communication
**Why**: Context lost, coordination breaks down
**Fix**: Use message templates, be specific

### ❌ Not logging activities
**Why**: Lost reproducibility, can't debug
**Fix**: Log everything in real-time

---

## Troubleshooting

### Problem: Task already being done

**Symptom**: Started task that's IN PROGRESS by another agent

**Fix**:
1. Stop work immediately
2. Check TASK_ASSIGNMENTS.md
3. If stale (>24h) → May be abandoned
4. If recent → Pick different task
5. Learn: Always check first

### Problem: Missed critical inbox message

**Symptom**: Proceeded without reading sub-agent report

**Fix**:
1. Stop current work
2. Read CEO_INBOX.md now
3. Address critical messages
4. Adjust plan based on findings
5. Learn: Read inbox at session start

### Problem: Validation blocking forgotten

**Symptom**: Proceeded past BLOCKING validation directive

**Fix**:
1. Stop immediately
2. Read VALIDATION_DIRECTIVE.md
3. Execute required validations
4. Recent work may be invalidated
5. Learn: Check directive every session

---

## Adding New Agent Types

To add a new agent type (e.g., Research Agent):

1. **Create agent definition**:
   - Copy `DATA_ENGINEER_AGENT.md` as template
   - Rename to `RESEARCH_AGENT.md`
   - Define role, responsibilities, lifecycle

2. **Update coordination files**:
   - Add agent type to this README
   - Add to COORDINATION_PROTOCOL.md if needed

3. **Create activity log path**:
   ```bash
   /home/akamath/sparky-ai/logs/agent_activity/research_YYYY-MM-DD.jsonl
   ```

4. **Test workflow**:
   - CEO spawns research agent
   - Research agent executes task
   - Research agent reports to CEO_INBOX.md
   - CEO reviews and acts

---

## System Health Checks

Periodically verify the system is working:

### Daily Checks
- [ ] CEO reads inbox at session start
- [ ] TASK_ASSIGNMENTS.md updated in real-time
- [ ] Activity logs written for all work
- [ ] No stale IN PROGRESS tasks (>24h)

### Weekly Checks
- [ ] Archive old read messages in CEO_INBOX.md
- [ ] Review completed tasks for patterns
- [ ] Check for process improvements
- [ ] Verify git commits match activity logs

### Monthly Checks
- [ ] Review agent definitions for updates needed
- [ ] Analyze coordination effectiveness
- [ ] Update templates based on learnings
- [ ] Train new agent types if needed

---

## Success Metrics

The multi-agent system is working well if:

- ✅ Zero duplicate work
- ✅ Zero missed critical messages
- ✅ Zero skipped validation gates
- ✅100% task tracking coverage
- ✅ Complete audit trail (all work logged)
- ✅ Fast context switching (new agents get up to speed quickly)
- ✅ High quality sub-agent reports
- ✅ Clear decision rationale

---

## Contact and Support

**Questions about the system**:
- Read agent definition files in this directory
- Read COORDINATION_PROTOCOL.md for detailed workflows
- Write question to CEO_INBOX.md for human review

**Reporting issues**:
- Document in CEO_INBOX.md with CRITICAL priority
- Include steps to reproduce
- Suggest fixes if possible

**Proposing improvements**:
- Write recommendation to CEO_INBOX.md
- Explain problem and proposed solution
- Estimate implementation effort

---

## Version History

- v1.0 (2026-02-16): Initial multi-agent system
  - CEO agent definition
  - Validation agent definition
  - Data engineer template
  - Coordination protocol
  - CEO startup checklist
  - Task tracking system
  - Inbox communication system

---

## Quick Reference

**CEO must read on every session start**:
1. `/home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md`
2. `/home/akamath/sparky-ai/roadmap/CEO_INBOX.md`
3. `/home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md`
4. `/home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md`

**All agents should read**:
- This README for system overview
- Their agent definition file for role
- COORDINATION_PROTOCOL.md for workflows

**When spawning sub-agent**:
1. Create task in TASK_ASSIGNMENTS.md
2. Provide clear scope and deliverables
3. Specify report location (CEO_INBOX.md)

**When completing work**:
1. Log task_completed
2. Update TASK_ASSIGNMENTS.md
3. Report to CEO_INBOX.md (if sub-agent)
4. Commit code with clear message

---

**This system is production-ready. Follow the protocols and the agents will coordinate effectively.**
